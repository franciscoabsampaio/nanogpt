from .config import ModuleConfig, ConfigurableModule
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(kw_only=True)
class Config(ModuleConfig):
    batch_size: int = 200
    block_size: int = 10
    channels_embedding: int = 2048
    number_of_heads: int = 4
    number_of_blocks: int = 2
    dropout_rate: float = 0.2
    share_embedding_weights: bool = False


class CausalSelfAttentionHead(nn.Module):
    def __init__(
        self,
        config: Config
    ):
        super().__init__()
        self.block_size = config.block_size
        self.channels_head = config.channels_head
        self.matrix_key = nn.Linear(config.channels_embedding, config.channels_head, bias=False)
        self.matrix_query = nn.Linear(config.channels_embedding, config.channels_head, bias=False)
        self.matrix_value = nn.Linear(config.channels_embedding, config.channels_head, bias=False)
        # Lower triangular
        self.register_buffer('causal_mask', torch.tril(torch.ones(config.block_size, config.block_size)))

        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        k = self.matrix_key(x)
        q = self.matrix_query(x)
        v = self.matrix_value(x)
        # Compute weights (normalize by channel dimensionality so softmax doesn't become too eager)
        weights = q @ k.transpose(-2, -1) * self.config.channels_head ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # Prevent looking into the future and use '-inf' to ensure attention scores sum to 1
        weights = weights.masked_fill(
            self.causal_mask[:self.config.block_size, :self.config.block_size] == 0,
            float('-inf')
        )  # (B, T, T)
        # Compute attention scores (affinities)
        weights = F.softmax(weights, dim=1)
        # Perform weighted aggregation of the values
        return self.dropout(weights) @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)


class MultiHeadSelfAttention(ConfigurableModule):
    def __init__(
        self,
        config: Config,
        # The legacy architecture creates a separate module for each head.
        # This is not as efficient as instantiating a single module
        # and doing some pytorch gymnastics for efficient splitting.
        legacy_architecture: bool = False
    ):
        super().__init__(config)
        assert config.channels_embedding % config.number_of_heads == 0
        self.channels_head = config.channels_embedding // config.number_of_heads

        self.legacy_architecture = legacy_architecture

        if self.legacy_architecture:
            self.heads = nn.ModuleList([
                CausalSelfAttentionHead(config)
                for _ in range(config.number_of_heads)
            ])
        else:
            # Q, K, V projections for all heads, but in a batch
            self.layer_qkv_projected = nn.Linear(config.channels_embedding, 3 * config.channels_embedding)
            # Lower triangular
            self.register_buffer('causal_mask', torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size))
            self.layer_output = nn.Linear(config.channels_embedding, config.channels_embedding)
        
        self.dropout = nn.Dropout(config.dropout_rate)

    def number_of_heads(self) -> int:
        return self.config.number_of_heads

    def forward(self, x):
        if self.legacy_architecture:
            x_projected = torch.cat([h(x) for h in self.heads], dim=-1)
        else:
            B, T, C = x.size()
            qkv = self.layer_qkv_projected(x)
            q, k, v = qkv.split(self.config.channels_embedding, dim=-1)
            # Before the split, qkv has C=3*channels_embedding
            # After the split, q, k, and v have C=channels_embedding each
            # So size(q) = size(k) = size(v) = (B, T, C)
            # can be rearranged to (B, T, n_heads, C/n_heads) through the view method
            # and .transpose(1, 2) to (B, n_heads, T, C/n_heads)
            # such that B and n_heads function as 2-dimensional batch dimensions.
            q = q.view(B, T, self.number_of_heads(), C // self.number_of_heads()).transpose(1, 2)
            k = k.view(B, T, self.number_of_heads(), C // self.number_of_heads()).transpose(1, 2)
            v = v.view(B, T, self.number_of_heads(), C // self.number_of_heads()).transpose(1, 2)
            # Attention (T,T)
            attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attention = attention.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            attention = F.softmax(attention, dim=-1)
            y = attention @ v  # (B, n_heads, T, T) @ (B, n_heads, T, C / n_heads) -> (B, nh, T, C/nh)
            # Reassemble all head outputs into a single (B, T, C) tensor
            y = y.transpose(1, 2).contiguous().view(B, T, C)

            return self.layer_output(y)

        return self.dropout(x_projected)


class FeedForward(nn.Module):
    def __init__(
        self,
        channels_embedding: int,
        dropout_rate: float,
        activation_function: nn.Module = nn.GELU()
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(channels_embedding, 4 * channels_embedding),
            activation_function,
            nn.Linear(4 * channels_embedding, channels_embedding),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(
        self,
        config: Config
    ):
        super().__init__()
        self.layer_multi_head_attention = MultiHeadSelfAttention(config)
        self.layer_feedforward = FeedForward(config.channels_embedding, config.dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(config.channels_embedding)
        self.layer_norm_2 = nn.LayerNorm(config.channels_embedding)


    def forward(self, x):
        x = x + self.layer_multi_head_attention(self.layer_norm_1(x))
        return x + self.layer_feedforward(self.layer_norm_2(x))


class Transformer(ConfigurableModule):
    def __init__(
        self,
        config: Config,
    ):
        """
        nn.Embedding is a thin wrapper of torch.tensor that represents a table of embeddings.
        """
        super().__init__(config)

        self.layer_embedding_token = nn.Embedding(config.vocabulary_size, config.channels_embedding)
        self.layer_embedding_position = nn.Embedding(config.block_size, config.channels_embedding)
        self.blocks = nn.Sequential(*([
            Block(config)
            for _ in range(config.number_of_blocks)
        ] + [nn.LayerNorm(config.channels_embedding)]))
        self.layer_output = nn.Linear(config.channels_embedding, config.vocabulary_size)

        self.layers_and_learning_rates = {
            'embedding_token': (self.layer_embedding_token, 1),
            'embedding_position': (self.layer_embedding_position, 1),
            'blocks': (self.blocks, 1),
            'output': (self.layer_output, 1),
        }

        # Weight sharing by pointing the embedding reference to the output head
        if config.share_embedding_weights:
            self.layer_embedding_token.weight = self.layer_output.weight
            self.layers_and_learning_rates.pop('output')

        # Initialize weights
        self.apply(self._init_weights)  # .apply() runs a function for every child Module

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)

    def forward(self, x: torch.tensor):
        """
        Logits are the scores for the next character in the sequence.
        The correct token should have a high score, while the incorrect tokens should have low scores.
        """
        x_token_emb = self.layer_embedding_token(x)  # shape: (Batch=batch_size, Time=block_size, Channel=vocabulary_size)
        x_position_emb = self.layer_embedding_position(torch.arange(self.block_size, device=self.device))
        
        x_emb = x_token_emb + x_position_emb
        return self.layer_output(self.blocks(x_emb))


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self.forward(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (Batch, Channel)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (Batch, 1)
            # Append the new token to the running sequence
            idx = torch.cat([idx, next_token], dim=1)  # shape: (Batch, Time + 1)
        return idx

    def receptive_field(self) -> int:
        return self.block_size