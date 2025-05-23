import torch
import torch.nn as nn
from torch.nn import functional as F


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        block_size: int,
        channels_embedding: int,
        channels_head: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.channels_head = channels_head
        self.matrix_key = nn.Linear(channels_embedding, channels_head, bias=False)
        self.matrix_query = nn.Linear(channels_embedding, channels_head, bias=False)
        self.matrix_value = nn.Linear(channels_embedding, channels_head, bias=False)
        # Lower triangular
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        k = self.matrix_key(x)
        q = self.matrix_query(x)
        v = self.matrix_value(x)
        # Compute weights (normalize by channel dimensionality so softmax doesn't become too eager)
        weights = q @ k.transpose(-2, -1) * self.channels_head ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # Prevent looking into the future and use '-inf' to ensure attention scores sum to 1
        weights = weights.masked_fill(self.tril[:self.block_size, :self.block_size] == 0, float('-inf'))  # (B, T, T)
        # Compute attention scores (affinities)
        weights = F.softmax(weights, dim=1)
        # Perform weighted aggregation of the values
        return weights @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)


class Transformer(nn.Module):
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        vocabulary_size: int,
        channels_embedding: int,
        channels_head: int,
        device: str
    ):
        """
        nn.Embedding is a thin wrapper of torch.tensor that represents a table of embeddings.
        """
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        self.layer_embedding_token = nn.Embedding(vocabulary_size, channels_embedding)
        self.layer_embedding_position = nn.Embedding(block_size, channels_embedding)
        self.layer_self_attention_head = SelfAttentionHead(block_size, channels_embedding, channels_head)
        self.layer_output = nn.Linear(channels_head, vocabulary_size)

        self.layers_and_learning_rates = {
            'embedding_token': (self.layer_embedding_token, 1),
            'embedding_position': (self.layer_embedding_position, 1),
            'self_attention_head': (self.layer_self_attention_head, 1),
            'output': (self.layer_output, 1),
        }
    
    def forward(self, x: torch.tensor):
        """
        Logits are the scores for the next character in the sequence.
        The correct token should have a high score, while the incorrect tokens should have low scores.
        """
        x_token_emb = self.layer_embedding_token(x)  # shape: (Batch=batch_size, Time=block_size, Channel=vocabulary_size)
        x_position_emb = self.layer_embedding_position(torch.arange(self.block_size, device=self.device))
        
        return self.layer_output(
            self.layer_self_attention_head(x_token_emb + x_position_emb)
        )


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