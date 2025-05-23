import torch
from torch import nn, functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int = 2, dilation: int = 1):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) * self.dilation
        self.padding_layer = nn.ZeroPad1d((self.padding, 0))
        self.conv = nn.Conv1d(
            in_channels, out_channels, self.kernel_size,
            dilation=self.dilation
        )

    def forward(self, x):
        # Pad left only
        return self.conv(self.padding_layer(x))


class WavenetBlock(nn.Module):
    def __init__(self, channels_gate, channels_residual, channels_skip, dilation, kernel_size: int = 2):
        super().__init__()
        self.dilated_conv = CausalConv1d(channels_residual, 2 * channels_gate, dilation, kernel_size)

        # The number of channels in the gated activation
        self.gate_channels = channels_gate

        # 1x1 convolutions for residual and skip connections
        self.residual_conv = nn.Conv1d(self.gate_channels, channels_residual, 1)
        self.skip_conv = nn.Conv1d(self.gate_channels, channels_skip, 1)

        # LayerNorm normalizes over the feature dimension (channels_residual)
        # Pre-normalization: normalize input (before dilated conv)
        # self.layer_norm_input = nn.LayerNorm(channels_residual, elementwise_affine=False)
        # Post-normalization: normalize output
        # self.layer_norm_output = nn.LayerNorm(channels_residual, elementwise_affine=False) 

    def forward(self, x):
        # Pre-normalization
        # x_norm_in = self.layer_norm_input(x.transpose(1, 2)).transpose(1, 2)

        # Pad left only
        x_conv = self.dilated_conv(x)
        
        # Gated activation
        x_gated = x_conv[:, :self.gate_channels, :] * torch.sigmoid(
            x_conv[:, self.gate_channels:, :]
        )
        
        residuals = self.residual_conv(x_gated) + x  # Shape (B, C_res, T)

        # Post-normalization (assuming self. is initialized in __init__)
        # Transpose to (B, T, C_res), apply norm, transpose back to (B, C_res, T)
        # residual_normalized = self.layer_norm_output(residuals.transpose(1, 2)).transpose(1, 2) 

        # return skip, residual_normalized # Return the normalized version
        return self.skip_conv(x_gated), residuals


class WaveNet(nn.Module):
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        n_layers: int,
        vocabulary_size: int,
        channels_embedding: int = 256,
        channels_gate: int = 256,
        channels_residual: int = 128,
        channels_skip: int = 256,
        kernel_size: int = 2,
        learning_rate: float = 1e-3
    ):
        """
        nn.Embedding is a thin wrapper of torch.tensor that represents a table of embeddings.
        """
        super().__init__()

        # NOTE: Neural networks should not receive raw integers as input,
        # because they are not normalized.
        # 
        # We *could* use one-hot encoding to represent the input (later converting it to a float tensor):
        # tensor_x_onehot = F.one_hot(tensor_x, num_classes=vocabulary_size).float()
        # but this is not efficient for large vocabularies.
        # 
        # Instead, we can map the input to a lower-dimensional space using an embedding layer.
        # This means that one-hot encoding is essentially an embedding layer
        # where an n-dimensional vector is mapped to another n-dimensional vector.
        self.vocabulary_size = vocabulary_size
        self.channels_embedding = channels_embedding
        self.layer_embedding = nn.Embedding(vocabulary_size, channels_embedding)

        self.batch_size = batch_size
        self.block_size = block_size
        self.n_layers = n_layers

        # Initial causal convolution
        self.layer_initial_conv = CausalConv1d(channels_embedding, channels_residual)

        self.kernel_size = kernel_size
        self.dilations = [2**i for i in range(self.n_layers)]  # e.g., [1, 2, 4, 8, 16, 32, ...]
        self.blocks = nn.ModuleList([
            WavenetBlock(
                channels_gate=channels_gate,      
                channels_residual=channels_residual, 
                channels_skip=channels_skip,     
                kernel_size=self.kernel_size,
                dilation=d
            )
            for d in self.dilations
        ])

        # Output layers
        # self.layer_output_1 = nn.Conv1d(channels_skip, channels_skip, 1)
        self.layer_output_2 = nn.Conv1d(channels_skip, vocabulary_size, 1)

        self.layers_and_learning_rates = {
            'embedding': (self.layer_embedding, learning_rate * 10),
            'initial_conv': (self.layer_initial_conv, learning_rate * 10),
            **{
                f'block_{i}_dilated': (block.dilated_conv, learning_rate * 10)
                for i, block in enumerate(self.blocks)
            },
            **{
                f'block_{i}_residual': (block.residual_conv, learning_rate)
                for i, block in enumerate(self.blocks)
            },
            **{
                f'block_{i}_skip': (block.skip_conv, learning_rate)
                for i, block in enumerate(self.blocks)
            },
            # 'output_1': (self.layer_output_1, 0.07),
            'output_2': (self.layer_output_2, learning_rate),
        }

        self.init_weights()


    def init_weights(self):
        # Initialize weights using Xavier uniform initialization
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.kaiming_normal_(p, mode='fan_in', nonlinearity='relu') 
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1: # Typically identifies bias vectors (and potentially LayerNorm params)
                # Initialize 1D parameters (likely biases) to zero
                nn.init.zeros_(p) 
            p.requires_grad = True

    
    def forward(self, x):
        x = self.layer_embedding(x)  # (batch_size, T=block_size, C=embedding_dims)
        # Initial causal convolution
        x = self.layer_initial_conv(x.transpose(1, 2))  # (batch_size, C=channels, T=block_size)

        # WaveNet blocks
        skip_connections = []
        for block in self.blocks:
            skip, x = block(x)
            skip_connections.append(skip)
        
        # Sum skip connections
        x = torch.stack(skip_connections, dim=0).sum(dim=0) / len(skip_connections)
        # x = self.layer_output_1(nn.functional.leaky_relu(x))
        x = self.layer_output_2(nn.functional.leaky_relu(x))

        # Is (B, C, T), permute to (B, T, C)
        return x.permute(0, 2, 1)

    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)
