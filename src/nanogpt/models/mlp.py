from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dims: int,
        batch_size: int,
        block_size: int,
        n_neurons: int,
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
        self.embedding_dims = embedding_dims
        self.embedding = nn.Embedding(vocabulary_size, embedding_dims)

        self.batch_size = batch_size
        self.block_size = block_size
        self.n_neurons = n_neurons

        # Hidden layer
        self.layer_hidden = nn.Linear(embedding_dims * block_size, n_neurons, bias=False)
        self.layer_activation = nn.LeakyReLU()
        
        # Batch normalization
        # self.layer_batchnorm = nn.BatchNorm1d(n_neurons)
        self.layer_output = nn.Linear(n_neurons, vocabulary_size)

        self.init_weights()

        # Learning rates
        self.layers_and_learning_rates = {
            'embedding': (self.embedding, 12),
            'hidden': (self.layer_hidden, 0.015),
            'output': (self.layer_output, 0.07),
        }


    def init_weights(self):
        # Initialize weights using Kaiming normal initialization
        nn.init.kaiming_normal_(self.layer_hidden.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.layer_output.weight, mode='fan_in', nonlinearity='linear')
        self.layer_hidden.weight.data *= 2
        self.layer_output.weight.data *= 1

        # Initialize biases
        nn.init.normal_(self.layer_output.bias)
        self.layer_output.bias.data *= (1 / self.vocabulary_size) ** 0.2
        
        # Initialize batch normalization parameters
        # nn.init.ones_(self.layer_batchnorm.weight)
        # nn.init.zeros_(self.layer_batchnorm.bias)

    
    def forward(self, x):
        x = self.embedding(x)  # (batch_size, block_size, embedding_dims)
        
        # Pre-activations (Linear layer)
        x = self.layer_hidden(
            # Flatten
            x.view(x.size(0), -1)
        )
        x = self.layer_activation(x)
        # Optional: normalize only if batch size > 1
        # if x.size(0) > 1:
        #     x = self.layer_batchnorm(x)
        x = self.layer_output(x)
        return x
