from nanogpt.bigram import BigramLanguageModel
from nanogpt.input import get_input_data
from nanogpt.token import get_encoder
from nanogpt.train import split_train_test, get_batch
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available
    
    data = get_input_data()
    
    enc, vocabulary_size = get_encoder('tiktoken', vocabulary=data['vocabulary'])
    tensor = torch.tensor(
        enc.encode(data['text']),
        dtype=torch.int64  # We're using integers because each char will be mapped to an int
    )

    tensor_train, tensor_test = split_train_test(tensor, train_size=0.8)
    tensor_validation, tensor_test = split_train_test(tensor_test, train_size=0.5)

    # Initial loss should be, at worst, equivalent to a random guess.
    # -log(p(x)) = -log(1 / vocabulary_size) = log(vocabulary_size)
    print(f"Initial loss shouldn't be greater than: {torch.tensor(vocabulary_size).log().item()}")

    torch.manual_seed(42)

    batch_size = 100  # The number of sequences to process in parallel
    block_size = 5  # The number of tokens used to predict the next token
    # tensor_x, tensor_y = get_batch(tensor_train, batch_size=batch_size, block_size=block_size)

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
    embedding_dims = 1000
    n_hidden_layer = 1000

    tensor_embedding_table = torch.randn((vocabulary_size, embedding_dims))
    tensor_weights_1 = torch.randn((embedding_dims * block_size, n_hidden_layer))
    # tensor_biases_1 = torch.randn(n_hidden_layer)
    tensor_weights_2 = torch.randn((n_hidden_layer, vocabulary_size))
    tensor_biases_2 = torch.randn(vocabulary_size)
    tensor_batch_normalization_gain = torch.ones((1, n_hidden_layer))
    tensor_batch_normalization_bias = torch.zeros((1, n_hidden_layer))
    tensor_batch_normalization_running_mean = torch.zeros((1, n_hidden_layer))
    tensor_batch_normalization_running_std = torch.ones((1, n_hidden_layer))
    
    list_of_parameters = [
        tensor_embedding_table,
        tensor_weights_1,
        # tensor_biases_1,
        tensor_weights_2,
        tensor_biases_2,
        tensor_batch_normalization_gain,
        tensor_batch_normalization_bias
    ]
    for p in list_of_parameters:
        p.requires_grad = True

    def forward_mlp(
        tensor_x,
        tensor_y,
        tensor_embedding_table,
        tensor_weights_1,
        # tensor_biases_1,
        tensor_weights_2,
        tensor_biases_2,
        tensor_batch_normalization_gain,
        tensor_batch_normalization_bias,
        tensor_batch_normalization_running_mean,
        tensor_batch_normalization_running_std,
    ):
        tensor_embeddings = tensor_embedding_table[tensor_x]  # (batch_size, block_size, embedding_dims)
        
        # Hidden layer
        tensor_layer_hidden = f.silu(
            # Pre-activations (Linear layer)
            tensor_embeddings.view(
                -1,
                tensor_embeddings.shape[1] * tensor_embeddings.shape[2]
            ) @ tensor_weights_1
        )

        # BatchNorm layer
        with torch.no_grad():
            tensor_batch_normalization_running_mean, tensor_batch_normalization_running_std = (
                # 0.1 is the momentum
                0.9 * tensor_batch_normalization_running_mean
                + 0.1 * tensor_layer_hidden.mean(
                    dim=0, keepdim=True
                ),
                0.9 * tensor_batch_normalization_running_std
                + 0.1 * tensor_layer_hidden.std(
                    dim=0, keepdim=True
                )
            )

        tensor_logits = (
            # Batch normalization
            tensor_batch_normalization_gain
            * (tensor_layer_hidden - tensor_batch_normalization_running_mean)
            / (tensor_batch_normalization_running_std + 1e-5)
            + tensor_batch_normalization_bias
        ) @ tensor_weights_2 + tensor_biases_2

        plt.hist(tensor_layer_hidden.view(-1).tolist(), 100)
        plt.savefig("output/histogram_hidden_layer.png")

        return f.cross_entropy(tensor_logits, tensor_y).log().mean()

    learning_rate = 0.0001
    loss = 1
    iterations = 0
    while iterations < 20 and loss > 0.1:
        iterations += 1
        # Forward
        tensor_x, tensor_y = get_batch(tensor_train, batch_size=batch_size, block_size=block_size)
        loss = forward_mlp(
            tensor_x,
            tensor_y[:, -1],
            tensor_embedding_table,
            tensor_weights_1,
            tensor_weights_2,
            tensor_biases_2,
            tensor_batch_normalization_gain,
            tensor_batch_normalization_bias,
            tensor_batch_normalization_running_mean,
            tensor_batch_normalization_running_std,
        )

        # Backward
        for p in list_of_parameters:
            p.grad = None
        loss.backward()
        # Update
        for p in list_of_parameters:
            p.data += -learning_rate * p.grad
        print(f"batch loss at iteration {iterations}: {loss.item()}")
        # break
    
    with torch.no_grad():
        tensor_x, tensor_y = get_batch(tensor_train, batch_size=len(tensor_train) // 5, block_size=block_size)
        loss = forward_mlp(
            tensor_x,
            tensor_y[:, -1],
            tensor_embedding_table,
            tensor_weights_1,
            tensor_weights_2,
            tensor_biases_2,
            tensor_batch_normalization_gain,
            tensor_batch_normalization_bias,
            tensor_batch_normalization_running_mean,
            tensor_batch_normalization_running_std,
        )
        print(f"training loss: {loss.item()}")

        tensor_x, tensor_y = get_batch(tensor_validation, batch_size=len(tensor_validation), block_size=block_size)
        loss = forward_mlp(
            tensor_x,
            tensor_y[:, -1],
            tensor_embedding_table,
            tensor_weights_1,
            tensor_weights_2,
            tensor_biases_2,
            tensor_batch_normalization_gain,
            tensor_batch_normalization_bias,
            tensor_batch_normalization_running_mean,
            tensor_batch_normalization_running_std,
        )
        print(f"validation loss: {loss.item()}")
    return

    # NOTE: Because we're assuming that the output of each neuron is the logarithm of the count (log-counts),
    # we need to exponentiate the logits to get the counts.
    # Since x is a one-hot vector (ones and zeros),
    # the neuron weights are, too, log-counts.
    # This is equivalent to applying the softmax function to the logits,
    # essentially converting them to probabilities.
    # 
    # tensor_counts = tensor_logits.exp()
    # tensor_probabilities = tensor_counts / tensor_counts.sum(dim=-1, keepdim=True)
    tensor_probabilities = f.softmax(tensor_logits.float(), dim=1)

    # NOTE: GOAL: Maximize likelihood of the data wrt model parameters
    # equivalent to mazimizing the log-likelihood of the data (because log is monotonic)
    # equivalent to minimizing the negative log-likelihood of the data
    # equivalent to minimizing the average negative log-likelihood of the data
    # 
    # NOTE: The logarithm of 0 is infinity and thus undefined,
    # so we need to add a small value (epsilon) to avoid division by zero
    # This is called model smoothing or Laplace smoothing
    # 
    # NOTE: We can then compute the negative log loss,
    # which is what we want to minimize.
    # A measure of the variance of the weights can be added to the loss function,
    # effectively minimizing the counts of the most frequent tokens,
    # forcing the model to learn probabilities that are more uniform,
    # snoothing the model and preventing overfitting.
    # This is called regularization.
    tensor_negative_log_loss = (
        -(tensor_probabilities[0, torch.arange(block_size), tensor_y] + 1e-8).log().mean()
        + 0.01 * (tensor_neuron_weights ** 2).mean()  # 0.01 is the regularization strength, a hyperparameter
    )
    print(tensor_negative_log_loss.item())

    tensor_neuron_weights.grad = None
    tensor_negative_log_loss.backward()

    tensor_neuron_weights.data += -0.5 * tensor_neuron_weights.grad

    # forward
    tensor_logits = tensor_x_onehot @ tensor_neuron_weights  # log-counts
    tensor_probabilities = f.softmax(tensor_logits.float(), dim=1)
    tensor_negative_log_loss = -tensor_probabilities[0, torch.arange(block_size), tensor_y].log().mean()
    
    print(tensor_negative_log_loss.item())

    import time

    current_token = tensor_x[0, 0]  # start token

    for _ in range(100):  # generate 100 tokens
        tensor_x_onehot = f.one_hot(current_token, num_classes=vocabulary_size).float()
        tensor_logits = tensor_x_onehot @ tensor_neuron_weights
        tensor_probabilities = f.softmax(tensor_logits.float(), dim=0)

        current_token = torch.multinomial(tensor_probabilities, num_samples=1)[0]
        print(enc.decode([current_token.item()]), end='', flush=True)
        time.sleep(0.5)
    
    return
    batch_size, block_size = 4, 8
    tensor_x_batch, tensor_y_batch = get_batch(tensor_train, batch_size=batch_size, block_size=block_size)
    
    print([(tensor_x_batch[i, j], tensor_y_batch[i, j]) for i in range(batch_size) for j in range(block_size)])

    m = BigramLanguageModel(vocabulary_size)
    logits, loss = m(tensor_x_batch, tensor_y_batch)
    print(loss)


if __name__ == "__main__":
    main()
