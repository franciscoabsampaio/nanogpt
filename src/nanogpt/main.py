from nanogpt.bigram import BigramLanguageModel
from nanogpt.input import get_input_data
from nanogpt.token import get_encoder
from nanogpt.train import split_train_test, get_batch
import torch
import torch.nn.functional as f
import sys


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available
    
    data = get_input_data()
    
    enc, vocabulary_size = get_encoder('tiktoken', vocabulary=data['vocabulary'])
    tensor = torch.tensor(
        enc.encode(data['text']),
        dtype=torch.int64  # We're using integers because each char will be mapped to an int
    )

    tensor_train, tensor_test = split_train_test(tensor)

    torch.manual_seed(42)

    block_size = 100
    tensor_x, tensor_y = get_batch(tensor_train, batch_size=1, block_size=block_size)
    
    # Neural networks should not receive raw integers as input,
    # because they are not normalized.
    # Instead, we use one-hot encoding to represent the input,
    # but convert it to a float tensor.
    tensor_x_onehot = f.one_hot(tensor_x, num_classes=vocabulary_size).float()

    tensor_neuron_weights = torch.randn((vocabulary_size, vocabulary_size), requires_grad=True)
    tensor_logits = tensor_x_onehot @ tensor_neuron_weights  # log-counts
    # Because we're assuming that the output of each neuron is the logarithm of the count (log-counts),
    # we need to exponentiate the logits to get the counts.
    # Since x is a one-hot vector (ones and zeros),
    # the neuron weights are, too, log-counts.
    # This is equivalent to applying the softmax function to the logits,
    # essentially converting them to probabilities.
    tensor_counts = tensor_logits.exp()
    # tensor_probabilities = tensor_counts / tensor_counts.sum(dim=-1, keepdim=True)
    tensor_probabilities = torch.nn.functional.softmax(tensor_counts.float(), dim=1)
    
    # We can then compute the negative log loss,
    # which is what we want to minimize.
    # A measure of the variance of the weights can be added to the loss function,
    # effectively minimizing the counts of the most frequent tokens,
    # forcing the model to learn probabilities that are more uniform,
    # snoothing the model and preventing overfitting.
    # This is called regularization.
    tensor_negative_log_loss = (
        -tensor_probabilities[0, torch.arange(block_size), tensor_y].log().mean()
        + 0.01 * (tensor_neuron_weights ** 2).mean()  # 0.01 is the regularization strength, a hyperparameter
    )
    print(tensor_negative_log_loss.item())

    tensor_neuron_weights.grad = None
    tensor_negative_log_loss.backward()

    tensor_neuron_weights.data += -0.5 * tensor_neuron_weights.grad

    tensor_logits = tensor_x_onehot @ tensor_neuron_weights  # log-counts
    tensor_counts = tensor_logits.exp()
    tensor_probabilities = torch.nn.functional.softmax(tensor_counts.float(), dim=1)
    tensor_negative_log_loss = -tensor_probabilities[0, torch.arange(block_size), tensor_y].log().mean()
    
    print(tensor_negative_log_loss.item())
    return
    bigrams = torch.stack((tensor_x[:, :-1], tensor_y[:, :-1]), dim=-1)
    # Reshape to [num_bigrams, 2] for counting
    bigrams = bigrams.view(-1, 2)

    # Count unique bigrams
    tensor_unique_bigrams, tensor_counts = torch.unique(bigrams, return_counts=True, dim=0)
    
    
    next_token = tensor_x[0, torch.randint(1, tensor_x.shape[1], (1,))].item()
    
    # GOAL: Maximize likelihood of the data wrt model parameters
    # equivalent to mazimizing the log-likelihood of the data (because log is monotonic)
    # eqhjuivalent to minimizing the negative log-likelihood of the data
    # equivalent to minimizing the average negative log-likelihood of the data
    import time
    while True:
        print(enc.decode_single_token_bytes(next_token))
        mask = tensor_unique_bigrams[:, 0] == next_token
        matching_bigrams = tensor_unique_bigrams[mask]
        # NOTE: The logarithm of 0 is infinity and thus undefined,
        # so we need to add a small value (epsilon) to avoid division by zero
        # This is called model smoothing or Laplace smoothing
        matching_probs = tensor_probabilities[mask] + 1e-8
        sample_idx = torch.multinomial(matching_probs / matching_probs.sum(), num_samples=1).item()
        next_token = matching_bigrams[sample_idx, 1]
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
