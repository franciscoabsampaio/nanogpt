from nanogpt.models.bigram import BigramLanguageModel
from nanogpt.models.wavenet import WaveNetModel
from nanogpt.input import get_input_data
from nanogpt.token import get_encoder
from nanogpt.train import split_train_test, get_batch
from nanogpt.visualization import save_plot_histogram_of_tensors, save_plot_loss, save_plot_update_to_data_ratios
import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available
    
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
    print(f"Initial loss shouldn't be greater than: {(
        max_initial_loss := torch.tensor(vocabulary_size).log().item()
    )}")

    torch.manual_seed(42)

    model = WaveNetModel(
        vocabulary_size=vocabulary_size,
        embedding_dims=200,
        batch_size=100,  # The number of sequences to process in parallel
        block_size=10,  # The number of tokens used to predict the next token
        n_neurons=200,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(([
        {'params': model.embedding.parameters(), 'lr': 12},  # faster for embeddings
        {'params': model.layer_hidden.parameters(), 'lr': 0.015},
        {'params': model.layer_output.parameters(), 'lr': 0.07},
        # {'params': model.layer_batchnorm.parameters(), 'lr': 1e-3},  # optional
    ]))

    list_of_losses = [max_initial_loss]
    iterations, max_iterations = 0, 2000
    list_of_update_to_data_ratios = []

    while iterations < max_iterations and list_of_losses[-1] > 0.1:
        iterations += 1
        # Forward
        tensor_x, tensor_y = get_batch(tensor_train, batch_size=model.batch_size, block_size=model.block_size)

        logits = model(tensor_x.to(device))
        loss = criterion(logits, tensor_y[:, -1].to(device))
        del tensor_x, tensor_y
        list_of_losses.append(loss.item())

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if iterations > max_iterations / 2:
            print(1)
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        optimizer.step()

        # Update
        list_of_update_to_data_ratios.append([
            # Iterate over parameter groups
            (param_group['lr'] * p.grad.std() / (p.data.std() + 1e-5)).log10().item()
            for param_group in optimizer.param_groups
            for p in param_group['params']
        ])
        print(f"batch loss at iteration {iterations}: {loss.item()}")

        save_plot_histogram_of_tensors(
            model.parameters(),
        )
        save_plot_loss(list_of_losses)
        save_plot_update_to_data_ratios(
            model.parameters(),
            list_of_update_to_data_ratios
        )
    
    with torch.no_grad():
        # tensor_x, tensor_y = get_batch(tensor_train, batch_size=len(tensor_train) // 5, block_size=model.block_size)
        # logits = model(tensor_x.to(device))
        # loss = criterion(logits, tensor_y[:, -1].to(device))
        # print(f"training loss: {loss.item()}")

        tensor_x, tensor_y = get_batch(tensor_validation, batch_size=len(tensor_validation), block_size=model.block_size)
        logits = model(tensor_x.to(device))
        loss = criterion(logits, tensor_y[:, -1].to(device))
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
