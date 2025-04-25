from nanogpt.models import *
from nanogpt.input import get_input_data
from nanogpt.token import get_encoder
from nanogpt.train import split_train_test, get_batch
from nanogpt.visualization import save_plot_histogram_of_tensors, save_plot_loss, save_plot_update_to_data_ratios
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    model = MLP(
        vocabulary_size=vocabulary_size,
        embedding_dims=200,
        batch_size=100,  # The number of sequences to process in parallel
        block_size=10,  # The number of tokens used to predict the next token
        n_neurons=200,
    )
    model = WaveNet(
        batch_size=24,
        block_size=512,
        n_layers=12,
        vocabulary_size=vocabulary_size,
        channels_embedding=512,
        channels_residual=256,
        channels_skip=512,
        kernel_size=3,
    )
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [{
            'params': layer[0].parameters(),
            'lr': 2e-3, #layer[1] / 50,
            'name': key,
            'named_params': dict(layer[0].named_parameters()),
        } for key, layer in model.layers_and_learning_rates.items() ],
        weight_decay=0.01,
    )

    list_of_losses = [max_initial_loss]
    iterations, max_iterations = 0, 1000
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-5)
    dict_of_update_to_data_ratios = {
        k: { m: [] for m, _ in v[0].named_parameters() }
        for k, v in model.layers_and_learning_rates.items()
    }

    while iterations < max_iterations and list_of_losses[-1] > 0.1:
        iterations += 1
        # Forward
        tensor_x, tensor_y = get_batch(tensor_train, batch_size=model.batch_size, block_size=model.block_size)

        # logits: (batch, vocab_size, block_size)
        logits = model(tensor_x.to(device))
        # Reshape for CrossEntropyLoss:
        # Need (Batch * BlockSize, VocabSize) for logits
        # Need (Batch * BlockSize) for targets
        B, C, T = logits.shape
        logits_reshaped = logits.view(B * T, C) 
        targets_reshaped = tensor_y.reshape(B * T).to(device) # Use reshape or view

        # Calculate loss over all positions
        loss = criterion(logits_reshaped, targets_reshaped)
        list_of_losses.append(loss.item())

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # for m, p in model.named_parameters():
        #     if p.grad is None:
        #         print(f"Parameter {m} has no gradient")

        # Update
        for param_group in optimizer.param_groups:
            for m, p in param_group['named_params'].items():
                if p.grad is not None:
                    # Append the update-to-data ratio for the parameter
                    dict_of_update_to_data_ratios[param_group['name']][m].append(
                        (param_group['lr'] * p.grad.std() / (p.data.std() + 1e-5)).log10().item()
                    )
                else:
                    print(f"Parameter {m} has no gradient")
        print(f"batch loss at iteration {iterations}: {loss.item()}")

        save_plot_histogram_of_tensors({k: v for k, v in model.named_parameters()})
        save_plot_loss(list_of_losses)
        save_plot_update_to_data_ratios(dict_of_update_to_data_ratios)
        
        del tensor_x, tensor_y, logits, logits_reshaped, targets_reshaped 
    
    # with torch.no_grad():
        # tensor_x, tensor_y = get_batch(tensor_train, batch_size=len(tensor_train) // 5, block_size=model.block_size)
        # logits = model(tensor_x.to(device))
        # loss = criterion(logits, tensor_y[:, -1].to(device))
        # print(f"training loss: {loss.item()}")

        # tensor_x, tensor_y = get_batch(tensor_validation, batch_size=len(tensor_validation), block_size=model.block_size)
        # logits = model(tensor_x.to(device))
        # loss = criterion(logits, tensor_y[:, -1].to(device))
        # print(f"validation loss: {loss.item()}")

    import time

    tensor_x, tensor_y = get_batch(tensor_validation, batch_size=model.batch_size, block_size=model.block_size)
    current_token = tensor_x[0, :]

    model.to('cpu')
    model.eval()

    for _ in range(100):
        with torch.no_grad():
            logits = model(current_token.unsqueeze(0))  # shape: (1, vocab_size, block_size)
            
            # Get logits from the last position in the sequence
            logits_last = logits[:, :, -1]  # shape: (1, vocab_size)
            probs = F.softmax(logits_last, dim=-1)  # shape: (1, vocab_size)

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probs[0], num_samples=1)[0]

            print(enc.decode([next_token.item()]), end='', flush=True)

            # Append the token and keep only the last `block_size` tokens
            current_token = torch.cat([current_token, next_token.unsqueeze(0)])[-model.block_size:]

            time.sleep(0.5)
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

    tensor_x_batch, tensor_y_batch = get_batch(tensor_train, batch_size=batch_size, block_size=block_size)
    
    print([(tensor_x_batch[i, j], tensor_y_batch[i, j]) for i in range(batch_size) for j in range(block_size)])

    m = BigramLanguageModel(vocabulary_size)
    logits, loss = m(tensor_x_batch, tensor_y_batch)
    print(loss)


if __name__ == "__main__":
    main()
