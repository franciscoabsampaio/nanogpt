import torch
import torch.nn.functional as F


def split_train_test(data: torch.tensor, train_size=0.8) -> tuple[torch.tensor, torch.tensor]:
    train_size = int(len(data) * train_size)
    return data[:train_size], data[train_size:]


def pad(data: torch.tensor, block_size: int):
    if len(data) <= block_size:
        return F.pad(data, (block_size - len(data), 0))
    else:
        return data


def get_batch(
    data: torch.tensor,
    batch_size: int  = 4,
    block_size: int = 8
) -> tuple[torch.tensor, torch.tensor]:
    """
    Batch_size is the number of independent sequences that will be processed in parallel.
    Block_size is the maximum context size for predictions.

    This function generates #batch_size number of IDs,
    and creates the corresponding number of input and target sequences,
    starting from each of these randomly selected IDs.

    x is a tensor of shape (batch_size, block_size) containing the input sequences.
    y is a tensor of shape (batch_size, block_size) containing the target sequences.
    y is the same as x, but shifted one position to the right (predicting the next token).
    """
    # If data is too short, pad it up to block_size+1
    data = pad(data, block_size + 1)
    max_starting_id = len(data) - block_size    

    # Sample random starting IDs for the sequence
    ix = torch.randint(0, max_starting_id, (batch_size,))
    
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def loss(tensor_logits: torch.tensor, tensor_y: torch.tensor):
    """
    NOTE: GOAL: Maximize likelihood of the data wrt model parameters
    equivalent to mazimizing the log-likelihood of the data (because log is monotonic)
    equivalent to minimizing the negative log-likelihood of the data
    equivalent to minimizing the average negative log-likelihood of the data
    
    NOTE: The logarithm of 0 is infinity and thus undefined,
    so we need to add a small value (epsilon) to avoid division by zero
    This is called model smoothing or Laplace smoothing
     
    NOTE: We can then compute the negative log loss,
    which is what we want to minimize.
    A measure of the variance of the weights can be added to the loss function,
    effectively minimizing the counts of the most frequent tokens,
    forcing the model to learn probabilities that are more uniform,
    snoothing the model and preventing overfitting.
    This is called regularization.
    
    Reshape for CrossEntropyLoss:
    Need (Batch * BlockSize, Channels) for logits
    Need (Batch * BlockSize) for targets
    """
    B, T, C = tensor_logits.shape
    return torch.nn.CrossEntropyLoss(label_smoothing=0.05)(
        tensor_logits.reshape(B * T, C),
        tensor_y.reshape(B * T)
    )


@torch.no_grad()
def estimate_val_loss(
    model,
    tensor_validation: torch.tensor,
    iterations_for_computing_loss: int,
    device: str = 'cpu'
):
    model.eval()
    tensor_losses = torch.zeros(iterations_for_computing_loss).to(device)
    for k in range(iterations_for_computing_loss):
        tensor_x, tensor_y = get_batch(tensor_validation, model.batch_size, model.block_size)
        tensor_logits = model(tensor_x.to(device))
        tensor_losses[k] = model.loss(tensor_logits, tensor_y.to(device))
    model.train()
    # Normalize over number of iterations
    return tensor_losses.mean().item()
