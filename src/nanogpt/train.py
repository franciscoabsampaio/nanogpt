import torch


def split_train_test(data: torch.tensor, train_size=0.8) -> tuple[torch.tensor, torch.tensor]:
    train_size = int(len(data) * train_size)
    return data[:train_size], data[train_size:]


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
    max_size = len(data) - block_size
    ix = torch.randint(0, max_size, (batch_size,))
    
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y
