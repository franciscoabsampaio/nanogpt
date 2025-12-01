from torch.utils.data import Dataset
import torch
import torch.nn.functional as F


def pad(data: torch.tensor, block_size: int):
    if len(data) <= block_size:
        return F.pad(data, (block_size - len(data), 0))
    else:
        return data


class SequenceDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        # If data is too short, pad it up to block_size+1
        self.data = pad(data_tensor, block_size+1)
        self.block_size = block_size
        self.max_start = len(self.data) - block_size - 1

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx):
        """
        x is a tensor of shape (batch_size, block_size) containing the input sequences.
        y is a tensor of shape (batch_size, block_size) containing the target sequences.
        y is the same as x, but shifted one position to the right (predicting the next token).
        """
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return {
            "input_ids": x,
            "labels": y
        }
