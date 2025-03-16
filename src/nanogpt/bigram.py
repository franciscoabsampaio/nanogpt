import torch
import torch.nn as nn
from torch.nn import functional as F


torch.manual_seed(42)


def compute_loss(logits, targets):
    """
    Cross-entropy loss is used to measure the difference between the predicted
    distribution and the actual distribution of the target token.
    The entropy of the correct token should be low,
    because the model should be confident in its prediction.

    Logits are rearranged because the PyTorch implementation of cross-entropy
    expects the logits to be in the form of (Batch, Channel).
    """
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(-1)

    return F.cross_entropy(logits, targets)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        """
        nn.Embedding is a thin wrapper of torch.tensor that represents a table of embeddings.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)
    
    def forward(self, idx: torch.tensor, targets: torch.tensor = None):
        """
        Logits are the scores for the next character in the sequence.
        The correct token should have a high score, while the incorrect tokens should have low scores.
        """
        logits = self.token_embedding_table(idx)  # shape: (Batch=batch_size, Time=block_size, Channel=vocabulary_size)
        loss = compute_loss(logits, targets) if targets else None
        
        return logits, loss


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
