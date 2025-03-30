from nanogpt.bigram import BigramLanguageModel
from nanogpt.input import get_input_data
from nanogpt.token import get_encoder
from nanogpt.train import split_train_test, get_batch
import torch
import sys


def main():
    data = get_input_data()
    
    enc, vocabulary_size = get_encoder('tiktoken', vocabulary=data['vocabulary'])
    tensor = torch.tensor(
        enc.encode(data['text']),
        dtype=torch.int64  # We're using integers because each char will be mapped to an int
    )

    tensor_train, tensor_test = split_train_test(tensor)

    # torch.manual_seed(42)

    block_size = len(tensor_train) - 1
    tensor_x, tensor_y = get_batch(tensor_train, batch_size=1, block_size=block_size)
    
    bigrams = torch.stack((tensor_x[:, :-1], tensor_y[:, :-1]), dim=-1)
    # Reshape to [num_bigrams, 2] for counting
    bigrams = bigrams.view(-1, 2)

    # Count unique bigrams
    tensor_unique_bigrams, tensor_counts = torch.unique(bigrams, return_counts=True, dim=0)
    tensor_probabilities = torch.nn.functional.softmax(tensor_counts.float(), dim=0)
    
    next_token = tensor_x[0, torch.randint(1, tensor_x.shape[1], (1,))].item()
    
    import time
    while True:
        print(enc.decode_single_token_bytes(next_token))
        mask = tensor_unique_bigrams[:, 0] == next_token
        matching_bigrams = tensor_unique_bigrams[mask]
        matching_probs = tensor_probabilities[mask] + 1e-8  # Apply Laplace smoothing to void zero probabilities
        sample_idx = torch.multinomial(matching_probs / matching_probs.sum(), num_samples=1).item()
        next_token = matching_bigrams[sample_idx, 1]
        time.sleep(0.5)

    sys.exit()
    batch_size, block_size = 4, 8
    tensor_x_batch, tensor_y_batch = get_batch(tensor_train, batch_size=batch_size, block_size=block_size)
    
    print([(tensor_x_batch[i, j], tensor_y_batch[i, j]) for i in range(batch_size) for j in range(block_size)])

    m = BigramLanguageModel(vocabulary_size)
    logits, loss = m(tensor_x_batch, tensor_y_batch)
    print(loss)


if __name__ == "__main__":
    main()
