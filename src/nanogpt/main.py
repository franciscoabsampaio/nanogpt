from nanogpt.bigram import BigramLanguageModel
from nanogpt.input import get_input_data
from nanogpt.token import get_encoder
from nanogpt.train import split_train_test, get_batch
import torch


def main():
    data = get_input_data()
    
    enc, vocabulary_size = get_encoder('char', vocabulary=data['vocabulary'])
    tensor = torch.tensor(
        enc.encode(data['text']),
        dtype=torch.int64  # We're using integers because each char will be mapped to an int
    )
    print(tensor)

    data_train, data_test = split_train_test(tensor)

    torch.manual_seed(42)
    xb, yb = get_batch(data_train)
    print(xb.shape, yb.shape)

    m = BigramLanguageModel(vocabulary_size)
    logits, loss = m(xb, yb)
    print(loss)

if __name__ == "__main__":
    main()
