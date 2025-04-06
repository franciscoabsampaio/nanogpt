import tiktoken
import sentencepiece


class CharEncoder:
    def __init__(self, vocabulary: str):
        self.str_to_int = {c: i for i, c in enumerate(vocabulary)}
        self.int_to_str = {i: c for i, c in enumerate(vocabulary)}

    def encode(self, string: str) -> list[int]:
        return [self.str_to_int[c] for c in string]
    
    def decode(self, integers: list[int]) -> str:
        if isinstance(integers, int):
            integers = [integers]
        return ''.join([self.int_to_str[i] for i in integers])


def get_encoder(tokenizer: str = 'tiktoken', vocabulary: str = None) -> list[int]:
    if tokenizer == 'tiktoken':
        enc = tiktoken.get_encoding("p50k_base")
        return enc, enc.n_vocab
    elif tokenizer == 'sentencepiece':
        sp = sentencepiece.SentencePieceProcessor()
        sp.Load("sentencepiece.model")
        return sp, sp.get_vocab_size()
    elif tokenizer == 'char':
        return CharEncoder(vocabulary), len(vocabulary)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")


def tokenize():
    ord()  # gets the unicode code point of a character
    "string".encode('utf-8')  # encodes a string to bytes