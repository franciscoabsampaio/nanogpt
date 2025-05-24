from nanogpt import input
import os
import tiktoken
from transformers import AutoTokenizer
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
        sp.Load("src/nanogpt/tokenizer/shakespeare35k.model")
        return sp, sp.vocab_size()
    elif tokenizer == 'autotokenizer':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tokenizer, tokenizer.vocab_size
    elif tokenizer == 'char':
        return CharEncoder(vocabulary), len(vocabulary)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")


def tokenize():
    ord()  # gets the unicode code point of a character
    "string".encode('utf-8')  # encodes a string to bytes


def train_tokenizer():
    options = dict(
        # input
        input=input.TINYSHAKESPEARE,
        input_format="text",
        # output
        model_prefix="shakespeare35k",  # output filename prefix
        # algorithm
        model_type="bpe",  # Byte-Pair Encoding
        vocab_size=32000,
        # normalization
        normalization_rule_name='identity',
        remove_extra_whitespaces=False,
        input_sentence_size=2_000_000,  # max number of training sentences
        max_sentence_length=4192,  # max number of bytes per sequence
        seed_sentencepiece_size=1_000_000,
        shuffle_input_sentence=True,
        # rare word treatment
        character_coverage=0.999995,
        byte_fallback=True,
        # merge_rules
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        max_sentencepiece_length=27,  # Shakespeare's longest word: Honorificabilitudinitatibus
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        # special tokens
        unk_id=0,  # Unknown tokens
        bos_id=1,  # Beginning of sentence
        eos_id=2,  # End of sentence
        pad_id=-1,
        # systems
        num_threads=os.cpu_count(),  # use all system resources
    )
    sentencepiece.SentencePieceTrainer.train(**options)
