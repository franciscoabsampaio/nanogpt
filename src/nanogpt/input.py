def get_input_data(filename: str = "/mnt/ai/tinyshakespeare/input.txt") -> dict:
    with open(filename) as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocabulary_size = len(chars)

    return {
        "text": text,
        "vocabulary": chars,
        "vocabulary_size": vocabulary_size
    }
