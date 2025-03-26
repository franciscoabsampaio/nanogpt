def get_input_data(filename: str = "/mnt/ai/tinyshakespeare/input.txt") -> dict:
    with open(filename) as f:
        text = f.read()
    vocabulary = sorted(list(set(text)))
    vocabulary_size = len(vocabulary)

    return {
        "text": text,
        "vocabulary": vocabulary,
        "vocabulary_size": vocabulary_size
    }
