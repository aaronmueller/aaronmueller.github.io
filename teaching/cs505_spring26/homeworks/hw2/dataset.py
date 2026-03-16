import torch

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 5000
        self.eos_token_id = 1
        self.word2idx = {"<unk>": 0, "</s>": 1}
        self.idx2word = {0: "<unk>", 1: "</s>"}

    def encode(self, text):
        # Dummy encoding
        return [self.word2idx.get(w, 0) for w in text.split()]
    
    def decode(self, ids):
        return " ".join([self.idx2word.get(i, "") for i in ids])

def get_tokenizer():
    return MockTokenizer()

def load_data(split):
    # Returns dummy tensor (Batch Size, Sequence Length)
    print(f"Loading {split} data for LM...")
    return torch.randint(0, 5000, (10, 32))

def get_translation_data(split):
    # Returns dummy tensor formatted as X =|= Y
    print(f"Loading {split} data for Translation...")
    return torch.randint(0, 5000, (10, 32))