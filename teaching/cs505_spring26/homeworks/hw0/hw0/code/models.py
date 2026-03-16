# CS505: NLP - Spring 2026
import torch
import numpy as np
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
import random

class BoWFeaturizer:
    """
    This is a bag-of-words featurizer. It uses `build_vocab` to load a list of Examples
    and uses the top `max_vocab_size` words by frequency as its vocabulary.
    For a given Example, it counts the number of instances of each word in
    `self.vocab` and returns this vector of counts.
    """
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.vocab = {} # mapping word -> index
        self.inverse_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, data):
        counts = Counter()
        # TODO: count the number instances of each token (here, just words and
        # punctuation) in `data`. Filter the vocab down to the `self.max_vocab_size` most
        # frequent tokens, and put these in a variable called `most_common`. HINT:
        # you can use the `word_tokenize` function that's been
        # imported above to tokenize the string.
        # STUDENT START ---------------------------------
        # STUDENT END ------------------------------------
        
        # you might need to remove the `count` variable here, depending on how you
        # implemented the above.
        self.vocab = {word: idx for idx, (word, count) in enumerate(most_common)}
        self.inverse_vocab = {idx: word for idx, (word, count) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)

        print(f"Vocabulary built with {self.vocab_size} words.")

    def get_feature_vector(self, text):
        # TODO: Return a bag-of-words feature vector. Each index in
        # the vocabulary should have a corresponding index in this vector.
        # A token's vector index should contain the frequency of that token
        # in `text`.
        # This shold return a torch tensor of size (vocab_size,).
        # STUDENT START -------------------------
        pass
        # STUDENT END ---------------------------


class BigramFeaturizer(BoWFeaturizer):
    def build_vocab(self, data):
        counts = Counter()
        for ex in data:
            tokens = word_tokenize(ex.text.lower())
            # TODO: generate bigrams
            # STUDENT START ----------------------------
            # STUDENT END ------------------------------
        
        # TODO: build your vocabulary of the `self.max_vocab_size` most frequent bigrams.
        # STUDENT START -------------------------------------------
        # STUDENT END ---------------------------------------------

    def get_feature_vector(self, text):
        tokens = word_tokenize(text.lower())
        vec = torch.zeros(self.vocab_size)
        
        # TODO: use the list of tokens to generate bigram features.
        # Return the bigram feature vector.
        # STUDENT START --------------------------------------
        # STUDENT END -----------------------------------------


class BlackBoxClassifier(torch.nn.Module):
    """
    This is a logistic regression classifier using PyTorch's built-in modules.
    Only used in Task 1. You will implement something like this from scratch
    in the LogisticRegressionClassifier class.
    """
    def __init__(self, input_dim, num_classes):
        super(BlackBoxClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # Returns logits (unnormalized scores)
        return self.linear(x)


class LogisticRegressionClassifier:
    def __init__(self, input_dim, num_classes):
        # Initialize weights and bias
        # Weights: (input_dim, num_classes), Bias: (num_classes)
        self.weights = torch.randn(input_dim, num_classes, requires_grad=False) * 0.01
        self.bias = torch.zeros(num_classes, requires_grad=False)

    def forward(self, x):
        # TODO: implement the logistic regression as z = W^T * x + b. Return z.
        # Hint: this should only require one line of code!
        # STUDENT START ---------------------------------
        pass
        # STUDENT END -----------------------------------

    def softmax(self, logits):
        # TODO: implement softmax. You may *not* use torch.nn.softmax or any
        # similar function. You may use torch.exp if you wish.
        # STUDENT START --------------------------------
        pass
        # STUDENT END ----------------------------------

    def predict(self, x):
        logits = self.forward(x)
        probs = self.softmax(logits)
        return torch.argmax(probs).item()


def train_logistic_regression(train_data, dev_data, featurizer, num_classes=4, lr=0.01, epochs=5,
                              method="bow"):
    input_dim = featurizer.vocab_size
    if method == "lr":
        model = LogisticRegressionClassifier(input_dim, num_classes)
    elif method == "bow":
        model = BlackBoxClassifier(input_dim, num_classes)
    
    print("Training logistic regression...")
    
    for epoch in range(epochs):
        shuffled_train = train_data.copy()
        random.shuffle(shuffled_train)
        total_loss = 0
        
        for ex in shuffled_train:
            x = featurizer.get_feature_vector(ex.text) # (vocab_size,)
            y_true = ex.label
            
            # 1. Call the forward function and compute the probability
            # of each class according to the model.
            logits = model.forward(x)
            probs = model.softmax(logits)
            
            # TODO: 2. Compute the negative log likelihood loss.
            # STUDENT START ----------------------------
            # STUDENT END ------------------------------
            
            # TODO: 3. Compute the gradient for the weights, and the gradient for
            # for the bias. You may not use .backward().
            # STUDENT START ----------------------------
            # STUDENT END ------------------------------
            
            # TODO: 4. Update the parameters by multiplying the gradients you
            # derived in the previous step by the learning rate, and then subtracting
            # them from the weights and biases. You will need at least 1 line to update the
            # weight matrix, and at least 1 line to update the bias.
            # STUDENT START ----------------------------
            # STUDENT END ------------------------------
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")
        
    return model


def train_torch_model(train_data, dev_data, featurizer, num_classes=4, lr=0.01, epochs=5):
    """
    Pre-provided gradient descent function using PyTorch's optimizer and loss.
    """
    input_dim = featurizer.vocab_size
    model = BlackBoxClassifier(input_dim, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print("Training Built-in PyTorch Model...")
    
    # This is an example of a training loop. Here, we're using only black-box
    # built-in PyTorch functions. You will implement the underlying functionality
    # of these functions as part of Task 2.
    for epoch in range(epochs):
        shuffled_train = train_data.copy()
        random.shuffle(shuffled_train)
        model.train() # Set model to training mode
        total_loss = 0
        
        for ex in shuffled_train:
            x = featurizer.get_feature_vector(ex.text)
            x_tensor = x.unsqueeze(0)
            y_tensor = torch.tensor([ex.label], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")
    
    # TODO: You're given the weight matrix of your trained model, which is of
    # shape (C, V), where C is the number of classes (here, 4) and V is
    # the vocabulary size. For each class, you will get the top-5 weight indices,
    # and print out the tokens they correspond to. No need to return anything here;
    # just print out the top weights/tokens and put them in your written report.
    # STUDENT START ----------------------------------
    weights = model.linear.weight
    # STUDENT END ------------------------------------
        
    return model