# lang_classifier.py
import argparse
import time
from dataset import load_data
import utils
import models
import torch

def main():
    parser = argparse.ArgumentParser(description='CS505 HW0: Text Classification')
    parser.add_argument('--model', type=str, default='TECH', 
                        choices=['TECH', 'BOW', 'LR', 'BIGRAM'],
                        help='Which model to run')
    parser.add_argument('--train_file', type=str, default='../data/train.tsv', help='Path to training data')
    parser.add_argument('--dev_file', type=str, default='../data/dev.tsv', help='Path to dev data')
    
    args = parser.parse_args()
    
    # Load dat
    print("Loading data...")
    # Note: You might need to adjust paths based on where you unzip the data
    train_data = load_data(args.train_file)
    dev_data = load_data(args.dev_file)
    
    # Extract labels for evaluation
    train_labels = [ex.label for ex in train_data]
    dev_labels = [ex.label for ex in dev_data]

    # Baseline
    if args.model == 'TECH':
        train_predictions = [3] * len(train_data)
        dev_predictions = [3] * len(dev_data)
        train_acc = utils.calculate_accuracy(train_predictions, train_labels)
        dev_acc = utils.calculate_accuracy(dev_predictions, dev_labels)
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Dev Accuracy: {dev_acc:.4f}")
        
        # TODO: uncomment these lines after you've implemented macro-F1.
        # dev_f1 = utils.macro_f1(dev_predictions, dev_labels) 
        # print(f"Dev macro-F1: {dev_f1:.4f}")

    elif args.model == 'BOW':
        start_time = time.time()
        
        featurizer = models.BoWFeaturizer(max_vocab_size=5000)
        featurizer.build_vocab(train_data)
        
        model = models.train_torch_model(train_data, dev_data, featurizer,
                                         epochs=10)
        
        train_predictions = []
        dev_predictions = []
        model.eval()
        with torch.no_grad():
            for ex in train_data:
                x = featurizer.get_feature_vector(ex.text).unsqueeze(0)  # (1, vocab)
                logits = model(x)
                pred = torch.argmax(logits, dim=-1).item()
                train_predictions.append(pred)
            for ex in dev_data:
                x = featurizer.get_feature_vector(ex.text).unsqueeze(0)  # (1, vocab)
                logits = model(x)
                pred = torch.argmax(logits, dim=-1).item()
                dev_predictions.append(pred)

        train_acc = utils.calculate_accuracy(train_predictions, train_labels)
        dev_acc = utils.calculate_accuracy(dev_predictions, dev_labels)
        elapsed = time.time() - start_time

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Dev Accuracy: {dev_acc:.4f}")

        # TODO: uncomment these lines after you've implemented macro-F1.
        # dev_f1 = utils.macro_f1(dev_predictions, dev_labels) 
        # print(f"Dev macro-F1: {dev_f1:.4f}")

        print(f"Time elapsed: {elapsed:.2f} seconds")


    elif args.model == 'LR':
        start_time = time.time()
        
        featurizer = models.BoWFeaturizer(max_vocab_size=5000)
        featurizer.build_vocab(train_data)
        
        model = models.train_logistic_regression(train_data, dev_data, featurizer, method="lr",
                                                 epochs=10, lr=0.01)
        
        train_predictions = []
        dev_predictions = []
        for ex in train_data:
            x = featurizer.get_feature_vector(ex.text)
            train_predictions.append(model.predict(x))
        for ex in dev_data:
            x = featurizer.get_feature_vector(ex.text)
            dev_predictions.append(model.predict(x))
            
        train_acc = utils.calculate_accuracy(train_predictions, train_labels)
        dev_acc = utils.calculate_accuracy(dev_predictions, dev_labels)
        elapsed = time.time() - start_time

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Dev Accuracy: {dev_acc:.4f}")

        # TODO: uncomment these lines after you've implemented macro-F1.
        # dev_f1 = utils.macro_f1(dev_predictions, dev_labels) 
        # print(f"Dev macro-F1: {dev_f1:.4f}")

        print(f"Time elapsed: {elapsed:.2f} seconds")

    elif args.model == 'BIGRAM':
        start_time = time.time()
        featurizer = models.BigramFeaturizer(max_vocab_size=20000)
        featurizer.build_vocab(train_data)
        
        model = models.train_logistic_regression(train_data, dev_data, featurizer, method="lr",
                                                 epochs=10)
        
        train_predictions = []
        dev_predictions = []
        for ex in train_data:
            x = featurizer.get_feature_vector(ex.text)
            train_predictions.append(model.predict(x))
        for ex in dev_data:
            x = featurizer.get_feature_vector(ex.text)
            dev_predictions.append(model.predict(x))

        train_acc = utils.calculate_accuracy(train_predictions, train_labels)
        dev_acc = utils.calculate_accuracy(dev_predictions, dev_labels)
        elapsed = time.time() - start_time

        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Dev Accuracy: {dev_acc:.4f}")

        # TODO: uncomment these lines after you've implemented macro-F1.
        # dev_f1 = utils.macro_f1(dev_predictions, dev_labels) 
        # print(f"Dev macro-F1: {dev_f1:.4f}")

        print(f"Time elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()