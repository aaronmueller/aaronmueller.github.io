# CS505: NLP - Spring 2026

def calculate_accuracy(predictions, labels):
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    return correct / len(predictions)

def macro_f1(predictions, labels, num_classes=4):
    # TODO: implement the macro-F1 score.
    # Recall that this involves computing the F1 score separately for
    # each label, and then taking the macroaverage. Return the macro-F1
    # score as a floating-point number.
    # STUDENT START --------------------------------------
    pass
    # STUDENT END -------------------------------------------