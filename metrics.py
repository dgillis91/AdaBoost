import numpy as np


def accuracy_score(true_targets, predicted_targets):
    total = len(true_targets)
    assert total == len(predicted_targets)
    return np.count_nonzero(true_targets == predicted_targets) / total
