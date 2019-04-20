import numpy as np
from util import tree_log, mid_point
from typing import TypeVar, Iterable, Tuple, List, Dict
from localtypes import Predictable


"""
Description:    
    We are asked to create a decision stump. From the text:
    Let x denote a a one-dimensional attribute and y denote
    the class label. Suppose we use only one-level binary 
    decision trees, with a test condition x <= k, where k
    is a split position chosen to minimize the entropy of
    the leaf nodes.
    
    Based on this specification, we will not compute
    information gain. Instead, we just compute the entropy
    of the children.
    
Assumptions:
    (1) The data we will test on is continuous. As such, we
        choose to split the data using entropy. This is 
        described in the algorithm, below. 
    (2) Assume binary target.
    (3) Integer targets in range [-1, 1].
    (4) When finding the best split, if there are two splits
        resulting in equal information gain, the second is
        chosen. This is arbitrary, and would need adjustment.
        
Decision Stump Algorithm:
    (1) Sort the targets by their inputs.
    (2) Find the indices where the target changes.
    (3) For each target change index:
    (4)     Compute the midpoint of the index, and its predecessor.
    (5)     Compute the entropy of that split.
"""


class HomogeneousClassError(Exception):
    pass


def sort_data(predictors: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert predictors.shape[0] == targets.shape[0]
    sorted_indices = np.argsort(predictors)
    return predictors[sorted_indices], targets[sorted_indices]


def find_delta_indices(targets: np.ndarray) -> List[int]:
    indices = []
    for i in range(1, targets.shape[0]):
        if targets[i] != targets[i - 1]:
            indices.append(i)
    return indices


def test_split(data: np.ndarray, index: int) -> Tuple[np.ndarray, np.ndarray]:
    return (
        data[0:index], data[index:len(data)]
    )


def class_counts(data: np.ndarray) -> Dict:
    counts = {}
    keys, values = np.unique(data, return_counts=True)
    for key, value in zip(keys, values):
        counts[key] = value
    return counts


def majority_class(data: np.ndarray) -> int:
    classes, counts = np.unique(data, return_counts=True)
    max_index = np.argmax(counts)
    return classes[max_index]


class StumpClassifier:
    _target_range = [-1, 1]

    def __init__(self):
        self._decision_boundary = None
        self._predictors, self._targets = [None] * 2
        self._left_prediction, self._right_prediction = [None] * 2
        self._information = 1.0

    @property
    def decision_boundary(self) -> float:
        return self._decision_boundary

    @property
    def information(self) -> float:
        return self._information

    def fit(self, predictors: np.ndarray, targets: np.ndarray) -> None:
        self._predictors = np.copy(predictors)
        self._targets = np.copy(targets)
        self._predictors, self._targets = sort_data(self._predictors, self._targets)
        self._find_best_split(self._predictors, self._targets)

    def predict(self, predictors: Predictable) -> np.ndarray:
        try:
            _ = iter(predictors)
        except TypeError:
            predictors = [predictors]

        return np.array(
            [self._predict_single(predictor) for predictor in predictors]
        )

    def _predict_single(self, predictor: float) -> int:
        prediction = None
        if predictor <= self.decision_boundary:
            prediction = self._left_prediction
        else:
            prediction = self._right_prediction
        return prediction

    def _find_best_split(self, predictors: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
        delta_indices = find_delta_indices(targets)
        if len(delta_indices) == 0:
            raise HomogeneousClassError()
        best_index, best_info = -1, -1
        for index in delta_indices:
            left_data, right_data = test_split(targets, index)
            info = self._info(left_data, right_data)
            if info >= best_info:
                best_index = index
                best_info = info
        self._set_model_params(best_index, best_info)

    def _info(self, left_data: np.ndarray, right_data: np.ndarray) -> float:
        total = len(self._targets)
        left_len, right_len = len(left_data), len(right_data)
        left_p, right_p = left_len / total, right_len / total
        parent_entropy = self._entropy(self._targets)
        return (
            parent_entropy - (left_p * self._entropy(left_data) + right_p * self._entropy(right_data))
        )

    def _entropy(self, data: np.ndarray) -> float:
        sigma = 0
        total = len(data)
        for target, target_count in class_counts(data).items():
            p = target_count / total
            sigma += -(p * tree_log(p))
        return sigma

    def _set_model_params(self, index: int, info_gain: float) -> None:
        self._decision_boundary = mid_point(self._predictors[index - 1], self._predictors[index])
        self._information = info_gain
        left_data, right_data = test_split(self._targets, index)
        self._left_prediction = majority_class(left_data)
        self._right_prediction = majority_class(right_data)

    def __repr__(self):
        def stringify_array(a):
            return [str(x) for x in a]

        return 'decision_boundary: {}\nPred: {}\nTarg: {}'.format(
            self.decision_boundary,
            '|'.join(stringify_array(self._predictors)),
            '|'.join(stringify_array(self._targets))
        )


if __name__ == '__main__':
    c = StumpClassifier()
    # pred = np.arange(1, 11) / 10
    # targets = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, 1])
    # print(pred.shape)
    # print(targets.shape)
    # pred = np.array([0.1, .2, .2, .3, .4, .4, .5, .6, .9, .9])
    # targets = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, 1])
    pred = np.array([.1, .2, .3, .4, .5, .8, .9, 1, 1, 1])
    targets = np.array([1, 1, 1, -1, -1, 1, 1, 1, 1, 1])
    c.fit(pred, targets)
    print(c)
