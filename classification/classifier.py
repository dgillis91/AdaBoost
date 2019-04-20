import numpy as np
from util import tree_log, mid_point
from typing import TypeVar, Iterable, Tuple, List, Dict

Predictable = TypeVar('Predictable', float, Iterable, np.ndarray)

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
        data[0:index], data[index:len(targets)]
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


# TODO: Do we need to update the split rules? For example, using majority votes from the parent.
# TODO: Write doc strings.
class Node:
    def __init__(self, predictors, targets, classes=np.array([-1, 1])):
        assert len(predictors) == len(targets)
        self.predictors = np.copy(predictors)
        self.targets = np.copy(targets)
        self._sort_data_inplace()
        self.classes = classes
        self._class_counts = self._get_class_counts()
        self._majority_vote = self._compute_majority_class_count()

    @property
    def class_counts(self):
        """
        :return: Dict with count of data for each target.
        """
        return self._class_counts

    @property
    def sample_count(self):
        """
        Number of records in the node's data.
        :return:
        """
        return len(self.predictors)

    @property
    def majority_vote(self):
        return self._majority_vote

    def _sort_data_inplace(self):
        sorted_indices = np.argsort(self.predictors)
        self.predictors = self.predictors[sorted_indices]
        self.targets = self.targets[sorted_indices]

    def _get_class_counts(self):
        """
        Compute the number of records in each class.
        :return: Dictionary with keys as classes, and values as the number of samples with that target.
        """
        class_counts = self._init_class_counts()
        for target in self.targets:
            class_counts[target] += 1
        return class_counts

    def _init_class_counts(self):
        """
        Iterate over the classes passed to this node and create a dict with
        a default of zero for each class. Note that using a defaultdict would
        not work. We require each class to be initialized to zero for our
        impurity measure.
        :return: A dictionary with a key for each class, and zero for the value.
        """
        class_counts = {}
        for _class in self.classes:
            class_counts[_class] = 0
        return class_counts

    def impurity(self):
        """
        Compute the Entropy Impurity.
        :return:
        """
        sigma = 0
        total_records = self.sample_count
        for target, target_count in self.class_counts.items():
            target_probability = target_count / total_records
            sigma += target_probability * tree_log(target_probability)
        return -sigma

    def _compute_majority_class_count(self):
        current_count = -1
        for target, target_count in self._class_counts.items():
            if current_count < target_count:
                current_count = target_count
                majority = target
        return majority


class ClassifierNode(Node):
    def __init__(self, predictors, targets, classes=np.array([-1, 1])):
        super(ClassifierNode, self).__init__(predictors, targets, classes)


class RootNode(Node):
    def __init__(self, targets, predictors):
        super(RootNode, self).__init__(targets, predictors, np.unique(predictors))
        self.left_node = self.right_node = None
        self.decision_boundary = None

    def fit(self):
        self.find_split()

    def predict(self, predictor):
        try:
            _ = iter(predictor)
        except TypeError:
            predictor = [predictor]

        predictions = []
        for pred in predictor:
            predictions.append(self._predict_singular(pred))

        return np.array(predictions)

    def _predict_singular(self, value):
        if value < self.decision_boundary:
            return self.left_node.majority_vote
        else:
            return self.right_node.majority_vote

    def find_split(self):
        """
        :return: the split which minimizes the impurity, as measured by entropy.
        """
        potential_splits = self.find_delta_indices()
        if len(potential_splits) == 0:
            raise HomogeneousClassError
        gain = -1
        for potential_split in potential_splits:
            data_left, data_right = self.make_nodes(potential_split)
            new_gain = self.information_gain(data_left, data_right)
            if new_gain > gain:
                keep_left, keep_right = data_left, data_right
                decision_split = potential_split
                gain = new_gain
        self.left_node = keep_left
        self.right_node = keep_right
        self.decision_boundary = mid_point(self.predictors[decision_split - 1], self.predictors[decision_split])

    def make_nodes(self, split):
        """
        Split the data into two nodes at index `split`.
        :param split: The index to split the data about.
        :param split: The index to split the data about.
        :return: A tuple of the form (left, right).
        """
        return (
            ClassifierNode(
                self.predictors[:split], self.targets[:split], self.classes
            ),
            ClassifierNode(
                self.predictors[split:], self.targets[split:], self.classes
            )
        )

    def information_gain(self, left, right):
        """
        :param left: left child
        :param right: right child
        :return: The information gain of the parent: I(Parent)
        """
        weighted_child_entropy = RootNode.average_entropy(left, right)
        return self.impurity() - weighted_child_entropy

    @staticmethod
    def average_entropy(left, right):
        """
        :param left: left node
        :param right: right node
        :return: The weighted average of the children
        """
        count_left = left.sample_count
        count_right = right.sample_count
        count_total = count_left + count_right
        return (
            (count_left / count_total) * left.impurity() +
            (count_right / count_total) * right.impurity()
        )

    def find_delta_indices(self):
        """
        Find the indices where the class value changes.

        >>> d = RootNode(
        >>>     np.array(0.1, 0.2, 0.3, 0.4]),
        >>>     np.array([ 1 ,  1 ,  1 , -1 ])
        >>> )
        >>> print(d.find_delta_indices()) # [3]
        :return: The indices where class values change.
        """
        indices = []
        # TODO: Use enumerate here.
        for i in range(1, self.sample_count):
            if self.targets[i] != self.targets[i - 1]:
                indices.append(i)
        return indices

    # TODO: Add accuracy, etc.


class DecisionStump(RootNode):
    def __init__(self, *args, **kwargs):
        super(DecisionStump, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    c = StumpClassifier()
    pred = np.arange(1, 11) / 10
    targets = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, 1])
    print(pred.shape)
    print(targets.shape)
    c.fit(pred, targets)
    print(c)
