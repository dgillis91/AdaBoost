import numpy as np
from util import tree_log, mid_point


# TODO: Do we need to update the split rules? For example, using majority votes from the parent.
# TODO: Write doc strings.
# TODO: Add assertions for data shape.
class Node:
    def __init__(self, predictors, targets, classes=np.array([-1, 1])):
        # TODO: Data has to be sorted.
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
    s = DecisionStump(
            np.array([.5, 3.0, 4.5, 4.6, 4.9, 5.2, 5.3, 5.5, 7.0, 9.5]),
            np.array([-1, -1, 1, 1, 1, -1, -1, 1, -1, -1])
    )

    s.fit()
    print(s.predict(5))
    print(s.decision_boundary)
