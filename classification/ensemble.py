import numpy as np
from classifier import StumpClassifier, HomogeneousClassError
from localtypes import Predictable
from typing import Tuple
import math


class AdaBoost:
    def __init__(self, boosting_rounds=10):
        self._predictors, self._targets, self._sample_indices = [None] * 3
        self.boosting_rounds = boosting_rounds
        self.ensemble = []
        self.alphas = []

    @staticmethod
    def uniform_probability_list(n_samples):
        sample_weight = 1 / n_samples
        return np.array([sample_weight] * n_samples)

    def fit(self, predictors: np.ndarray, targets: np.ndarray, verbose = True) -> None:
        self._initialize_data(predictors, targets)
        sample_weights = AdaBoost.uniform_probability_list(len(self._targets))
        boosting_round = 0
        while boosting_round < self.boosting_rounds:
            sample_predictors, sample_targets = self._get_sample(sample_weights)
            stump = StumpClassifier()
            try:
                stump.fit(sample_predictors, sample_targets)
            except HomogeneousClassError:
                sample_weights = AdaBoost.uniform_probability_list(len(self._targets))
                continue
            predictions = stump.predict(self._predictors)
            misclassed = self._misclassed_predictions(predictions)
            weighted_error = self._weighted_error(misclassed, sample_weights)
            if weighted_error >= .5:
                sample_weights = self.uniform_probability_list(len(self._targets))
                continue
            else:
                boosting_round += 1
                alpha = .5 * math.log((1 - weighted_error) / weighted_error)
                self._add_model(stump, alpha)
                self._update_weights(sample_weights, misclassed, alpha)
                if verbose:
                    def stringify_array(a):
                        return [str(x) for x in a]
                    print(
                        'Alpha: {}\nError: {}'.format(
                            alpha, weighted_error
                        )
                    )
                    print('Weights:')
                    print(
                        '|'.format(stringify_array(sample_weights))
                    )
                    print(stump)

    def predict(self, values):
        try:
            _ = iter(values)
        except TypeError:
            values = [values]
        predictions = []
        for value in values:
            predictions.append(self._majority_vote(value))
        return np.array(predictions)

    def _majority_vote(self, value):
        sigma = 0
        for alpha, model in zip(self.alphas, self.ensemble):
            sigma += (alpha * model.predict(value)[0])
        if sigma < 0:
            return -1
        else:
            return 1

    def _initialize_data(self, predictors: np.ndarray, targets: np.ndarray) -> None:
        self._predictors = np.copy(predictors)
        self._targets = np.copy(targets)
        self._sample_indices = list(range(len(targets)))

    def _get_sample(self, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        random_indices = np.random.choice(self._sample_indices, size=len(self._targets), replace=True, p=probabilities)
        return self._predictors[random_indices], self._targets[random_indices]

    def _misclassed_predictions(self, predictions):
        return self._targets != predictions

    @staticmethod
    def _weighted_error(misclassed_selectors: np.ndarray, sample_weights: np.ndarray):
        misclassed_bitmap = misclassed_selectors.astype(np.int)
        return np.dot(sample_weights, misclassed_bitmap)

    def _add_model(self, model, alpha):
        self.ensemble.append(model)
        self.alphas.append(alpha)

    def _update_weights(self, weights, misclassed, alpha):
        new_weights = []
        for is_misclassed, weight in zip(misclassed, weights):
            if is_misclassed:
                a_exp = -alpha
            else:
                a_exp = alpha
            new_weights.append(weight * math.exp(a_exp))
        new_weights = np.array(new_weights)
        new_weights /= new_weights.sum()
        return new_weights

# TODO: Step through code.
# TODO: Add the verbose requirement from the project text.
class AdaBoostClassifier:
    def __init__(self, predictors, targets, boosting_rounds=10):
        self._sample_count = len(targets)
        assert self._sample_count == len(predictors)
        self.boosting_rounds = boosting_rounds
        self.predictors = predictors
        self.targets = targets
        self.sample_indices = list(range(self.sample_count))
        self.ensemble = []
        self.alphas = []

    @property
    def sample_count(self):
        return self._sample_count

    @staticmethod
    def uniform_probability_list(n_samples):
        sample_weight = 1 / n_samples
        return np.array([sample_weight for _ in range(n_samples)])

    def fit(self):
        # Initialize uniform random weights
        sample_weights = AdaBoostClassifier.uniform_probability_list(self.sample_count)
        # Iterate over the boosting rounds
        boosting_round = 0
        while boosting_round < self.boosting_rounds:
            # Get the samples
            sample_predictors, sample_targets = self._get_sample(sample_weights)
            # Train a stump and get predictions (on the whole training set).
            stump = DecisionStump(sample_predictors, sample_targets)
            try:
                stump.fit()
            except HomogeneousClassError:
                sample_weights = AdaBoostClassifier.uniform_probability_list(self.sample_count)
                continue
            predictions = stump.predict(self.predictors)
            misclassed_selectors = self._misclassed_predictions(predictions)
            weighted_error = AdaBoostClassifier._weighted_error(misclassed_selectors, sample_weights)
            if weighted_error >= .5:
                sample_weights = AdaBoostClassifier.uniform_probability_list(self.sample_count)
                continue
            else:
                boosting_round += 1
                alpha = .5 * math.log((1 - weighted_error) / weighted_error)
                self._add_model(stump, alpha)
                sample_weights = self._update_weights(sample_weights, misclassed_selectors, alpha)

    def predict(self, values):
        try:
            _ = iter(values)
        except TypeError:
            values = [values]
        predictions = []
        for value in values:
            predictions.append(self._majority_vote(value))
        return np.array(predictions)

    def _majority_vote(self, value):
        sigma = 0
        for alpha, model in zip(self.alphas, self.ensemble):
            sigma += (alpha * model.predict(value)[0])
        if sigma < 0:
            return -1
        else:
            return 1

    def _add_model(self, model, alpha):
        self.ensemble.append(model)
        self.alphas.append(alpha)

    def _get_sample(self, probabilities):
        """
        :param probabilities: The distribution of weights for choosing from the training data.
        :return: The randomly chosen indices.
        """
        random_indices = np.random.choice(self.sample_indices, size=self.sample_count, replace=True, p=probabilities)
        return self.predictors[random_indices], self.targets[random_indices]

    @staticmethod
    def _weighted_error(misclassed_selectors: np.ndarray, sample_weights: np.ndarray):
        misclassed_bitmap = misclassed_selectors.astype(np.int)
        return np.dot(sample_weights, misclassed_bitmap)

    def _misclassed_predictions(self, predictions):
        return self.targets != predictions

    def _update_weights(self, weights, misclassed, alpha):
        new_weights = []
        for is_misclassed, weight in zip(misclassed, weights):
            if is_misclassed:
                a_exp = -alpha
            else:
                a_exp = alpha
            new_weights.append(weight * math.exp(a_exp))
        new_weights = np.array(new_weights)
        new_weights /= new_weights.sum()
        return new_weights


if __name__ == '__main__':
    predictors = np.array([.5, 3.0, 4.5, 4.6, 4.9, 5.2, 5.3, 5.5, 7.0, 9.5])
    targets = np.array([-1, -1, 1, 1, 1, -1, -1, 1, -1, -1])
    classifier = AdaBoost(30)
    classifier.fit(predictors, targets, verbose=False)
    print(classifier.predict(predictors))

