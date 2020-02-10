from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np


class SBS:
    """Class for running best subsets using any classifier"""

    def __init__(
        self,
        estimator,
        k_features,
        scoring=roc_auc_score,
        test_size=0.25,
        random_state=1,
    ):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.random_state = random_state
        self.test_size = test_size

        """ Initializes the SBS object

        Parameters
        ----------
        estimator : sklearn classifier object
            The classifier object to use for best subsets
        k_features : int
            The number of features to use for the subsets e.g. 5 would try all combinations of 5 features
        scoring : str
            Method of scoring e.g. 'roc_auc'
        test_size : float
            Percentage of the data to use as a test set
        random_state : int
            Random number seed

        """

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score