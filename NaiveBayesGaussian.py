from __future__ import annotations

import math
from typing import Tuple, Dict, Any

import numpy as np


class NaiveBayesGaussian:
    """
    Gaussian Naive Bayes for continuous features.

    Uses variance smoothing to stabilize likelihoods.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.class_prior_log_: Dict[Any, float] = {}
        self.mean_: Dict[Any, np.ndarray] = {}
        self.var_: Dict[Any, np.ndarray] = {}

    def fit(
        self, X: np.ndarray, y: np.ndarray
    ) -> "NaiveBayesGaussian":

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        # Class priors
        for c in self.classes_:
            Xc = X[y == c]

            self.class_prior_log_[c] = math.log(
                len(Xc) / len(X)
            )

            # Mean and variance for each feature
            mu = Xc.mean(axis=0)
            var = Xc.var(axis=0) + self.var_smoothing

            self.mean_[c] = mu
            self.var_[c] = var

        return self

    def _log_gaussian_likelihood(
        self, X: np.ndarray, c: Any
    ) -> np.ndarray:

        mu = self.mean_[c]
        var = self.var_[c]

        return -0.5 * (
            np.log(2 * math.pi * var)
            + ((X - mu) ** 2) / var
        )

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:

        X = np.asarray(X, dtype=float)

        log_probs = []

        for c in self.classes_:
            log_likelihood = self._log_gaussian_likelihood(X, c)

            # Sum likelihoods of all features
            total_log_likelihood = log_likelihood.sum(axis=1)

            # Add class prior
            total = (
                self.class_prior_log_[c]
                + total_log_likelihood
            )

            log_probs.append(total)

        return np.column_stack(log_probs)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        log_probs = self.predict_log_proba(X)

        # Numerical stability
        max_log = np.max(log_probs, axis=1, keepdims=True)

        probs = np.exp(log_probs - max_log)

        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:

        log_probs = self.predict_log_proba(X)

        indices = np.argmax(log_probs, axis=1)

        return self.classes_[indices]
