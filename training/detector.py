import numpy as np


class AnomalyDetector:
    """Wraps a fitted sklearn pipeline and returns score plus anomaly flag."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X):
        self.pipeline.fit(X)
        return self

    def predict(self, X):
        scores = self.pipeline.decision_function(X)
        labels = (scores < 0).astype(np.int64)
        return np.column_stack([scores, labels])
