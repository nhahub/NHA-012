import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class MeanEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        preds = np.column_stack([model.predict(X) for model in self.models])
        return preds.mean(axis=1)

class BoolToIntImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.apply(lambda col: col.map({True: 1, False: 0})).values
