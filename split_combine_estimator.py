"""
http://stats.stackexchange.com/questions/188817/splitting-pipeline-in-sklearn

implement a "splitter" transform that is given a column to split on,
and splits into N equal sized buckets, then creates models for each split
(remembers what the split criteria is so that OOS rows can be routed correctly)

? where does extra column (full of predictions) get added? Through fit_transform?

YAGNI/TODO: split on continuous features [low,high)
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline

class SplitEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, clf_class, n_buckets=None, max_buckets=100):
        """
        :param col_name: name of column to split on
        :param n_buckets: if None, group by (discrete) values in column; else
        split into N buckets
        """
        self.col_name = col_name
        self.clf_class = clf_class

        self._n_buckets = n_buckets
        self._buckets = None
        self._estimators = None
        self._params = None  # will be applied to each estimator
        self.max_buckets = max_buckets

    def get_params(self, deep=True):
        return self._params.copy()

    def set_params(self, **params):
        """
        apply parameters to every estimator
            - what if fit() hasn't been called?
            - what if fit() HAS been called?
        :param params: dict
        :return: self
        """
        self._params = params.copy()
        return self

    def fit(self, X, y=None):
        if not self._n_buckets:
            ''' Create buckets.
                Bucket keys are the unique value routed to that bucket.
            '''
            self._buckets = X[self.col_name].unique()
            n_buckets = len(self._buckets)
            assert n_buckets < self.max_buckets, 'TOO MANY BUCKETS: %s' % self._buckets

        # construct an estimator for each bucket:
        self._estimators = {bk:self.clf_class() for bk in self._buckets}

        if self._params:
            for clf in self._estimators.values():
                clf.set_params(self._params) # inplace?

        # subset for each bucket value
        for bk in self._estimators.keys():
            # Here some logic to divide dataset
            subset = X[self.col_name] == bk
            self._estimators[bk].fit(X[subset], y[subset])

    def predict(self, X, y=None):
        assert self._estimators, 'fit() must be called before predict()'

        # TODO: align predictions for subsets of elements of y
        ps = pd.Series(index=X.index)
        for bk in self._estimators.keys():
            subset = X[self.col_name] == bk
            ps[subset] = self._estimators[bk].predict(X[subset])
        assert np.isnan(ps).any() == False
        return ps

    def fit_transform(self, X, y=None):
        """
        fit() -> transform() -> ... add/remove columns to X
        """
        self.fit(X, y)
        ps = self.predict(X)
        return np.concatenate((X, ps), axis=1)


if __name__ == '__main__':
    ROWS = 1000
    SPLITS = 10
    df = pd.DataFrame(np.random.randn(ROWS,4),columns=list('ABCD'))
    split_vals = []
    for _ in range(SPLITS):
        split_vals.extend([_]*(ROWS/SPLITS))
    df['S'] = split_vals
    y = pd.Series(np.random.randn(ROWS))
    clf = SplitEstimator('S', LinearRegression)
    clf.fit(df, y)
    print clf.fit_transform(df, y)