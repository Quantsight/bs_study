"""
http://stats.stackexchange.com/questions/188817/splitting-pipeline-in-sklearn

implement a "splitter" transform that is given a column to split on,
and splits into N equal sized buckets, then creates models for each split
(remembers what the split criteria is so that OOS rows can be routed correctly)

YAGNI/TODO: split on continuous features [low,high)
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline

class SplitEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, col_name, clf_class, n_buckets=None):
        """
        :param col_name: name of column to split on
        :param n_buckets: if None, group by (discrete) values in column; else
        split into N buckets
        """
        self.col_name = col_name
        self.clf_class = clf_class
        self.n_buckets = n_buckets
        self.buckets = None
        self.estimators = None

    def fit(self, X, y=None):
        if not self.n_buckets:
            self.buckets = X[self.col_name].unique()
            n_buckets = len(self.buckets)
            assert n_buckets < 100, 'TOO MANY BUCKETS: %s' % self.buckets

        # construct an estimator for each bucket:
        params = self.get_params()
        self.estimators = {bv: self.clf_class().set_params(params) for bv in 
                           self.buckets}
        for k in self.estimators.keys():
            # Here some logic to divide dataset
            subset = X[self.col_name] == k
            self.estimators[k].fit(X[subset], y[subset])

    def predict(self, X, y=None):
        # TODO: align predictions for subsets of elements of y
        ps = pd.Series(index=X.index)
        for k in self.estimators.keys():
            # Here some logic to divide dataset
            subset = X[self.col_name] == k
            ps[subset] = self.estimators[k].predict(X[subset], y[subset])
        assert np.isnan(ps).any() == False
        return ps

if __name__ == '__main__':
    clf = SplitEstimator('x2', LogisticRegression)
