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

from sklearn.base import BaseEstimator, TransformerMixin

from model import ModelBase, from_dict

class SplitEstimator(BaseEstimator, TransformerMixin, ModelBase):
    def __init__(self, name, **params):
        self.name = name
        self.inputs = params['inputs'].split()
        self._clf_params  = params['clf_params']
        self.split_on     = self._clf_params['split_on'] # required
        self.remove_split = self._clf_params.get('remove_split', True) # optional
        self.max_buckets  = self._clf_params.get('max_buckets', 100) # optional
        self._n_buckets   = self._clf_params.get('n_buckets', None) # optional
        self._buckets = None
        self._estimators = None

        self.split_model_params = params['clf_params']['split_model']  # will be applied to each estimator


    def get_params(self, deep=True):
        return self._clf_params.copy()

    def set_params(self, **params):
        """
        PASS THROUGH parameters to every estimator
            - what if fit() hasn't been called?
            - what if fit() HAS been called?
        :param params: dict
        :return: self
        """
        self._clf_params = params.copy()
        return self

    def fit(self, X, y=None):
        """
        :param split_on: name of column to split on
        :param n_buckets: if None, group by (discrete) values in column; else
        split into N buckets
        """
        if not self._n_buckets:
            ''' Create buckets.
                Bucket keys are the unique value routed to that bucket.
            '''
            self._buckets = X[self.split_on].unique()
            n_buckets = len(self._buckets)
            assert n_buckets < self.max_buckets, 'TOO MANY BUCKETS: %s' % self._buckets

        # need to do the import here to avoid circular references:
        # construct an estimator for each bucket:
        self._estimators = {bk: from_dict(self.split_model_params)
                            for bk in self._buckets}

        ''' not sure why have explicit call to set_params when can pass to
            constructors (above)
        if self.split_model_params:
            for clf in self._estimators.values():
                clf.set_params(self.split_model_params) # inplace?
        '''

        # subset for each bucket value
        for bk in self._estimators.keys():
            subset = (X[self.split_on] == bk)
            if self.remove_split:
                inputs = [_ for _ in self.inputs if _ != self.split_on]
                self._estimators[bk].fit(X.loc[subset, inputs],
                                         y[subset])
            else:
                self._estimators[bk].fit(X[subset], y[subset])

    def predict(self, X, y=None):
        assert self._estimators, 'fit() must be called before predict()'

        # TODO: align predictions for subsets of elements of y
        ps = pd.Series(index=X.index)
        for bk in self._estimators.keys():
            inputs = self.inputs - self.split_on
            subset = X[self.split_on] == bk
            if self.remove_split:
                ps[subset] = self._estimators[bk].predict(X.loc[subset, inputs])
            else:
                ps[subset] = self._estimators[bk].predict(X[subset])
        assert not np.isnan(ps).any()
        return ps

if __name__ == '__main__':
    X, y = generate_test_data()

    from sklearn.linear_model import LinearRegression
    clf = SplitEstimator('S', LinearRegression)
    clf.fit(df, y)
    print clf.fit_transform(df, y)

    '''
    When applying models in a pipeline (using fit_transform to add a prediction
    column), how do the next models know which of the previous columns are to be
    used for prediction?
    All of them?

    How do you pass fit_params to CV so that all the models in the pipeline can be optmized?
    "The purpose of the pipeline is to assemble several steps that can be cross-
    validated together while setting different parameters. For this, it enables 
    setting parameters of the various steps using their names and the parameter name separated by a __'''
