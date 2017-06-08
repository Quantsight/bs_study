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

class SplitEstimator(TransformerMixin, ModelBase):
    def __init__(self, name, split_on, inputs=None, max_buckets=100, n_buckets=None,
        sub_params={}, cv_params={}, scorer=None, **clone_args):

        ModelBase.__init__(self, name, sub_params, inputs, cv_params, scorer)
        self.split_on     = split_on
        self.max_buckets  = max_buckets
        self._n_buckets   = n_buckets
        self._buckets = None
        self._estimators = None

        # only used when clone() is called and params from get_params passed in:
        for (k,v) in clone_args.items():
            k = k.replace('_sub__','')
            self._sub_params[k] = v

        # (cannot create estimators until we have the split-on data)

    '''
    def __init__(self, name, **params):
        super(KFold, self).__init__(n_splits, shuffle, random_state)
        #!!! seem to not be able to pass **params, must use named args directly.
        self.name = name
        self._sub_params  = params['sub_params']
        self.split_on     = params['split_on'] # required
        self.max_buckets  = params.get('max_buckets', 100) # optional
        self._n_buckets   = params.get('n_buckets', None) # optional
        self._buckets = None
        self._estimators = None

        # will be applied to each estimator
        # (cannot create estimators until we have the split-on data)
    '''

    def get_params(self, deep=True):
        # SplitEstimator has no 'optimizable' parameters; only its sub-estimators do
        out = {}
        #params = from_dict(self._sub_params).get_params()
        for (key, value) in self._sub_params.items():
            out['_sub__%s' % (key,)] = value
        out['name']        = self.name
        out['split_on']    = self.split_on
        out['inputs']      = self.inputs
        out['max_buckets'] = self.max_buckets
        out['_n_buckets']  = self._n_buckets
        return out

    def set_params(self, **params):
        """
        TODO: PASS THROUGH parameters to every estimator
            - what if fit() hasn't been called?
            - what if fit() HAS been called?
        :param params: dict
        :return: self
        """
        self._sub_params = {}
        for (k, v) in params.items():
            new_key = k[6:] # strip off leading '_sub__'
            self._sub_params[new_key] = v
        return self

    def get_param_dist(self, xs):
        print 'how do params get updated? each fit() call re-initializes from params'
        dist = from_dict(self._sub_params).get_param_dist(xs)
        return dist #['%s__%s' % (self.name, k) for k in dist.keys()]

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

        # construct an estimator for each bucket:
        self._estimators = {bk: from_dict(self._sub_params)
                            for bk in self._buckets}

        ''' not sure why have explicit call to set_params when can pass to
            constructors (above)
        if self._sub_params:
            for clf in self._estimators.values():
                clf.set_params(self._sub_params) # inplace?
        '''

        # subset for each bucket value
        for bk in self._estimators.keys():
            subset = (X[self.split_on] == bk)
            self._estimators[bk].fit(X[subset], y[subset])

    def predict(self, X, y=None):
        assert self._estimators, 'fit() must be called before predict()'

        # TODO: align predictions for subsets of elements of y
        ps = pd.Series(index=X.index)
        for bk in self._estimators.keys():
            subset = X[self.split_on] == bk
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
