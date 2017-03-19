from __future__ import print_function

import numpy as np

class ModelBase(object):
    def __init__(self, name, params={}):
        self.name = name
        self._clf = None
        self._params = params

    def get_model(self):
        return self._clf

    def get_param_dist(self, xs):
        return {}

    def fit(self, X, y):
        self._clf = None
        self._y_shape = y.shape[0]
        return

    def fit_transform(self, X, y):
        self.fit(X, y)
        ps = self.predict(X)
        assert self.name not in X.columns
        # return np.concatenate((X, ps), axis=1)
        X[self.name] = ps  # MODIFYING THE PASSED-IN X!!
        return X

    def transform(self, X):
        ps = self.predict(X)
        # dual purpose of ensuring X follows Dataframe interface,
        # as well as doesn't already have a column called <self.name>
        assert self.name not in X.columns
        # return np.concatenate((X, ps), axis=1)
        X[self.name] = ps  # MODIFYING THE PASSED-IN X!!
        return X

    def predict(self, xs):
        return self._clf.predict(X=xs)

    def score(self, ):
        pass

    def print(self):
        print(self._clf)
