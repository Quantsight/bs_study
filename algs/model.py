from __future__ import print_function

class ModelBase(object):
    def __init__(self, params={}):
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
        ps = self.predict(X).values.reshape(self._y_shape, 1)
        return np.concatenate((X, ps), axis=1)

    def transform(self, X):
        ps = self.predict(X).values.reshape(self._y_shape, 1)
        return np.concatenate((X, ps), axis=1)

    def predict(self, xs):
        return self._clf.predict(X=xs)

    def score(self, ):
        pass

    def print(self):
        print(self._clf)
