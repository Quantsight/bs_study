from __future__ import print_function

class ModelBase:
    def __init__(self):
        self._clf = None

    def get_model(self):
        return self._clf

    def get_param_dist(self, xs):
        return {}

    def fit(self, xs_trn, ys_trn):
        self._clf = None
        return

    def predict(self, xs):
        return self._clf.predict(X=xs)
