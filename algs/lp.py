from __future__ import print_function

from model import ModelBase
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars

"""
    def __init__(self, type="LinearRegression"):
        http://stackoverflow.com/questions/2399307/how-to-invoke-the-super-constructor

linear_model.LogisticRegressionCV([Cs, ...])	Logistic Regression CV (aka logit, MaxEnt) classifier.
linear_model.MultiTaskElasticNetCV([...])	Multi-task L1/L2 ElasticNet with built-in cross-validation.
linear_model.MultiTaskLassoCV([eps, ...])	Multi-task L1/L2 Lasso with built-in cross-validation.
linear_model.OrthogonalMatchingPursuitCV([...])	Cross-validated Orthogonal Matching Pursuit model (OMP)
linear_model.RidgeCV([alphas, ...])	Ridge regression with built-in cross-validation.
"""


class LP(ModelBase):
    def __init__(self, name, inputs=None, cv_params={}, fit_intercept=False,
                 normalize=False, n_jobs=-1):
        ModelBase.__init__(self, name, inputs=inputs, cv_params=cv_params)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.n_jobs = n_jobs
        self._clf = None

    def fit(self, X, y, verbosity=0):
        self._clf = LinearRegression(fit_intercept=self.fit_intercept,
                                     normalize=self.normalize, n_jobs=self.n_jobs)
        if self.normalize:
            stdx = X[self.inputs].std(axis=0)
            stdx.replace(0, 1, inplace=True)
            xs_norm = X[self.inputs] / stdx
            self._clf.fit(xs_norm, y)
            self._clf.coef_ /= stdx
        else:
            self._clf.fit(X[self.inputs], y)
        if verbosity > 0:
            print('MSE: %8.2f' % ((y - self._clf.predict(X[self.inputs])) ** 2).mean(),)
        return


class EN(ModelBase):
    def __init__(self, name, inputs=None, cv_params={}, fit_intercept=False,
                 normalize=False, alpha=1.0, l1_ratio=0.5):
        ModelBase.__init__(self, name, inputs=inputs, cv_params=cv_params)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self._clf = None

    def fit(self, X, y, verbosity=0):
        self._clf = ElasticNet(fit_intercept=self.fit_intercept,
                                 normalize=self.normalize,
                                 l1_ratio=self.l1_ratio, alpha=self.alpha)
        self._clf.fit(X[self.inputs], y)
        return


class LL(ModelBase):
    def __init__(self, name, inputs=None, cv_params={}, fit_intercept=False,
                 normalize=True, alpha=1.0):
        ModelBase.__init__(self, name, inputs=inputs, cv_params=cv_params)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.alpha = alpha
        self._clf = None

    def fit(self, X, y, verbosity=0):
        self._clf = LassoLars(fit_intercept=self.fit_intercept,
                                 normalize=self.normalize, alpha=self.alpha)
        self._clf.fit(X[self.inputs], y)
        return
