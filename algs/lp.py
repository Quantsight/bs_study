from __future__ import print_function

from sklearn.linear_model import LinearRegression

from model import ModelBase

"""
http://stackoverflow.com/questions/8718885/import-module-from-string-variable

    def __init__(self, type="LinearRegression"):
        http://stackoverflow.com/questions/2399307/how-to-invoke-the-super-constructor

linear_model.ElasticNetCV([l1_ratio, eps, ...])	Elastic Net model with iterative fitting along a regularization path
linear_model.LarsCV([fit_intercept, ...])	Cross-validated Least Angle Regression model
linear_model.LassoCV([eps, n_alphas, ...])	Lasso linear model with iterative fitting along a regularization path
linear_model.LassoLarsCV([fit_intercept, ...])	Cross-validated Lasso, using the LARS algorithm
linear_model.LogisticRegressionCV([Cs, ...])	Logistic Regression CV (aka logit, MaxEnt) classifier.
linear_model.MultiTaskElasticNetCV([...])	Multi-task L1/L2 ElasticNet with built-in cross-validation.
linear_model.MultiTaskLassoCV([eps, ...])	Multi-task L1/L2 Lasso with built-in cross-validation.
linear_model.OrthogonalMatchingPursuitCV([...])	Cross-validated Orthogonal Matching Pursuit model (OMP)
linear_model.RidgeCV([alphas, ...])	Ridge regression with built-in cross-validation.
linear_model.RidgeClassifierCV([alphas, ...])	Ridge classifier with built-in cross-validation.
"""

class LP(ModelBase):
    def __init__(self, name, inputs=None, cv_params={},
        fit_intercept=False, normalize=False, n_jobs=-1):
        ModelBase.__init__(self, name, inputs=inputs, cv_params=cv_params)
        self.fit_intercept = fit_intercept
        self.normalize     = normalize
        self.n_jobs        = n_jobs

    ''' this is handled by cv_params (in the .yaml files)
    def get_param_dist(self, X):
        param_dist = {
            'fit_intercept':[True, False],
            'normalize':[True, False],
        }
        return param_dist
    '''

    def fit(self, X, y, verbosity=0):
        self._clf = LinearRegression(fit_intercept=self.fit_intercept, normalize=
            self.normalize, n_jobs=self.n_jobs)
        stdx = X[self.inputs].std(axis=0)
        stdx.replace(0, 1, inplace=True)
        xs_norm = X[self.inputs] / stdx
        self._clf.fit(xs_norm, y)
        self._clf.coef_ /= stdx
        if verbosity > 0:
            print('MSE: %8.2f' % ((y - self._clf.predict(X[self.inputs])) ** 2).mean(),)
        return
