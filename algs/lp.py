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
    def get_param_dist(self, xs):
        total_features = xs.shape[1]
        min_features = int(max(1, 0.10 * total_features))
        max_features = int(max(1, 0.70 * total_features))
        feature_step = int(round((max_features - min_features) / 10, 0))
        param_dist = {
            'max_features': range(min_features, max_features, feature_step),
            'min_samples_leaf': range(1, 1000, 50)
            # "max_features": sp_randint(min_features, max_features),
            # "min_samples_leaf": sp_randint(1, 1000),
            # "learning_rate": sp_uniform(loc=0.01, scale=0.19),
            # "subsample": sp_uniform(loc=0.01, scale=0.89)
        }
        return param_dist

    def fit(self, xs_trn, ys_trn):
        self._clf = LinearRegression(fit_intercept=False, n_jobs=-1,
                                     normalize=False)
        stdx = xs_trn.std(axis=0)
        stdx.replace(0, 1, inplace=True)
        xs_norm = xs_trn / stdx
        self._clf.fit(xs_norm, ys_trn)
        self._clf.coef_ /= stdx
        return
