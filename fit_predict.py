import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from models.cross_validate import cross_validate_rf
# import util

from sklearn.metrics import mean_squared_error

def fit_predict(xs_trn, ys_trn, xs_tst, ys_tst):
    clf = RandomForestRegressor(oob_score=True,
        n_estimators=100, min_samples_split=300, n_jobs=-1)
    # clf = cross_validate_rf(clf, xs_trn, ys_trn, n_iter=1.0, fit_params=None)
    clf.fit(xs_trn, ys_trn)
    preds = clf.predict(X=xs_tst)
    mse = mean_squared_error(ys_tst, preds)
    print('\t%7.15f' % (mse,))
    return preds, clf.oob_prediction_, clf


def fit_predict_lin(xs_trn, ys_trn, xs_tst, ys_tst):
    clf = LinearRegression(fit_intercept=False, n_jobs=1, normalize=False)
    xs_norm = xs_trn / xs_trn.std(axis=0)
    clf.fit(xs_norm, ys_trn)
    clf.coef_ /= xs_trn.std(axis=0)
    preds = clf.predict(X=xs_tst)
    #print(' '.join(['%0.15f' % _ for _ in clf.coef_]))
    mse = mean_squared_error(ys_tst, preds)
    print('\t%7.15f' % (mse,))
    return preds

