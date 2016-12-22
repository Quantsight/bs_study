from __future__ import print_function

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# from models.cross_validate import cross_validate_rf
# import util

from sklearn.metrics import mean_squared_error

def fit_predict_rf(xs_trn, ys_trn, xs_tst, ys_tst):
    clf = RandomForestRegressor(oob_score=True,
        n_estimators=256, min_samples_split=100, n_jobs=-1)
    # clf = cross_validate_rf(clf, xs_trn, ys_trn, n_iter=1.0, fit_params=None)
    clf.fit(xs_trn, ys_trn)
    # tr_preds = clf.predict(X=xs_trn)
    tr_preds = clf.oob_prediction_
    ts_preds = clf.predict(X=xs_tst)
    mse = mean_squared_error(ys_tst, ts_preds)
    print('\t%7.3f' % (mse,))
    return ts_preds, tr_preds, clf


def fit_predict_lin(xs_trn, ys_trn, xs_tst, ys_tst):
    clf = LinearRegression(fit_intercept=False, n_jobs=1, normalize=False)
    stdx = xs_trn.std(axis=0)
    stdx.replace(0, 1, inplace=True)
    xs_norm = xs_trn / stdx
    clf.fit(xs_norm, ys_trn)
    clf.coef_ /= stdx
    preds = clf.predict(X=xs_tst)
    tr_preds = clf.predict(X=xs_trn)
    #print(' '.join(['%0.15f' % _ for _ in clf.coef_]))
    mse = mean_squared_error(ys_tst, preds)
    print('\t%7.3f' % (mse,))
    return preds, tr_preds, clf

