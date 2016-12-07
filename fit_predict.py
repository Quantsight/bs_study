from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import util


def fit_predict(df, xs_trn, ys_trn, xs_tst, ys_tst):
    clf = LinearRegression()
    # clf = RandomForestRegressor()
    with util.timed_execution('Fit'):
        clf.fit(xs_trn, ys_trn)
    preds = clf.predict(xs_tst)
    mse = mean_squared_error(ys_tst, preds)
    print('MSE:%10.8f' % (mse,))
