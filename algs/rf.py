from __future__ import print_function

from model import ModelBase

from sklearn.ensemble import RandomForestRegressor

"""
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(ys_trn, ts_preds)
print('\t%7.3f' % (mse,))
"""

class RF(ModelBase):
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
        self._clf = RandomForestRegressor(oob_score=False, n_estimators=128,
                                          criterion='mse', min_samples_leaf=100, n_jobs=-1)
        self._clf.fit(xs_trn, ys_trn)
        return
