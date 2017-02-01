from __future__ import print_function

from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint

import tensorflow as tf
from tffm import TFFMRegressor

from model import ModelBase

class FM(ModelBase):
    def get_param_dist(self, xs):
        num_rows = xs.shape[0]
        num_features = xs.shape[1]
        param_dist = {
            'rank': sp_randint(1, num_features),
            'batch_size': sp_randint(1, num_rows),
            'lr': sp_uniform(loc=0.001, scale=0.01),
        }
        return param_dist

    def fit(self, xs_trn, ys_trn, order=2, rank=10, lr=0.001, n_epochs=1,
        batch_size=100, std=0.001, lda=1e-6, log_dir='/tmp/jprior/logs',
            verbosity=0):
        self._clf = TFFMRegressor(
            seed=0,
            order=order,
            rank=rank,
            optimizer=tf.train.FtrlOptimizer(learning_rate=lr),
            n_epochs=n_epochs,
            batch_size=batch_size,
            # smaller init_std -> lower contribution of higher order terms
            init_std=std,
            reg=lda,
            #input_type='sparse',
            log_dir=log_dir,
            verbose=verbosity
        )
        # tffm doesn't deal with DataFrames correctly (although it tries...)
        self._clf.fit(xs_trn.values, ys_trn.values, show_progress=True)
        return
