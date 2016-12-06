from __future__ import print_function

import joblib
import numpy as np
import os
import pandas as pd

import util

from fit_predict import fit_predict

np.set_printoptions(linewidth = 190, threshold = 100000,
    formatter={'float':lambda x:'%6s' % (x,) if x!=float(x) else '%8.2f' % (float(x),)})
pd.set_option('display.width', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_columns', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_rows', 2000)
pd.set_option('display.precision', 2)

'''
raw columns:
    year, month, day, time, sym,
    0...20 Predictor Inputs,
    0...n extra inputs
    target: used for training predictor; clamped value of raw_target
    raw_target: estimated gain/loss per trade, assuming a buy.
      For buys +raw_target is good, - is bad.
      For sells, -raw_target is good.

Of the "Predictor Inputs":
columns 0-18 are common inputs to buy and sell predictors
column 19 is input for either the buy or the sell predictor
column 20 (last column) is +/-1 indicating buy or sell
'''
def main(args):
    output_col = 'output'
    if args.in_csv:  # READ CSV
        if os.path.isfile(args.in_file):
            print('%s already exists and would be overwritten' % (args.in_file))
            exit(0)
        with util.timed_execution('Reading %s' % (args.in_csv,)):
            df = pd.read_csv(args.in_csv, delim_whitespace=True, header=None, nrows=args.limit)
            n_cols = len(df.columns)
            n_xtra = n_cols - 29
            pi_names = ' '.join([' p%02d' % (_,) for _ in range(0, 19)])
            xtra_names = ' '.join([' x%02d' % (_,) for _ in range(0, n_xtra)])
            cnames = ('year month day time sym ' + pi_names + ' bs_spcfc bs '
                      + xtra_names + ' target raw ' + output_col).split()
            df.columns = cnames
            df.year = df.year.astype(int)
            df.month = df.month.astype(int)
            df.day = df.day.astype(int)
            df.sym = df.sym.astype(int)
            df.bs = df.bs.astype(int)
        with util.timed_execution('Writing %s' % (args.in_file,)):
            joblib.dump(df, args.in_file)  # WRITE DATAFRAME
    with util.timed_execution('Reading %s' % (args.in_file,)):
        df = joblib.load(args.in_file)  # READ DATAFRAME
    if args.limit:
        df = df[:args.limit]
    input_cols = [_ for _ in df.columns if _ != output_col]

    # from datetime import date
    # dates = df.apply(lambda x: date(x.year, x.month, x.day))
    with util.timed_execution('Creating months'):
        df['months'] = df.apply(lambda x: '%04d/%02d' % (x.year, x.month), axis=1)  # don't forget the axis!!!

    from time_series_loo import TimeSeriesLOO
    with util.timed_execution('Constructing LOO'):
        loo = TimeSeriesLOO(df['months'], args.tr_n, args.ts_n)

    for tr_periods, ts_periods in loo():
        assert len(tr_periods) == args.tr_n
        assert len(ts_periods) == args.ts_n
        assert len(np.union1d(tr_periods, ts_periods)) == args.tr_n + args.ts_n
        print ([_ for _ in tr_periods], end='')
        print ([_ for _ in ts_periods])
        xs_trn = df.loc[df.month in tr_periods, input_cols]
        xs_tst = df.loc[df.month in ts_periods, input_cols]
        print(xs_trn.describe())
        print(xs_tst.describe())

        ys_trn = df.loc[df.month in tr_periods, output_col]
        ys_tst = df.loc[df.month in ts_periods, output_col]
        print(ys_trn.describe())
        print(ys_tst.describe())

        fit_predict(df, xs_trn, ys_trn, xs_tst, ys_tst)

    '''
    from keras.wrappers.scikit_learn import KerasRegressor
    from scipy.stats import randint as sp_randint
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import mean_squared_error, make_scorer

    n_in_cols  = 19
    n_out_cols = 1

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(df)
    x = scaler.transform(df)
    # consider inputs to be first <n_in_cols> contiguous, outputs to be last
    # <n_out_cols> contiguous columns
    y = x[:, -n_out_cols:]
    x = x[:, :n_in_cols] # don't do this before setting y's !!!

    xtrn, xtst, ytrn, ytst = train_test_split(x, y, test_size=args.val_pct)

    nne = NNEstimator(n_in_cols, n_out_cols, [2,], batch_size=args.batch_size,
            nb_epochs=args.nb_epochs, verbose=args.verbose)

    param_dist = {}
    for h in range(args.hl):
        name = 'H%d' % (h,)
        param_dist[name] = sp_randint(args.min_hn, args.max_hn)

    #sk_params = {'n_in_cols':n_in_cols, 'n_out_cols':n_out_cols, 'hiddens':[
    # 2,]}
    #foo = KerasRegressor(build_fn=build_model, **sk_params)
    #cv(foo, xtrn, ytrn, param_dist=param_dist, verbose=3)
    from models.cross_validate import cross_validate as cv
    best = cv(nne, xtrn, ytrn, param_dist=param_dist, verbose=3,
       scorer=make_scorer(mean_squared_error), n_iter=args.cv_tests,
              results_file=args.results_file, folds=args.folds)

    nne = NNEstimator(n_in_cols, n_out_cols, None,
                      batch_size=args.batch_size,
            nb_epochs=args.nb_epochs, verbose=args.verbose)
    nne.set_params(**best.params)
    print()
    loss_and_metrics = nne.model.evaluate(xtrn, ytrn)
    print('TRAIN: ' + str(loss_and_metrics))
    loss_and_metrics = nne.model.evaluate(xtst, ytst)
    print(' TEST: ' + str(loss_and_metrics))
    '''


if __name__ == '__main__':
    import argparse
    import sys
    import pdb, traceback
    def info(type, value, tb):
        traceback.print_exception(type, value, tb)
        pdb.pm()
    sys.excepthook = info

    parser = argparse.ArgumentParser()
    parser.add_argument('in_file')
    parser.add_argument('--in_csv')
    parser.add_argument('--results_file',
                        default='/home/John/Scratch/quanterra/results.txt')
    parser.add_argument('--tr_n', type=int, default=3)
    parser.add_argument('--ts_n', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cv_tests', type=int, default=10)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--verbose', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    main(args)