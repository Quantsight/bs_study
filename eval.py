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
pd.set_option('display.precision', 4)

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
    output_col = 'target'
    n_predictors = 19
    n_xtra = 10
    pi_names = ' '.join([' p%02d' % (_,) for _ in range(n_predictors)])
    xtra_names = ' '.join([' x%02d' % (_,) for _ in range(n_xtra)])
    cnames = ('year month day time sym ' + pi_names + ' bs_spcfc bs '
              + xtra_names + ' target raw nickpred').split()
    input_cols = pi_names.split()
    print(input_cols)

    if args.in_csv:  # READ CSV
        if os.path.isfile(args.in_file):
            print('%s already exists and would be overwritten' % (args.in_file))
            exit(0)
        with util.timed_execution('Reading %s' % (args.in_csv,)):
            df = pd.read_csv(args.in_csv, delim_whitespace=True, header=None, nrows=args.limit)
            df.columns = cnames
            df.year = df.year.astype(int)
            df.month = df.month.astype(int)
            df.day = df.day.astype(int)
            df.sym = df.sym.astype(int)
            df.bs = df.bs.astype(int)
        with util.timed_execution('Creating months'):
            # don't forget the axis!!!
            df['period'] = df.apply(lambda x:
                int('%04d%02d' % (x.year, x.month)), axis=1)
        with util.timed_execution('Writing %s' % (args.in_file,)):
            joblib.dump(df, args.in_file)  # WRITE DATAFRAME

    with util.timed_execution('Reading %s' % (args.in_file,)):
        df = joblib.load(args.in_file)  # READ DATAFRAME
    if args.limit:
        df = df[:args.limit]

    from time_series_loo import TimeSeriesLOO
    with util.timed_execution('Constructing LOO'):
        loo = TimeSeriesLOO(df.period, args.tr_n, args.ts_n)

    for tr_periods, ts_periods in loo():
        print ([_ for _ in tr_periods], end='')
        print ([_ for _ in ts_periods])
        assert len(tr_periods) == args.tr_n
        assert len(ts_periods) == args.ts_n
        assert len(np.union1d(tr_periods, ts_periods)) == args.tr_n + args.ts_n

        xs_trn = df.loc[df.period.isin(tr_periods), input_cols]
        xs_tst = df.loc[df.period.isin(ts_periods), input_cols]

        ys_trn = df.loc[df.period.isin(tr_periods), output_col]
        ys_tst = df.loc[df.period.isin(ts_periods), output_col]
        print(pd.DataFrame({'trn':ys_trn, 'tst':ys_tst}).describe())

        fit_predict(df, xs_trn, ys_trn, xs_tst, ys_tst)

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