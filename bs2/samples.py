import numpy as np
import os
import pandas as pd

from joblib import load, dump

import util

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
    n_predictors = 19
    n_xtra = 10
    pi_names = ['p%02d' % (_,) for _ in range(n_predictors)]
    xtra_names = ['x%02d' % (_,) for _ in range(n_xtra)]
    all_column_names = 'year month day time sym'.split() + \
       pi_names + 'bs_spcfc bs'.split() + \
       xtra_names + 'target raw nickpred'.split()

    row_limit = int(args.limit) if args.limit else None

    # CHECK FOR EXISTING OUT_FILE
    if os.path.isfile(args.out_file):
        print('%s already exists and would be overwritten' % (args.out_file))
        exit(0)

    # READ CSV
    with util.timed_execution('Reading %s' % (args.in_csv,)):
        df = pd.read_csv(args.in_csv, delim_whitespace=True, header=None,
                         nrows=row_limit, dtype=np.float64)
        df.columns = all_column_names
        df.year = df.year.astype(int)
        df.month = df.month.astype(int)
        df.day = df.day.astype(int)
        df.sym = df.sym.astype(int)
        df.bs = df.bs.astype(int)

    # CREATE MONTH LABELS FROM OTHER COLUMNS
    with util.timed_execution('Creating months'):
        df['period'] = df.apply(lambda x:
            int('%04d%02d' % (x.year, x.month)), axis=1) # don't forget axis!

    df.index.name='raw_index'

    # WRITE DATAFRAME
    with util.timed_execution('Writing %s' % (args.out_file,)):
        dump(df, args.out_file)


if __name__ == '__main__':
    import argparse
    import sys

    import pdb, traceback
    def info(type, value, tb):
        traceback.print_exception(type, value, tb)
        pdb.pm()
    sys.excepthook = info

    parser = argparse.ArgumentParser()
    parser.add_argument('in_csv')
    parser.add_argument('out_file')
    parser.add_argument('--limit', type=float)

    args = parser.parse_args(sys.argv[1:])
    print(args)
    main(args)
