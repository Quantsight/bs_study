from __future__ import print_function

from joblib import load, dump
import numpy as np
import os
import pandas as pd

import util

from fit_predict import fit_predict
from tree_tools import dump_tree

np.set_printoptions(linewidth = 250, threshold = 100000,
    formatter={'float':lambda x:'%6s' % (x,) if x!=float(x) else '%8.2f' % (float(x),)})
pd.set_option('display.width', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_columns', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_rows', 70)
pd.set_option('display.precision', 4)

def report_perf(df_ts, preds, bs, verbose=0):
    if bs:
        ix =  np.where(df_ts.bs == bs)[0]
        profit = bs * df_ts.iloc[ix]['raw'].values
        asc = bs < 0
    else:
        bs = 0
        ix = range(len(df_ts))
        profit = df_ts.raw.values
        asc = False
        # invert sell polarities (both prediction and raw target)
        sell_ix =  np.where(df_ts.bs == -1)[0]
        preds[sell_ix] *= -1
        profit[sell_ix] *= -1
    # pass by .values to strip index
    data = pd.DataFrame({'target': df_ts.iloc[ix]['target'].values,
                         'preds': preds[ix],
                         'nickpred': df_ts.iloc[ix]['nickpred'].values,
                         'raw': profit})
    srtd = data.sort_values('preds', ascending=asc)
    srtd['cumraw'] = srtd.raw.cumsum()
    lbl_max = srtd.cumraw.idxmax()
    i_max = srtd.index.get_loc(lbl_max)
    pct_taken = 100*float(i_max)/len(srtd)
    cumraw = srtd.iloc[i_max].cumraw
    if verbose > 0:
        print('%+d %10.4f  %3.0f%%  %8.0f  %8.4f' % (
            bs, srtd.iloc[i_max].preds, pct_taken, cumraw, cumraw/len(srtd)))
    return cumraw

'''
def assign_preds(sym, df_tr, df_ts):
    preds = fit_predict(df_tr.loc[df_tr.sym == sym, input_cols],
                        df_tr.loc[df_tr.sym == sym, output_col],
                        df_ts.loc[df_ts.sym == sym, input_cols],
                        df_ts.loc[df_ts.sym == sym, output_col])
    df_ts.loc[df_ts.sym == sym, 'new_pred'] = preds
'''
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
    output_col = args.output_col
    n_predictors = 19
    n_xtra = 10
    pi_names = ' '.join([' p%02d' % (_,) for _ in range(n_predictors)])
    xtra_names = ' '.join([' x%02d' % (_,) for _ in range(n_xtra)])
    cnames = ('year month day time sym ' + pi_names + ' bs_spcfc bs '
              + xtra_names + ' target raw nickpred').split()
    input_cols = (pi_names + ' sym bs_spcfc bs').split()
    grp_fit_cols = list(input_cols)

    if args.in_csv:  # READ CSV
        if os.path.isfile(args.in_file):
            print('%s already exists and would be overwritten' % (args.in_file))
            exit(0)
        with util.timed_execution('Reading %s' % (args.in_csv,)):
            df = pd.read_csv(args.in_csv, delim_whitespace=True, header=None,
                             nrows=args.limit, dtype=np.float64)
            df.columns = cnames
            df.year = df.year.astype(int)
            df.month = df.month.astype(int)
            df.day = df.day.astype(int)
            df.sym = df.sym.astype(int)
            df.bs = df.bs.astype(int)
            df['new_pred'] = 0  # placeholder for predictions
        with util.timed_execution('Creating months'):
            # don't forget the axis!!!
            df['period'] = df.apply(lambda x:
                int('%04d%02d' % (x.year, x.month)), axis=1)
        with util.timed_execution('Writing %s' % (args.in_file,)):
            dump(df, args.in_file)  # WRITE DATAFRAME

    with util.timed_execution('Reading %s' % (args.in_file,)):
        df = load(args.in_file)  # READ DATAFRAME
    if args.limit:
        df = df.sample(n=args.limit)

    if args.group_fit:
        ''' GLOBAL FIT; no time-series '''
        ts_preds = fit_predict(df[input_cols], df[output_col],
                            df[input_cols], df[output_col])
        bcr = report_perf(df, ts_preds,  1, verbose=1)
        scr = report_perf(df, ts_preds, -1, verbose=1)
        cr  = report_perf(df, ts_preds, None, verbose=1)

    from time_series_loo import TimeSeriesLOO
    with util.timed_execution('Constructing LOO'):
        loo = TimeSeriesLOO(df.period, args.tr_n, args.ts_n)

    if args.group_fit_input:
        pred_col = 'grp'
        input_cols.append('grp')
    else:
        pred_col = 'new_pred'

    tbcr = tscr = 0
    for tr_periods, ts_periods in loo():
        assert len(tr_periods) == args.tr_n
        assert len(ts_periods) == args.ts_n
        assert len(np.union1d(tr_periods, ts_periods)) == args.tr_n + args.ts_n
        ts_desc = ' '.join(str(_) for _ in ts_periods)

        is_tr = df.period.isin(tr_periods)
        is_ts = df.period.isin(ts_periods)

        if args.group_fit or args.group_fit_input:
            with util.timed_execution('Fitting all symbols'):
                ts_preds, tr_preds, clf = fit_predict(
                    df.loc[is_tr, grp_fit_cols],
                    df.loc[is_tr, output_col],
                    df.loc[is_ts, grp_fit_cols],
                    df.loc[is_ts, output_col])
                df.loc[is_tr, pred_col] = tr_preds
                df.loc[is_ts, pred_col] = ts_preds
            if args.verbose > 0:
                print('%s ' % (ts_desc,), end='')
            bcr = report_perf(df.loc[is_ts], ts_preds, 1, verbose=args.verbose)
            if args.verbose > 0:
                print('%s ' % (ts_desc,), end='')
            if args.dump_group_fit:
                fn = 'grp_%s' % (ts_desc,)
                dump_tree(clf, grp_fit_cols, fn=fn, dir=args.dump_group_fit,
                          max_depth=4)
            scr = report_perf(df.loc[is_ts], ts_preds, -1, verbose=args.verbose)
            tbcr += bcr
            tscr += scr

        if not args.group_fit:
            for sym in sorted(df.sym.unique()):
                is_sym = df.sym==sym
                print('%24s %12s %12d %12d %4d ' % (
                    [_ for _ in tr_periods], [_ for _ in ts_periods],
                    (is_tr & is_sym).sum(), (is_ts & is_sym).sum(), sym), end='')
                ts_preds, tr_preds, clf = fit_predict(
                    df.loc[is_tr & is_sym, input_cols],
                    df.loc[is_tr & is_sym, output_col],
                    df.loc[is_ts & is_sym, input_cols],
                    df.loc[is_ts & is_sym, output_col])
                df.loc[is_ts & is_sym, pred_col] = ts_preds

                fn = 'sym_%s_%s' % (sym, ts_desc)
                dump_tree(clf, input_cols, fn=fn, dir=args.dump_group_fit,
                          max_depth=4)

                if args.verbose == 2:
                    print('%s %2d ' % (ts_desc, sym), end='')
                    bcr = report_perf(df.loc[is_ts & is_sym], ts_preds,  1)
                    print('%s %2d ' % (ts_desc, sym), end='')
                    scr = report_perf(df.loc[is_ts & is_sym], ts_preds, -1)
                    tbcr += bcr
                    tscr += scr
    bcr = report_perf(df, df[pred_col].values, 1, verbose=1)
    scr = report_perf(df, df[pred_col].values, -1, verbose=1)
    cr = report_perf(df, df[pred_col].values, None, verbose=1)

    # print('   Buy Profit: %10.2f' % (tbcr,))
    # print('  Sell Profit: %10.2f' % (tscr,))
    print('Global Profit: %10.2f' % (cr,))

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
    parser.add_argument('--dump_group_fit')
    parser.add_argument('--results_file',
                        default='/home/John/Scratch/quanterra/results.txt')
    parser.add_argument('--tr_n', type=int, default=3)
    parser.add_argument('--ts_n', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--cv_tests', type=int, default=10)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--group_fit', type=int, default=0)
    parser.add_argument('--group_fit_input', action='store_true')
    parser.add_argument('--output_col', default='target')

    args = parser.parse_args(sys.argv[1:])
    main(args)