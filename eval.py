from __future__ import print_function

from joblib import load, dump
import numpy as np
import os
import pandas as pd

import util

from fit_predict import fit_predict_rf, fit_predict_lin
from tree_tools import dump_tree

np.set_printoptions(linewidth = 250, threshold = 100000,
    formatter={'float':lambda x:'%6s' % (x,) if x!=float(x) else '%8.2f' % (float(x),)})
pd.set_option('display.width', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_rows', 70)
pd.set_option('display.precision', 4)

def report_perf(df_ts, preds, bs, verbose=1):
    profit = df_ts.raw.values.copy()
    preds = preds.copy()
    if bs:
        ix =  np.where(df_ts.bs == bs)[0]
        profit = bs * profit[ix]
        asc = bs < 0
    else:
        bs = 0
        ix = range(len(df_ts))
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


def report_all_perf(df, ys, verbose=1):
    _  = report_perf(df, ys,  1, verbose=verbose)
    _  = report_perf(df, ys, -1, verbose=verbose)
    cr = report_perf(df, ys, None, verbose=verbose)
    print('Global Profit: %10.2f' % (cr,))
    print(np.corrcoef(df.grp_pred, df.sym_pred))

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
    sym_inputs = (pi_names + ' bs_spcfc bs').split()
    grp_inputs = list(sym_inputs) # create a copy

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
            df['grp_pred'] = 0  # placeholder for predictions
            df['sym_pred'] = 0  # placeholder for predictions
        with util.timed_execution('Creating months'):
            # don't forget the axis!!!
            df['period'] = df.apply(lambda x:
                int('%04d%02d' % (x.year, x.month)), axis=1)
        with util.timed_execution('Writing %s' % (args.in_file,)):
            dump(df, args.in_file)  # WRITE DATAFRAME

    with util.timed_execution('Reading %s' % (args.in_file,)):
        df = load(args.in_file)  # READ DATAFRAME
    if args.sym: # limit to only symbol
        df = df.loc[df.sym==args.sym]
    if args.limit:
        df = df.sample(n=args.limit)

    if args.noncv_fit:
        ''' GLOBAL FIT; no time-series '''
        ts_preds, tr_preds, clf = fit_predict_lin(df[sym_inputs], df[output_col],
                            df[sym_inputs], df[output_col])
        _ = report_perf(df, ts_preds,  1)
        _ = report_perf(df, ts_preds, -1)
        _ = report_perf(df, ts_preds, None)

    from time_series_loo import TimeSeriesLOO
    with util.timed_execution('Constructing LOO'):
        loo = TimeSeriesLOO(df.period, args.tr_n, args.ts_n)

    model_lookup = {'RF':fit_predict_rf, 'LP':fit_predict_lin}
    if args.grp_fit:
        sym_inputs.append('grp_pred')
        if args.grp_fit == 'RF' and not args.no_RF_sym:
            grp_inputs.append('sym')
        grp_model = model_lookup[args.grp_fit]
    else:
        grp_model = None

    if args.sym_fit:
        sym_model = model_lookup[args.sym_fit]
    else:
        sym_model = None

    for tr_periods, ts_periods in loo():
        assert len(tr_periods) == args.tr_n
        assert len(ts_periods) == args.ts_n
        assert len(np.union1d(tr_periods, ts_periods)) == args.tr_n + args.ts_n
        ts_desc = ' '.join(str(_) for _ in ts_periods)

        is_tr = df.period.isin(tr_periods)
        is_ts = df.period.isin(ts_periods)

        if grp_model:
            with util.timed_execution('Fitting all symbols'):
                print('%24s %12s %12d %12d ' % (
                    [_ for _ in tr_periods], [_ for _ in ts_periods],
                    is_tr.sum(), is_ts.sum()), end='')
                ts_preds, tr_preds, clf = grp_model(df.loc[is_tr, grp_inputs],
                                                    df.loc[is_tr, output_col],
                                                    df.loc[is_ts, grp_inputs],
                                                    df.loc[is_ts, output_col])
                df.loc[is_tr, 'grp_pred'] = tr_preds
                df.loc[is_ts, 'grp_pred'] = ts_preds
            if args.verbose > 0:
                report_all_perf(df.loc[is_ts], ts_preds)
            if args.dump_grp:
                fn = 'grp_%s' % (ts_desc,)
                dump_tree(clf, grp_inputs, fn=fn, dir=args.dump_grp, max_depth=4)
            sys.stdout.flush()

        if sym_model:
            for sym in sorted(df.sym.unique()):
                is_sym = df.sym==sym
                print('%24s %12s %12d %12d %4d ' % (
                    [_ for _ in tr_periods], [_ for _ in ts_periods],
                    (is_tr & is_sym).sum(), (is_ts & is_sym).sum(), sym), end='')
                ts_preds, tr_preds, clf = sym_model(
                    df.loc[is_tr & is_sym, sym_inputs],
                    df.loc[is_tr & is_sym, output_col],
                    df.loc[is_ts & is_sym, sym_inputs],
                    df.loc[is_ts & is_sym, output_col])
                df.loc[is_ts & is_sym, 'sym_pred'] = ts_preds

                if args.dump_sym:
                    fn = 'sym_%s_%s' % (sym, ts_desc)
                    dump_tree(clf, sym_inputs, fn=fn, dir=args.dump_sym, max_depth=4)

                if args.verbose == 2:
                    report_all_perf(df.loc[is_ts & is_sym], ts_preds)
            sys.stdout.flush()

    if grp_model:
        print('GROUP FIT PERFORMANCE')
        report_all_perf(df, df.grp_pred.values)

    if sym_model:
        print('SYMBOL FIT PERFORMANCE')
        report_all_perf(df, df.sym_pred.values)

    if args.dump_preds:
        df['year month day time sym bs raw grp_pred sym_pred'.split()
            ].to_csv(args.dump_preds, sep=',', header=True, index=True)


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
                        default='/home/John/Scratch/quantera/results.txt')
    parser.add_argument('--tr_n', type=int, default=3)
    parser.add_argument('--ts_n', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sym', type=int)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--noncv_fit', type=int, default=0)
    parser.add_argument('--grp_fit')
    parser.add_argument('--sym_fit')
    parser.add_argument('--dump_grp')
    parser.add_argument('--dump_sym')
    parser.add_argument('--dump_preds')
    parser.add_argument('--no_RF_sym', action='store_true')
    parser.add_argument('--output_col', default='target')

    args = parser.parse_args(sys.argv[1:])
    print(args)
    main(args)