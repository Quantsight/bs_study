from __future__ import print_function

from joblib import load, dump
import numpy as np
import os
import pandas as pd

import util

from fit_predict import fit_predict_rf, fit_predict_lin
from tree_tools import dump_tree
import ftp

np.set_printoptions(linewidth = 250, threshold = 100000,
    formatter={'float':lambda x:'%6s' % (x,) if x!=float(x) else '%8.2f' % (float(x),)})
pd.set_option('display.width', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_columns', 28)
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

from scipy.stats import spearmanr
def report_all_perf(df, ys, verbose=1):
    _  = report_perf(df, ys,  1, verbose=verbose)
    _  = report_perf(df, ys, -1, verbose=verbose)
    cr = report_perf(df, ys, None, verbose=verbose)
    print('Global Profit: %10.2f' % (cr,))

    if np.isnan(df.grp_pred_tst).sum() == 0:
        src = spearmanr(df.grp_pred_tst, ys)[0]
        print('ys / GRP_PRED_TEST SRC: %8.5f' % (src,))

    # there can be NaN's in symbol fit if only group fit had been called.
    if np.isnan(df.sym_pred_test).sum() == 0:
        src = spearmanr(df.sym_pred_test, ys)[0]
        print('ys / SYM_PRED      SRC: %8.5f' % (src,))

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

    row_limit = int(args.limit) if args.limit else None

    if args.in_csv:  # READ CSV
        if os.path.isfile(args.in_file):
            print('%s already exists and would be overwritten' % (args.in_file))
            exit(0)
        with util.timed_execution('Reading %s' % (args.in_csv,)):
            df = pd.read_csv(args.in_csv, delim_whitespace=True, header=None,
                             nrows=row_limit, dtype=np.float64)
            df.columns = cnames
            df.year = df.year.astype(int)
            df.month = df.month.astype(int)
            df.day = df.day.astype(int)
            df.sym = df.sym.astype(int)
            df.bs = df.bs.astype(int)
            df['grp_pred_trn'] = np.nan # placeholder for predictions (in-sample)
            df['grp_pred_tst'] = np.nan # placeholder for predictions (out-of-sample)
            df['sym_pred_test'] = np.nan  # placeholder for predictions
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
    if row_limit:
        df = df.sample(n=row_limit)

    if args.noncv_fit:
        ''' GLOBAL FIT; no time-series '''
        tst_preds, trn_preds, clf = fit_predict_lin(df[sym_inputs], df[output_col],
                            df[sym_inputs], df[output_col])
        _ = report_perf(df, tst_preds,  1)
        _ = report_perf(df, tst_preds, -1)
        _ = report_perf(df, tst_preds, None)

    from time_series_loo import TimeSeriesLOO
    with util.timed_execution('Constructing LOO'):
        loo = TimeSeriesLOO(df.period, args.trn_n, args.tst_n)

    model_lookup = {'RF':fit_predict_rf, 'LP':fit_predict_lin}
    if args.grp_fit:
        sym_inputs_trn = sym_inputs + ['grp_pred_trn']
        sym_inputs_tst = sym_inputs + ['grp_pred_tst']
        if args.grp_fit == 'RF' and not args.no_RF_sym:
            grp_inputs.append('sym')
        grp_model = model_lookup[args.grp_fit]
    else:
        sym_inputs_trn = list(sym_inputs)
        sym_inputs_tst = list(sym_inputs)
        grp_model = None

    if args.sym_fit:
        sym_model = model_lookup[args.sym_fit]
    else:
        sym_model = None

    df['grp_pred_trn'] = np.nan
    df['grp_pred_tst'] = np.nan
    df['sym_pred_test'] = np.nan
    for trn_periods, tst_periods in loo():
        assert len(trn_periods) == args.trn_n
        assert len(tst_periods) == args.tst_n
        assert len(np.union1d(trn_periods, tst_periods)) == args.trn_n + args.tst_n
        tst_desc = ' '.join(str(_) for _ in tst_periods)

        is_trn = df.period.isin(trn_periods)
        is_tst = df.period.isin(tst_periods)

        if grp_model:
            with util.timed_execution('Fitting all symbols'):
                print('%24s %12s %12d %12d ' % (
                    [_ for _ in trn_periods], [_ for _ in tst_periods],
                    is_trn.sum(), is_tst.sum()), end='')
                tst_preds, trn_preds, clf = grp_model(df.loc[is_trn, grp_inputs],
                                                    df.loc[is_trn, output_col],
                                                    df.loc[is_tst, grp_inputs],
                                                    df.loc[is_tst, output_col])
                df.loc[is_trn, 'grp_pred_trn'] = trn_preds
                df.loc[is_tst, 'grp_pred_tst'] = tst_preds
            if args.verbose > 0:
                print('GROUP FIT PERFORMANCE')
                report_all_perf(df.loc[is_tst], tst_preds)
            if args.dump_grp:
                fn = 'grp_%s' % (tst_desc,)
                dump_tree(clf, grp_inputs, fn=fn, dir=args.dump_grp, max_depth=4)
            sys.stdout.flush()

        if sym_model:
            for sym in sorted(df.sym.unique()):
                is_sym = df.sym==sym
                print('%24s %12s %12d %12d %4d ' % (
                    [_ for _ in trn_periods], [_ for _ in tst_periods],
                    (is_trn & is_sym).sum(), (is_tst & is_sym).sum(), sym), end='')
                tst_preds, trn_preds, clf = sym_model(
                    df.loc[is_trn & is_sym, sym_inputs_trn],
                    df.loc[is_trn & is_sym, output_col],
                    df.loc[is_tst & is_sym, sym_inputs_tst],
                    df.loc[is_tst & is_sym, output_col])
                df.loc[is_tst & is_sym, 'sym_pred_test'] = tst_preds

                if args.dump_sym:
                    fn = 'sym_%s_%s' % (sym, tst_desc)
                    dump_tree(clf, sym_inputs_trn, fn=fn, dir=args.dump_sym,
                              max_depth=4)

                if args.verbose == 2:
                    print('SYMBOL FIT PERFORMANCE')
                    report_all_perf(df.loc[is_tst & is_sym], tst_preds)
                sys.stdout.flush()

    if grp_model:
        print('GROUP FIT PERFORMANCE')
        report_all_perf(df, df.grp_pred_tst.values)

    if sym_model:
        print('SYMBOL FIT PERFORMANCE')
        report_all_perf(df, df.sym_pred_test.values)

    if args.dump_preds:
        df['year month day time sym bs raw grp_pred_trn grp_pred_tst sym_pred_test'.split()
            ].to_csv(args.dump_preds, sep=',', header=True, index=True)
        path = ''
        ftp.put(path, out_fn= args.dump_preds, config_fn = args.ftp_config)


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
    parser.add_argument('--resultst_file',
                        default='/home/John/Scratch/quantera/results.txt')
    parser.add_argument('--ftp_config', default='config.ini')
    parser.add_argument('--trn_n', type=int, default=3)
    parser.add_argument('--tst_n', type=int, default=1)
    parser.add_argument('--limit', type=float)
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
