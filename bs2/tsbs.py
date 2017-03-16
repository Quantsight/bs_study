from __future__ import print_function

import ftp
from joblib import load, dump
import numpy as np
import os
import pandas as pd
import re
import yaml

import util

from algs.rf import RF
from algs.lp import LP
#from algs.fm import FM
from tree_tools import dump_tree
from model_from_dict import pipeline_from_dicts

np.set_printoptions(linewidth = 250, threshold = 100000,
    formatter={'float':lambda x:'%6s' % (x,) if x!=float(x) else '%8.2f' % (float(x),)})
pd.set_option('display.width', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_columns', 28)
pd.set_option('display.max_rows', 70)
pd.set_option('display.precision', 4)

def report_perf(df_ts, preds, bs, verbose=1):
    # passed preds could be a Series or np.array; we will destructively sort.
    profit = df_ts.raw.values.copy()
    preds = np.array(preds)
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
    # print('Global Profit: %10.2f' % (cr,))

    '''
    if np.isnan(df.grp_pred_tst).sum() == 0:
        src = spearmanr(df.grp_pred_tst, ys)[0]
        print('ys / GRP_PRED_TEST SRC: %8.5f' % (src,))

    # there can be NaN's in symbol fit if only group fit had been called.
    if np.isnan(df.sym_pred_tst).sum() == 0:
        src = spearmanr(df.sym_pred_tst, ys)[0]
        print('ys / SYM_PRED      SRC: %8.5f' % (src,))
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
    np.random.seed(0)

    with util.timed_execution('Reading %s' % (args.in_file,)):
        df = load(args.in_file)  # READ DATAFRAME

    if args.sym: # limit to only symbol
        df = df.loc[df.sym==args.sym]

    if args.limit:
        df = df.sample(n=args.limit)

    pi_names =   [_ for _ in df.columns if re.match(r'p\d{2}', _)]
    xtra_names = [_ for _ in df.columns if re.match(r'x\d{2}', _)]
    inputs = pi_names + xtra_names + 'time bs_spcfc bs'.split()

    config = yaml.safe_load(open(args.config_fn))
    model_dcts = config['model_pipeline']
    output_col = model_dcts[0]['output_col']
    pipe = pipeline_from_dicts(model_dcts)
    print(pipe)

    ''' TODO
    test group fit alone
    add various inputs to group fit (sym, time, etc)

    test symbol fit alone
    add various inputs to symbol fit (sym, time, etc) 

    test grp-then-symbol
    '''

    trn_n = config['data_setup']['trn_n']
    tst_n = config['data_setup']['tst_n']
    from time_series_loo import TimeSeriesLOO
    with util.timed_execution('Constructing LOO'):
        loo = TimeSeriesLOO(df.period, trn_n, tst_n)

    df['pred_trn'] = np.nan
    df['pred_tst'] = np.nan
    for trn_periods, tst_periods in loo():
        assert len(trn_periods) == trn_n
        assert len(tst_periods) == tst_n
        assert len(np.union1d(trn_periods, tst_periods)) == trn_n + tst_n
        tst_desc = ' '.join(str(_) for _ in tst_periods)

        is_trn = df.period.isin(trn_periods)
        is_tst = df.period.isin(tst_periods)

        np.random.seed(0)
        with util.timed_execution('Fitting all symbols'):
            print('%24s %12s %12d %12d ' % (
                [_ for _ in trn_periods], [_ for _ in tst_periods],
                is_trn.sum(), is_tst.sum()))
            pipe.fit(df.loc[is_trn, inputs],
                     df.loc[is_trn, output_col])
            #pipe.print()
            df.loc[is_trn, 'pred_trn'] = pipe.predict(df.loc[is_trn, inputs])
            df.loc[is_tst, 'pred_tst'] = pipe.predict(df.loc[is_tst, inputs])

        dump_details = model_dcts[0].get('dump_details')
        if dump_details:
            fn = 'grp_%s' % (tst_desc,)
            dump_tree(pipe.get_model(), inputs, fn=fn,
                      dir=dump_details, max_depth=4)

        if args.verbose > 0:
            print('GROUP FIT PERFORMANCE')
            report_all_perf(df.loc[is_tst], df.loc[is_tst, 'pred_tst'])
        sys.stdout.flush()

    print('GROUP FIT PERFORMANCE')
    report_all_perf(df,  df.pred_tst.values)

    dump_preds = model_dcts[0].get('dump_preds')
    if dump_preds:
        df['year month day time sym bs raw pred_trn pred_tst'.split()
            ].to_csv(dump_preds, sep=',', header=True, index=True,
                     compression='gzip')
        ftp.put(dump_preds, out_path='', out_fn='predictions.csv.gz',
                config=config['ftp'])


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
    parser.add_argument('--results_file',
                        default='/home/John/Scratch/quantera/results.txt')
    parser.add_argument('--config_fn', default='config.yaml')

    parser.add_argument('--limit', type=float)
    parser.add_argument('--sym', type=int)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--no_time_series_fit', type=int, default=0)
    parser.add_argument('--no_RF_sym', action='store_true')
    parser.add_argument('--input_extras', action='store_true')
    parser.add_argument('--input_time', action='store_true')

    args = parser.parse_args(sys.argv[1:])
    print(args)
    main(args)
