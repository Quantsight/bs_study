from __future__ import print_function

import ftp
from joblib import load, dump
import numpy as np
import os
import re
import yaml

import util

from algs.rf import RF
from algs.lp import LP
#from algs.fm import FM
from tree_tools import dump_tree
from algs.model import pipeline_from_dicts

np.set_printoptions(linewidth = 250, threshold = 100000,
    formatter={'float':lambda x:'%6s' % (x,) if x!=float(x) else '%8.2f' % (float(x),)})
import pandas as pd
pd.set_option('display.width', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_columns', 28)
pd.set_option('display.max_rows', 70)
pd.set_option('display.precision', 4)

def thres_cum(y_true, y_pred, verbose=1, asc=False):
    data = pd.DataFrame({'y_true': y_true,
                         'y_pred': y_pred})
    data.sort_values('y_pred', ascending=asc, inplace=True)
    data['cumraw'] = data.y_true.cumsum()
    lbl_max = data.cumraw.idxmax()
    i_max = data.index.get_loc(lbl_max)
    pct_taken = 100*float(i_max)/len(data)
    cumraw = data.iloc[i_max].cumraw
    if verbose > 0:
        print('%8d  %8d (%3.0f%%) t:%6.2f  $%8.0f  $%2.2f/trade' % (
            len(y_true), i_max, pct_taken, 
            data.iloc[i_max].y_pred, cumraw, cumraw/len(data)))
    return cumraw


def report_perf(df_ts, preds, bs, verbose=1):
    # passed preds could be a Series or np.array; we will destructively sort.
    profit = np.array(df_ts.raw)
    preds = np.array(preds)
    if bs:
        ix =  np.where(df_ts.bs == bs)[0]
        profit = bs * profit[ix]
        asc = bs < 0
    else:
        bs = 0
        ix = range(len(df_ts))
        asc = False
    
        '''
        # invert sell polarities (both prediction and raw target)
        sell_ix =  np.where(df_ts.bs == -1)[0]
        preds[sell_ix] *= -1
        profit[sell_ix] *= -1
        '''
        preds *= df_ts.bs
        profit *= df_ts.bs
    # pass by .values to strip index
    if verbose > 0:
        print('%+d ' % (bs,), end='')
    return thres_cum(profit, preds[ix], verbose, asc)


from scipy.stats import spearmanr
def report_all_perf(df, ys, verbose=1):
    _ = report_perf(df, ys,  1, verbose=verbose)
    _ = report_perf(df, ys, -1, verbose=verbose)
    cr = report_perf(df, ys, None, verbose=verbose)
    return cr
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

def foo(clf, X, y, trn_n, tst_n, periods, tests, perf_fn, folds=3, verbose=0):
    from sklearn.metrics import make_scorer
    scorer = make_scorer(thres_cum, greater_is_better=True)

    from time_series_loo import NearestOrPreviousLeaveOneOut
    nop_cv = NearestOrPreviousLeaveOneOut(trn_n, tst_n, periods)

    # some parameters aren't optimizable (and therefore some clf's have no optimizable parameters)
    param_dist = {}
    for name, step in clf.named_steps.items():
        dct = step.get_param_dist(X)
        for (k,v) in dct.items():
            new_key = '%s__%s' % (name, k)
            param_dist[new_key] = v

    from cross_validate import cross_validate as CV
    cv = CV(clf, X, y,
            param_dist=param_dist,
            verbose=verbose,
            scorer=scorer,
            n_iter=tests, results_file=perf_fn, folds=folds)

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
        print('Limiting to sym = %s' + args.sym)
        df = df.loc[df.sym==args.sym]

    if args.limit:
        df = df.sample(n=args.limit)

    config = yaml.safe_load(open(args.config_fn))
    addtnl_inputs = config['data_setup'].get('addtnl_inputs', [])
    use_extra = config['data_setup'].get('use_extra', False)
    pi_names = [_ for _ in df.columns if re.match(r'p\d{2}', _)]
    xtra_names = [_ for _ in df.columns if re.match(r'x\d{2}', _)] if \
        use_extra else []
    inputs = pi_names + xtra_names + addtnl_inputs +'bs_spcfc bs'.split()

    # ALL models will be pipelines (even if pipeline holds a single model)
    model_dcts = config['model_pipeline']
    output_col = model_dcts[0]['output_col']
    model_pipeline = pipeline_from_dicts(model_dcts)
    print(model_pipeline)

    if args.test_csv:
        truevals_df = pd.read_csv(args.test_csv)

    trn_n = config['data_setup']['trn_n']
    tst_n = config['data_setup']['tst_n']

    if args.tests > 0:
        foo(model_pipeline, df[inputs], df[output_col], trn_n, tst_n, df.period,
            tests   = args.tests,
            perf_fn = config['cv']['perf_fn'],
            folds   = args.folds,
            verbose = args.verbose)
    else:
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
            with util.timed_execution('Fitting.'):
                print('%24s %12s %12d %12d ' % (
                    [_ for _ in trn_periods], [_ for _ in tst_periods],
                    is_trn.sum(), is_tst.sum()))
                model_pipeline.fit(df.loc[is_trn, inputs],
                         df.loc[is_trn, output_col])
                # print(model_pipeline)
                df.loc[is_trn, 'pred_trn'] = model_pipeline.predict(df.loc[is_trn, inputs])
                df.loc[is_tst, 'pred_tst'] = model_pipeline.predict(df.loc[is_tst, inputs])

            if 'dump_details' in model_dcts[0]:
                fn = 'grp_%s' % (tst_desc,)
                dump_tree(model_pipeline.get_model(), inputs, fn=fn,
                          dir=model_dcts[0]['dump_details'], max_depth=4)

            if args.verbose > 0:
                print('PERFORMANCE')
                report_all_perf(df.loc[is_tst], df.loc[is_tst, 'pred_tst'])

            if args.test_csv:
                for sym in df.sym.unique():
                    for period in tst_periods:
                        mo = period % 201600
                        true = truevals_df[(truevals_df.sym == sym) & (truevals_df.month == mo)].iloc[0]
                        test = df[(df.sym == sym) & (df.month == mo)]
                        assert test.shape[0] == true.tst_n
                        test_mse = ((test.pred_tst - test.target) ** 2).mean()
                        ratio = test_mse / true.test_mse
                        assert ratio > 0.997 and ratio < 1.002
                        print('%02d %02d %8.4f' % (sym, mo, ratio))

            sys.stdout.flush()

        print('PERFORMANCE')
        perf = report_all_perf(df,  df.pred_tst.values)
        if args.test_csv:
            ratio = perf / 13553076
            assert ratio > 0.999 and ratio < 1.001

        dump_preds = model_dcts[0].get('dump_preds')
        if dump_preds:
            df['year month day time sym bs raw pred_trn pred_tst'.split()
                ].to_csv(dump_preds, sep=',', header=True, index=True,
                         compression='gzip')
            if 'ftp' in config:
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
    parser.add_argument('--test_csv')

    parser.add_argument('--limit', type=float)
    parser.add_argument('--sym', type=int)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--no_time_series_fit', type=int, default=0)
    parser.add_argument('--no_RF_sym', action='store_true')
    parser.add_argument('--input_extras', action='store_true')
    parser.add_argument('--input_time', action='store_true')

    parser.add_argument('--tests', type=int, default=0, help='number of CV tests')
    parser.add_argument('--folds', type=int, default=3, help='number of CV folds')


    args = parser.parse_args(sys.argv[1:])
    print(args)
    main(args)
