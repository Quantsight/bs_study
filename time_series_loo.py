#!/usr/bin/env python

import pandas as pd
import numpy as np


class TimeSeriesLOO:
    """
    Starting with first date, sequentially pull <periods_n> test periods out at a time, paired 
    with the nearest <tr_n> train months that are either earlier or later the tesperiods months,
    with higher priorities given to earlier train months.

    Returns (train_beg, test_beg, test_end) tuples.
    """
    def __init__(self, periods, tr_n, ts_n):
        """
        Generates all possible (paired) training & testing windows.
        The resolution of the passed-in <periods> determines the resolution of the
        output.
        :param periods: pandas index containing candidate periods
        :param tr_n: number of periods in training windows
        :param periods: number of periods in testing windows
        """
        self.periods = sorted(periods.unique())
        self.tr_n = tr_n
        self.ts_n = ts_n
        if len(self.periods) < tr_n + ts_n:
            err = ('Not enough periods (%d) to fulfill tr_n:%d and ts_n:%d' %
                  (len(self.periods), tr_n, ts_n))
            err += '\nPeriods found: %s' % (self.periods,)
            raise Exception(err)

    def __len__(self): return len(self.periods)
    
    def __call__(self):
        """
        generate in-order train/test periods
        """
        ts_beg_i = 0
        for ii in range(len(self.periods)):
            periods = self.periods[ii : ii + self.tr_n + self.ts_n]

            # slide testing window forward until it would drop off end;
            # do NOT reset <ts_beg_i>; once it's at the rightmost part of
            # <periods>, it stays at the rightmost of all subsequent <periods>
            for jj in range(ts_beg_i, len(periods) - self.ts_n + 1):
                ts_periods = periods[jj:jj+self.ts_n]
                tr_periods = np.setdiff1d(periods, ts_periods)
                yield tr_periods, ts_periods
            ts_beg_i = jj


from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import BaseCrossValidator
class NearestOrPreviousLeaveOneOut(BaseCrossValidator):
    def __init__(self, trn_n, tst_n, periods):
        self.trn_n = trn_n
        self.tst_n = tst_n
        self.loo = TimeSeriesLOO(periods, self.trn_n, self.tst_n)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for trn_periods, tst_periods in self.loo():
            train_mask  = groups.isin(trn_periods)
            test_mask   = groups.isin(tst_periods)
            train_idxs = indices[train_mask]
            test_idxs  = indices[test_mask]
            yield train_idxs, test_idxs

    def get_n_splits(self, X, y=None, groups=None):
        return len(groups.unique())


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('tr_n', type=int)
    parser.add_argument('ts_n', type=int)
    args = parser.parse_args(sys.argv[1:])

    periods = ["2011030600","2011031317","2011032013","2011032701","2011040116","2011040720","2011041322","2011041923","2011042608","2011050212","2011050822","2011051611","2011052311","2011053017","2011060610","2011061411","2011062119","2011062911","2011070721","2011071420","2011072017","2011072713","2011080213","2011080816","2011081421","2011081910","2011082417","2011083016","2011090521","2011091014","2011091508","2011092016","2011092523","2011093002","2011100508","2011100917","2011101215","2011101615","2011101917","2011102320","2011102619","2011103018","2011110218","2011110714","2011111012","2011111405","2011111611","2011111821","2011112207","2011112611","2011112911","2011120116","2011120509","2011120710","2011121008","2011121313","2011121611","2011122008","2011122318","2011122720","2011123007","2012010214","2012010417","2012010622","2012010917","2012011112","2012011313","2012011615","2012011821","2012012119","2012012421","2012012717","2012013021","2012020210","2012020510","2012020802","2012021017","2012021407","2012021620","2012022014","2012022220","2012022516","2012022813","2012030204","2012030519","2012030815","2012031215","2012031516","2012031916","2012032214","2012032613","2012032912","2012040214","2012040519","2012041013","2012041410","2012041814","2012042314","2012042715","2012050211"]

    periods = pd.to_datetime(periods, format='%Y%m%d%H')
    tr_n = args.tr_n
    ts_n = args.ts_n

    nop_cv = NearestOrPreviousLeaveOneOut(tr_n, ts_n, periods)
    n_splits = 0
    for train_idxs, test_idxs in nop_cv.split(periods, y=None, groups=periods):
        print [pd.to_datetime(_).strftime('%Y.%m.%d') for _ in periods[train_idxs]],
        print [pd.to_datetime(_).strftime('%Y.%m.%d') for _ in periods[test_idxs]]
        n_splits += 1

    assert nop_cv.get_n_splits(X=None, y=None, groups=periods) == n_splits

    cv = TimeSeriesLOO(periods, tr_n=tr_n, ts_n=ts_n)
    for tr_periods, ts_periods in cv():
        assert len(tr_periods) == tr_n
        assert len(ts_periods) == ts_n
        assert len(np.union1d(tr_periods, ts_periods)) == tr_n + ts_n
        print [pd.to_datetime(_).strftime('%Y.%m.%d') for _ in tr_periods],
        print [pd.to_datetime(_).strftime('%Y.%m.%d') for _ in ts_periods]
