from __future__ import print_function

import logging

from operator import itemgetter
from time import time

import numpy as np
np.set_printoptions(linewidth = 190, threshold = 100000,
    formatter={'float':lambda x:'%6s' % (x,) if x!=float(x) else '%8.5f' % (float(x),)})

import pandas as pd
pd.set_option('display.width', pd.util.terminal.get_terminal_size()[0])
pd.set_option('display.max_rows', 500)
pd.set_option('precision', 5)

from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cross_validation import KFold, ShuffleSplit

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

N_TOP = 10


def scorer_default(estimator, X, y):
    return estimator.score(X, y) # higher is better


def cross_validate(clf, X, y=None, param_dist=None, n_iter=10,
                   folds=3, fit_params=None,
                   scorer=scorer_default, verbose=3, cvs=None, error_score=0,
                   refit=True, results_file='crossval.txt'):
    if cvs is None:
        # shuffle to ensure ordering doesn't matter
        cvs = KFold(X.shape[0], n_folds=folds, shuffle=True, random_state=None)
        # random shuffle each time, same samples could be returned:
        #cvs = ShuffleSplit(X.shape[0], n_iter=folds, test_size=0.1)

    # Utility function to report best scores
    def report(random_search, n_top=N_TOP):
        df = pd.DataFrame(random_search.cv_results_)
	# use worst case 1/3 of distribution to judge cross validation results 
        df['dist'] = df.mean_test_score + df.std_test_score
        df.sort_values('mean_test_score', ascending=True, inplace=True)
        print(df[['rank_test_score', 'mean_test_score',
                  'std_test_score', 'dist', 'params']])
        df[['rank_test_score', 'mean_test_score',
                  'std_test_score', 'dist', 'params']].to_csv(results_file)
        return df.iloc[0]

    # run randomized search; use n_jobs=1 because each model-fit will use all CPUs
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
            n_iter=n_iter, n_jobs=1, scoring=scorer, cv=cvs,
            verbose=verbose, fit_params=fit_params, error_score=error_score,
            refit=refit)
    '''
    from evolutionary_search import EvolutionaryAlgorithmSearchCV
    random_search = EvolutionaryAlgorithmSearchCV(estimator=clf,
                                   params=param_dist,
                                   scoring=scorer,
                                   cv=cvs,
                                   verbose=verbose,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
				   n_jobs=1)
    '''

    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter))
    best = report(random_search)
    return best # random_search.best_estimator_


def scorer_rf(estimator, X, y):
    #return estimator.oob_improvement_[-1]
    return estimator.oob_score_


def cross_validate_rf(clf, X, y, n_iter=1.0, fit_params=None):
    # specify parameters and distributions to sample from
    # range (low, high) is (loc, loc+scale)
    total_features = X.shape[1]
    min_features = int(max(1, 0.10*total_features))
    max_features = int(max(1, 0.70*total_features))
    param_dist = {
        "max_features": sp_randint(min_features, max_features),
        "min_samples_leaf": sp_randint(1, 1000),
        #"learning_rate": sp_uniform(loc=0.01, scale=0.19),
        #"subsample": sp_uniform(loc=0.01, scale=0.89)
        }
    return cross_validate(clf, X, y, param_dist, n_iter=n_iter, fit_params=fit_params,
                          scorer=scorer_rf)
