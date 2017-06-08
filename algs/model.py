from __future__ import print_function

import numpy as np

from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint

class ModelBase(BaseEstimator):
    def __init__(self, name, sub_params={}, inputs=None,  
        cv_params={}, scorer=None):
        ''' 
        name: string with name of the instance
        clf: sklearn Estimator class to use
        inputs: string storing comma-separated list of names
        sub_params: COULD BE EMPTY if using defaults (e.g. LP)
        cv_params:

        '''
        self.name = name
        self.inputs = inputs
        self._sub_params = sub_params
        self._clf_score = scorer
        self._cv_params = cv_params
        # input_pattern

    def get_param_dist(self, xs):
        param_dist = {}
        for k,v in self._cv_params.items():
            if isinstance(v, (list, tuple)) and not isinstance(v, basestring):
                vtype = list
            else:
                vtype = eval(v['type'])
            assert vtype in [int, float, list]
            if vtype == int:
                param_dist[k] = sp_randint(v['min'], v['max'])
            elif vtype == float:
                param_dist[k] = sp_uniform(v['min'], v['max'])
            elif vtype == list:
                param_dist[k] = v
        assert param_dist, 'No cv_params passed by cv attempted.'
        return param_dist

    '''
    def get_params(self, deep=True):
        """Get parameters for this estimator.
            Parameters
            ----------
            deep : boolean, optional
                If True, will return the parameters for this estimator and
                contained subobjects that are estimators.
            Returns
            -------
            params : mapping of string to any
                Parameter names mapped to their values.
        """
        return {'name': self.name, 'inputs':self.inputs, 
                'sub_params':sub_params, 'cv_params': cv_params,
                'scorer':scorer}

    def set_params(self, **params):
        """
        TODO: PASS THROUGH parameters to every estimator
            - what if fit() hasn't been called?
            - what if fit() HAS been called?
        :param params: dict
        :return: self
        """
        self._sub_params = params.copy()
        return self
    '''

    def _filter_inputs(self, X):
        ''' preprocess data (filter input columns) '''
        
    def fit(self, X, y):
        return

    def fit_transform(self, X, y):
        self.fit(X[self.inputs], y)
        ps = self.predict(X[self.inputs])
        assert self.name not in X.columns
        # return np.concatenate((X, ps), axis=1)
        X[self.name] = ps  # MODIFYING THE PASSED-IN X!!
        return X

    def transform(self, X):
        ps = self.predict(X[self.inputs])
        # dual purpose of ensuring X follows Dataframe interface,
        # as well as doesn't already have a column called <self.name>
        assert self.name not in X.columns
        # return np.concatenate((X, ps), axis=1)
        X[self.name] = ps  # MODIFYING THE PASSED-IN X!!
        return X

    def predict(self, X):
        return self._clf.predict(X[self.inputs])

    def score(self, y, y_pred, **kwargs):
        if self._clf_score:
            self._clf_score(y, y_pred, kwargs)
        else:
            return self._clf.score(y, y_pred, kwargs)

    def print(self):
        print(self._clf)


def from_dict(model_params_dct):
    ''' there are 3 top-level attributes required:
        [klass_module, klass_class, params]
    '''
    import sys

    import algs.lp
    import algs.rf
    import algs.split_combine_estimator

    name = model_params_dct['name']
    # klass = globals()[model_params_dct['klass']]
    klass = getattr(sys.modules[model_params_dct['klass_module']],
        model_params_dct['klass_class'])
    return klass(name, **(model_params_dct.get('params',{})))


def pipeline_from_dicts(model_params_dcts):
    # kinda weird, both the pipeline and Model itself store model names.
    models = [(m['name'], from_dict(m)) for m in model_params_dcts]
    from sklearn.pipeline import Pipeline
    pipe = Pipeline(models)
    return pipe


if __name__ == '__main__':
    import test_data as td
    X, y = td.generate_test_data()

    import yaml
    #config_fn = '/home/John/Quantera/bs_study/algs/model_test.yaml'
    config_fn = 'model_test.yaml'
    model_configs = yaml.safe_load(open(config_fn))['model_pipeline']
    for model_config in model_configs:
        model = from_dict(model_config)
        print(model)
        model.fit(X, y)

