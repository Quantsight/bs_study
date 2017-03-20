from __future__ import print_function

import numpy as np

class ModelBase(object):
    def __init__(self, name=None, inputs=None, inputs_exclude_flag=False, clf_params={}, cv_params={}):
        ''' 
        name: string with name of the instance
        clf: sklearn Estimator class to use
        inputs: string storing comma-separated list of names
        clf_params:
        cv_params:
        inputs_exclude_flag: if true, then <inputs> is list to _exclude_

        '''
        assert name is not None
        assert inputs is not None
        self.name = name
        self.inputs = inputs.split()
        self._clf_params = clf_params
        # fit_params ?
        # inputs ?
        # inputs_exclude_flag ?

    def get_param_dist(self, xs):
        return {}

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

    def score(self, ):
        pass

    def print(self):
        print(self._clf)


def from_dict(model_params_dct):
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
    models = [(m['name'], from_dict(**m)) for m in model_params_dcts]
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

