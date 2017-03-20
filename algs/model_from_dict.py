def from_dict(name, clf, inputs, inputs_exclude_flag=False, params={}, cv_params={}):
    ''' config is dict with:

    clf: class (derived from Model) to use
    inputs: string storing comma-separated list of names
    clf_params:
    cv_params:
    inputs_exclude_flag: if true, then <inputs> is list to _exclude_

    '''
    model_class = globals()[clf]
    return model_class(name, params=params)


def pipeline_from_dicts(dcts):
    # kinda weird, both the pipeline and Model itself store model names.
    models = [(m['name'], from_dict(**m)) for m in dcts]
    from sklearn.pipeline import Pipeline
    pipe = Pipeline(models)
    return pipe


if __name__ == '__main__':
    import test_data as td
    X, y = td.generate_test_data()

    from algs.rf import RF
    import yaml
    #config_fn = '/home/John/Quantera/bs_study/config.yaml'
    config_fn = '/mnt/c/Users/John/qta/config.yaml'
    model_configs = yaml.safe_load(open(config_fn))['model']
    for model_config in model_configs:
        model = from_dict(model_config['name'], model_config['clf'], model_config['clf'])
        model.fit(X, y)

from algs.rf import RF
from algs.lp import LP
from algs.split_combine_estimator import SplitEstimator
# from algs.fm import FM
