from algs.rf import RF
from algs.lp import LP
from algs.split_combine_estimator import SplitEstimator
#from algs.fm import FM

def from_dict(config):
    klass = config['class']
    model_class = globals()[klass]
    model = model_class(params=config['params'])
    return model

if __name__ == '__main__':
    import test_data as td
    X, y = td.generate_test_data()

    from algs.rf import RF
    import yaml
    #config_fn = '/home/John/Quantera/bs_study/config.yaml'
    config_fn = '/mnt/c/Users/John/qta/config.yaml'
    model_configs = yaml.safe_load(open(config_fn))['model']
    for model_config in model_configs:
        model = from_dict(model_config['clf'])
        model.fit(X, y)