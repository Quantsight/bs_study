import numpy as np
import pandas as pd

def generate_test_data():
    ROWS = 1000
    SPLITS = 10
    df = pd.DataFrame(np.random.randn(ROWS,4),columns=list('ABCD'))
    split_vals = []
    for _ in range(SPLITS):
        split_vals.extend([_]*(ROWS/SPLITS))
    df['S'] = split_vals
    y = pd.Series(np.random.randn(ROWS))
    return df, y
