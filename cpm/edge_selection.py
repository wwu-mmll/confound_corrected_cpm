import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from pingouin import partial_corr


def one_sample_t_test(x):
    # use two-sided for correlations (functional connectome) or one-sided positive for NOS etc (structural connectome)
    _, p_value = ttest_1samp(x, popmean=0, nan_policy='omit', alternative='two-sided')
    return p_value


def partial_correlation(X, y, covariates, method: str = 'pearson'):
    p_values = list()

    df = pd.concat([pd.DataFrame(X, columns=[f"x{s}" for s in range(X.shape[1])]), pd.DataFrame({'y': y})], axis=1)
    cov_names = list()
    for c in range(covariates.shape[1]):
        df[f'cov{c}'] = covariates[:, c]
        cov_names.append(f'cov{c}')

    for xi in range(X.shape[1]):
        res = partial_corr(data=df, x=f'x{xi}', y='y', covar=cov_names, method=method)['p-val'].iloc[0]
        p_values.append(res)
    return np.asarray(p_values)
