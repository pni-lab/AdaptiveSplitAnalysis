from ..base.utils import *
from sklearn.metrics import get_scorer as sklearn_get_scorer, check_scoring
from sklearn.model_selection._split import check_cv
import numpy as np


def get_sklearn_scorer(scoring):
    scoring = sklearn_get_scorer(scoring)

    def score_fun(x, y, **kwargs):
        return scoring._sign * scoring._score_func(x, y) 
        # _score_func had a **kwargs argument that amounted to {random_seed:None}
        # it crashes since the function does not accept a random_seed argument
        # see also power.py;

    score_fun.__name__ = 'sklearn_' + scoring._score_func.__name__
    return score_fun


def statfun_as_callable(stat_fun):
    if isinstance(stat_fun, str):
        return get_sklearn_scorer(stat_fun)
    else:
        return stat_fun

def calculate_ci(X, ci='95%'):
    if ci == '90%':
        Z = 1.64
    elif ci == '95%':
        Z = 1.96
    elif ci == '98%':
        Z = 2.33
    elif ci == '99%':
        Z = 2.58

    moe = Z*(np.std(X)/np.sqrt(len(X)))
    ci_lower = np.mean(X) - moe
    ci_upper = np.mean(X) + moe
    
    return ci_lower, ci_upper
