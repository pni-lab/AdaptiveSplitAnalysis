from adaptivesplit.sklearn_interface.split import AdaptiveSplit
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from adaptivesplit.base.resampling import PermTest
from sklearn.base import is_classifier
import pandas as pd
import numpy as np
import operator

##
# Define functions;
##

def truncate_dataset(X, y, sample_size):
    
    # Data is truncated depending on sample_size,
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_samples = X.to_numpy()
        X_samples = X_samples[0:sample_size, :]
    
    else:
        X_samples = X[0:sample_size, :]

    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y_samples = np.squeeze(y.to_numpy())
        y_samples = y_samples[0:sample_size]

    else:
        y_samples = np.squeeze(y)
        y_samples = y_samples[0:sample_size]

    return X_samples, y_samples

def custom_split(X, y, test_size, shuffle=False, random_state=None):
    
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        shuffle=shuffle)

    stop = len(y_train)

    return X_train, X_test, y_train, y_test, stop

def adaptive_split(X, y, estimator, total_sample_size, scoring=None,
                   stratify=None, fast_mode=False, predict=False, shuffle=False,
                   random_state=None, power_bootstrap_samples=1, window_size=10, 
                   plotting=False, n_jobs=-1):
    
    if not scoring:
        if is_classifier(estimator):
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_absolute_error'

    adsplit = AdaptiveSplit(total_sample_size=total_sample_size,
                            scoring=scoring, power_bootstrap_samples=power_bootstrap_samples,
                            window_size=window_size,n_jobs=n_jobs,
                            plotting=plotting)
    res, lc, pc, fig = adsplit(X, y, estimator, stratify=stratify, fast_mode=fast_mode, predict=predict,
                               random_state=random_state)
    stop = res.estimated_stop

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=stop,
                                                        random_state=random_state,
                                                        shuffle=shuffle)

    return X_train, X_test, y_train, y_test, stop, lc, pc, fig

def split_scores(X, y, estimator, sample_sizes, method, stratify=None, n_permutations=5000, plotting=False,
                 random_state=None, power_bootstrap_samples = None, window_size=None, n_jobs=-1, verbose=1):
    

    scores_on_train = []
    scores_on_test = []
    p_vals = []
    stops = []
    learning_curves = []
    power_curves = []
    plots = []

    for sample_size in sample_sizes:

        # truncate datasets;
        X_samples, y_samples = truncate_dataset(X, y, sample_size)

        # split truncated data using a custom split or adaptivesplit;
        if method == 'pareto':
            X_train, X_test, y_train, y_test, stop = custom_split(X_samples, y_samples,
                                                                  test_size=0.2, shuffle=False,
                                                                  random_state=random_state)

            # define empty variables for pareto;
            lc = None
            pc = None
            fig = None

        elif method == 'halfsplit':
            X_train, X_test, y_train, y_test, stop = custom_split(X_samples, y_samples,
                                                                  test_size=0.5, shuffle=False,
                                                                  random_state=random_state)

            # define empty variables for halfsplit;
            lc = None
            pc = None
            fig = None

        elif method == '90-10split':
            X_train, X_test, y_train, y_test, stop = custom_split(X_samples, y_samples,
                                                                  test_size=0.1, shuffle=False,
                                                                  random_state=random_state)

            # define empty variables for 90-10split;
            lc = None
            pc = None
            fig = None

        elif method == 'adaptivesplit':
            X_train, X_test, y_train, y_test, stop, lc, pc, fig = adaptive_split(X_samples, y_samples, estimator,
                                                                                 total_sample_size=sample_size,
                                                                                 random_state=random_state,
                                                                                 stratify=stratify,
                                                                                 power_bootstrap_samples=power_bootstrap_samples,
                                                                                 window_size=window_size,
                                                                                 plotting=plotting,
                                                                                 n_jobs=n_jobs)

        if is_classifier(estimator):
            scorer = accuracy_score
        else:
            scorer = mean_absolute_error

        # get stopping point, learning curve and power curve;
        stops.append(stop)
        learning_curves.append(lc)
        power_curves.append(pc)
        plots.append(fig)

        # get training and test score;
        estim = estimator.fit(X_train, y_train)
        y_pred = estim.predict(X_test)
        test_score = scorer(y_test, y_pred)

        if scorer == mean_absolute_error:  # get neg_mean_absolute_error;
            scores_on_train.append(-estim.score(X_train, y_train))
            scores_on_test.append(-test_score)
            compare_operator = operator.le
        else:
            scores_on_train.append(estim.score(X_train, y_train))
            scores_on_test.append(test_score)
            compare_operator = operator.ge

        # get test p-value;
        perm = PermTest(scorer, num_samples=n_permutations, compare=compare_operator)
        res = perm.test(y_test, y_pred, n_jobs=n_jobs, random_seed=random_state)
        p_vals.append(res[1])

    return scores_on_train, scores_on_test, p_vals, stops, learning_curves, power_curves, plots
