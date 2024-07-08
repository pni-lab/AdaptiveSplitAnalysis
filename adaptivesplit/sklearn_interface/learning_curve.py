#extends basse.learning_curve with scikit-learn functionality
from ..base.learning_curve import *
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.base import is_classifier, is_regressor
from .utils import check_cv, check_scoring
from .resampling import SubSampleCV


"""
def lc_keymaker(estimator, X, y, ns, cv=5, cv_stat=np.mean, dummy_estimator=None,
                shuffle=-1, replacement=False, scoring=None, verbose=True, n_jobs=None, random_state=None,
                *args, **kwargs):
    # resolve nans:
    if dummy_estimator is None:
        dummy_estimator = 'default'
    if scoring is None:
        scoring = 'default'
    if n_jobs is None:
        n_jobs = 'default'
    if random_state is None:
        random_state = np.random.normal()

    return str(estimator), str(X), str(y), str(ns), str(cv), str(
        cv_stat), dummy_estimator, shuffle, replacement, scoring, verbose, n_jobs, random_state
"""


# factory function for sklearn
# @cached(max_size=64, custom_key_maker=lc_keymaker)
def calculate_learning_curve(estimator, X, y, sample_sizes, stratify=None, cv=5, cv_stat=np.mean, dummy_estimator=None,
                             num_samples=1,
                             power_estimator=None,
                             scoring=None, verbose=True,
                             n_jobs=None,
                             random_state=None,
                             *args, **kwargs):
    """
            Factory function to calculate learning curve(s) with a given model and data.
            Learning curve(s) for the test data are stried in the current object (overwrites current learning_curve).
            Learning curve(s) for the training data are returned as a new LearningCurve object.
            :param estimator: estimator object.
            A object of that type is instantiated for each grid point.
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a score function, or scoring must be passed.
            If it is e.g. a GridSearchCV then nested cv is performed (recommended).
            :param X: array-like of shape (n_samples, n_features)
            The data to fit as in scikit-learn. Can be for example a list, an array or pandas DataFrame.
            :param y: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning, as in scikit-learn.
            :param sample_sizes: int or list of int
            sample sizes to calculate the learning curve
            :param cv: int, cross-validation generator or an iterable, default=None
            Determines the cross-validation splitting strategy, as in scikit-learn. Possible inputs for cv are:
            - None, to use the default 5-fold cross validation,
            - int, to specify the number of folds in a (Stratified)KFold,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.
            For int/None inputs, if the estimator is a classifier and y is either binary or multiclass,
            StratifiedKFold is used. In all other cases, Fold is used.
            These splitters are instantiated with shuffle=False so the splits will be the same across calls.
            :param cv_stat: callable
            Function for aggregating cv-wise scores.
            :param dummy_estimator: estimator object or None (default None)
            A scikit-learn-like dummy estimator to evaluate baseline performance.
            If None, either DummyClassifier() or DummyRegressor() are used, based on 'estimator's tpye.
            :param num_samples: int
            Nubmer of iterations to shuffle data before determining subsamples.
            The first iteration (index 0) is ALWAYS unshuffled (num_samples=1 implies no resampling at all, default).
            :param power_estimator: callable or None
            Callable must be a power_estimator function, see the 'create_power_estimator*' factory functions.
            If None, power curve is not claculated.
            :param scoring: str, callable, list, tuple or dict, default=None
            Scikit-learn-like score to evaluate the performance of the cross-validated model on the test set.
            If scoring represents a single score, one can use:
            - a single string (see The scoring parameter: defining model evaluation rules);
            - a callable (see Defining your scoring strategy from metric functions) that returns a single value.
            If scoring represents multiple scores, one can use:
            - a list or tuple of unique strings;
            - a callable returning a dictionary where the keys are the metric names and the values are the metric scores;
            - a dictionary with metric names as keys and callables a values.
            If None, the estimator’s score method is used.
            :param verbose: bool
             Print progress.
            :param n_jobs: int
            Number of jobs to run in parallel (default=None).
            Training the estimator and computing the score are parallelized over the cross-validation splits.
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
            :param random_state :int or None, default=None
            Controls the randomness of the bootstrapping of the samples used when building sub-samples (if shuffle!=-1)
            :param *args, **kwargs
            Extra parameters passed to sklearn.model_selection.cross_validate
            """

    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = np.squeeze(y.to_numpy()) # squeezing avoids errors with some datasets;
    else:
        y = np.squeeze(y)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    if isinstance(sample_sizes, (int, float)):
        inc = (len(y) - cv.get_n_splits()) / sample_sizes
        sample_sizes = np.arange(start=cv.get_n_splits(), stop=len(y)+inc, step=inc)

    if scoring is None:
        scoring = check_scoring(estimator)

    if dummy_estimator is None:
        if is_classifier(estimator):
            dummy_estimator = DummyClassifier() # strategy='stratified'?
        elif is_regressor(estimator):
            dummy_estimator = DummyRegressor()
        else:
            raise RuntimeError("Estimator can only be classifier or regressor.")

    subsampler = SubSampleCV(estimator=estimator,
                             dummy_estimator=dummy_estimator,
                             sample_size=sample_sizes,
                             num_samples=num_samples,
                             cv=cv,
                             cv_stat=cv_stat,
                             power_estimator=power_estimator,
                             scoring=scoring,
                             verbose=verbose,
                             n_jobs=n_jobs
                             )
    stats = subsampler.subsample(X, y, stratify=stratify, random_seed=random_state)

    # return the stuff
    lc_train = LearningCurve(data=stats[0, :, :],
                             ns=sample_sizes,
                             scoring=scoring,
                             description={
                                 "shuffles": num_samples},
                             curve_type="train"
                             )

    lc_test = LearningCurve(data=stats[1, :, :],
                            ns=sample_sizes,
                            scoring=scoring,
                            description={
                                "shuffles": num_samples},
                            curve_type="test"
                            )

    lc_dummy = LearningCurve(data=stats[2, :, :],
                             ns=sample_sizes,
                             scoring=scoring,
                             description={
                                 "shuffles": num_samples},
                             curve_type="dummy"
                             )

    if power_estimator is not None:
        lc_power = LearningCurve(data=stats[3, :, :],
                                 ns=sample_sizes,
                                 scoring=scoring,
                                 description={
                                     "shuffles": num_samples},
                                 curve_type="power"
                                 )
        return lc_train, lc_test, lc_dummy, lc_power

    return lc_train, lc_test, lc_dummy