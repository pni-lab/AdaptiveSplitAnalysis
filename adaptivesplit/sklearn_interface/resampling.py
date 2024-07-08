# extends adaptivesplit.base.resampling with scikit-learn functionality

import collections
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.base import is_classifier, is_regressor
from .utils import check_cv
from .utils import statfun_as_callable as utils_statfun_as_callable
from ..base.resampling import Resample as BaseResample
from ..base.resampling import PermTest as BasePermTest


class PermTest(BasePermTest):
    def _set_stat_fun(self, stat_fun):
        self.stat_fun = utils_statfun_as_callable(stat_fun)


class Resample(BaseResample):
    def _set_stat_fun(self, stat_fun):
        self.stat_fun = utils_statfun_as_callable(stat_fun)


SubSampledStats = collections.namedtuple('SubsampledStats', ['train_score', 'test_score', 'dummy_score', 'power'])


class SubSampleCV(Resample):

    def __init__(self, estimator, sample_size, dummy_estimator=None, num_samples=100,
                 cv=None, cv_stat=np.mean, groups=None, scoring=None, power_estimator=None,
                 n_jobs=-1, verbose=True, message="Calculating learning curve"):

        self.cv = cv
        self.cv_stat = cv_stat
        if scoring is None:
            self.scoring = estimator.score  # default scorer of the estimator
        else:
            self.scoring = scoring
        self.groups = groups
        self.power_estimator = power_estimator

        # statfun to be evaluated in the outer reasampling (subsampling).
        def stat_fun(x, y, cv, scoring, groups, **kwargs):

            cvfit = cross_validate(estimator, x, y,
                                   cv=cv,
                                   scoring=scoring,
                                   return_train_score=True,
                                   return_estimator=True)
            if dummy_estimator is not None:
                cvfit_dummy = cross_validate(dummy_estimator, x, y, cv=cv, scoring=scoring)
                dummy_score = cv_stat(cvfit_dummy["test_score"])
            else:
                dummy_score = None

            # get cross-validated predictions:
            if power_estimator is not None:
                bestimators = cvfit["estimator"]
                cv = check_cv(cv, y, classifier=is_classifier(estimator))
                idx = 0
                predicted = np.zeros_like(y) # instead of np.zeros(len(y))
                # calculate cross-validated predictions
                for train, test in cv.split(x, y, groups):
                    predicted[test] = bestimators[idx].predict(x[test])
                    idx += 1

                # calculate Power here:
                power = power_estimator.estimate(y_true=y, y_pred=predicted,
                                                 sample_size=len(y),
                                                 power_statfun=scoring,  # this always overrides it! # todo: fixme
                                                 **kwargs
                                                 )[0]  # one sample size only
            else:
                power = None

            train_score = cv_stat(cvfit["train_score"])
            test_score = cv_stat(cvfit["test_score"])

            return SubSampledStats(train_score, test_score, dummy_score, power)

        super().__init__(stat_fun=stat_fun, sample_size=sample_size, num_samples=num_samples,
                         replacement=False, n_jobs=n_jobs, verbose=verbose, message=message)

    # convenience function:
    def subsample(self, x, y, stratify=None, sample_size=None, num_samples=None, replacement=None,
                  cv=None, cv_stat=None, groups=None, scoring=None,
                  random_seed=None, n_jobs=None, verbose=None, *args, **kwargs):
        self._update_defaults(cv=cv, cv_stat=cv_stat, scoring=scoring, groups=groups)
        return self.fit_transform(x, y, stratify=stratify, sample_size=sample_size, num_samples=num_samples, replacement=replacement,
                                  compare=None, n_jobs=n_jobs, verbose=verbose, random_seed=random_seed, **kwargs)

    def fit_transform(self, x, y, stratify=None, sample_size=None, num_samples=None, replacement=None, compare=None,
                      n_jobs=None, verbose=None, random_seed=None, **kwargs):
        stats = super().fit_transform(x, y, stratify=stratify, sample_size=sample_size, num_samples=num_samples, replacement=replacement,
                                      compare=compare, n_jobs=n_jobs, verbose=verbose, random_seed=random_seed,
                                      cv=self.cv, scoring=self.scoring, groups=self.groups)  # passed as kwargs

        self.stats = np.array(stats).transpose(
            (2, 1, 0))  # (['train', 'test', 'dummy', 'power'], num_samples, sample_size)

        return self.stats

    def plot(self, *args, **kwargs):
        if isinstance(self.sample_size, (int, float)):
            sample_size = [self.sample_size]
        else:
            sample_size = self.sample_size

        titles = ['training', 'test', 'dummy', 'power']
        kwargs.setdefault('kind', 'line')
        for s_i, s in enumerate(self.stats):
            if np.any(s) is not None:
                df = pd.DataFrame(s.transpose(), index=np.array(sample_size, dtype=int))
                ax = self._plot(df, blend_hist=0, *args, **kwargs)
                ax.set(xlabel='sample_size')
                ax.set(title=titles[s_i])
