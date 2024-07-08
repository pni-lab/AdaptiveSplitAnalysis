# extends adaptivesplit.base.power with scikit-learn functionality
import collections
import pandas as pd
from ..base.learning_curve import LearningCurve
from sklearn.model_selection import cross_val_predict
from ..base.power import *
from ..base.power import _get_sample_sizes
from .resampling import Resample
from .utils import statfun_as_callable

# shuffleCV predict x_y
# y, y_pred
# subsample, y, y_pred
# power for subsamples
# score for subsamples
#
#
# power curve predicted
# score confidence interval curve predicted


PredictedScoreAndPower = collections.namedtuple('Predicted', ['score', 'power'])


def predict_power_curve(estimator, X, y, power_estimator,
                        total_sample_size, stratify=None, sample_sizes=None, step=None,
                        cv=5,
                        num_samples=100,
                        scoring=None, verbose=True,
                        n_jobs=None,
                        random_state=None,  # todo: implement it!
                        **kwargs):
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = np.squeeze(y.to_numpy()) # squeezing avoids errors with some datasets;
    else:
        y = np.squeeze(y)

    sample_sizes = _get_sample_sizes(sample_sizes, step, len(y), total_sample_size)
    sample_sizes_power = total_sample_size - sample_sizes

    power_estimator.total_sample_size = None

    if scoring is None:
        #scoring = estimator.score  # todo does this work?
        pass # no, it does not work, requires fitted estimator. pass for now;
    else:
        scoring = statfun_as_callable(scoring)

    def stat_fun_score_and_power(_x, _y, **kwargs):
        if len(_y) == 0:
            score = np.nan
            power = 0
        else:
            score = scoring(_x, _y, **kwargs)
            power = power_estimator.estimate(_x, _y, sample_size=len(_y),
                                             verbose=False)[0]
        return PredictedScoreAndPower(score, power)

    def stat_fun_aggregate_samplesizes(_x, _y, **kwargs):
        # data has been shuffled
        # do cv prediction forsample_sizes whole sample
        pred_y = cross_val_predict(estimator, _x, _y,
                                   cv=cv)
        # cross_val_predict had a **kwargs argument that amounted to {random_seed:None}
        # it crashes since the function doesn't accept a random_seed argument
        # see also utils.py;

        # subsample score
        subsampler = Resample(stat_fun=stat_fun_score_and_power,
                              sample_size=sample_sizes_power,
                              num_samples=1,  # we use the outer scope
                              n_jobs=1,
                              verbose=False,
                              replacement=True)
        scores_and_powers = subsampler.bootstrap(pred_y, _y)
        return scores_and_powers  # per subsample

    shuffler = Resample(stat_fun=stat_fun_aggregate_samplesizes,
                        sample_size=len(y),
                        num_samples=num_samples,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        replacement=False,
                        message="Predict Power Curve"
                        )

    results = shuffler.subsample(X, y, stratify=stratify)

    results = results.squeeze().transpose([2, 0, 1])  # (type, bootstrap_iters, samples)

    pred_score = LearningCurve(data=results[0],
                               ns=sample_sizes,
                               scoring=scoring.__name__,
                               curve_type="predicted score",
                               description="predicted score",
                               )

    pred_power = LearningCurve(data=results[1],
                               ns=sample_sizes,
                               scoring=str(power_estimator),
                               curve_type="predicted power",
                               description="predicted power",
                               )

    return PredictedScoreAndPower(pred_score, pred_power)
