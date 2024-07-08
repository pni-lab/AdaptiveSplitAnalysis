import collections
import pandas as pd
from ..base.split import *
from .power import PowerEstimatorBootstrap, create_power_stat_fun, predict_power_curve
from .learning_curve import calculate_learning_curve
from sklearn.base import is_classifier, is_regressor
from .utils import calculate_ci, check_cv, check_scoring, statfun_as_callable
import adaptivesplit.config as config
from sklearn import linear_model
from regressors import stats
from pygam import LinearGAM

AdaptiveSplitResults = collections.namedtuple('AdaptiveSplitResults', ['stop', 'predicted',
                                                                       'estimated_stop',
                                                                       'current_sample_size',
                                                                       'score_if_stop',
                                                                       'score_if_stop_now_ci',
                                                                       'power_if_stop_now',
                                                                       'power_if_stop_now_ci',
                                                                       'score_predicted',
                                                                       'score_predicted_ci',
                                                                       'power_predicted',
                                                                       'power_predicted_ci'])

# todo: investigate multi-target compatibility
def _calc_slope(series, alpha=0.05):

    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    Y = series.to_numpy().reshape(-1, 1)
    X = np.array(series.index.to_list()).reshape(-1, 1)

    model = linear_model.LinearRegression().fit(X, Y)
    p_vals = stats.coef_pval(model, X, Y)

    if max(p_vals) < alpha:
        return model.coef_[0][0]
    else:
        return np.NAN


def _pred_score(series, x_test, alpha=0.05):

    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    Y = series.to_numpy().reshape(-1, 1)
    X = np.array(series.index.to_list()).reshape(-1, 1)

    model = linear_model.LinearRegression().fit(X, Y)
    p_vals = stats.coef_pval(model, X, Y)

    if x_test is None:
        x_test = X[-1]

    # if p_vals[1] < alpha and model.coef_[0][0] > 0:
    if max(p_vals) < alpha and model.coef_[0][0] > 0:
        return model.predict(np.array(x_test).reshape(-1, 1))
    else:
        return np.NAN


class AdaptiveSplit:
    def __init__(self, total_sample_size=config._total_sample_size_, scoring=config._scoring_, cv=config._cv_,
                 step=config._step_, bootstrap_samples=config._bootstrap_samples_, 
                 power_bootstrap_samples =None, window_size=None, verbose=True, 
                 plotting=True, n_jobs=config._n_jobs_):

        self.total_sample_size = total_sample_size
        self.scoring = scoring
        self.cv = cv
        self.step = step
        self.bootstrap_samples = bootstrap_samples
        self.verbose = verbose
        self.plotting = plotting
        self.n_jobs = n_jobs
        self.reason = []
        self.window_size = window_size

        if not window_size:
            self.window_size = config._window_size_
        if not power_bootstrap_samples:
            self.power_bootstrap_samples = config._power_bootstrap_samples_
        else:
            self.power_bootstrap_samples = power_bootstrap_samples
            
    def fit(self, X, Y, estimator, stratify=None, fast_mode=False, predict=True, random_state=None):  
            # todo: stratification by target from config file

        self.X = X
        self.Y = Y
        self.estimator = estimator
        self.fast_mode = fast_mode
        self.predict = predict
        self.random_state = random_state

        if self.total_sample_size < len(Y):
            raise AttributeError(
                "total_sample_size must be greater than or equal to the actual sample size: " + str(len(Y)))
        
        self.power_estimator = PowerEstimatorBootstrap(
            power_stat_fun=create_power_stat_fun('permtest', statfun_as_callable(self.scoring), num_perm=100),
            bootstrap_samples=self.power_bootstrap_samples,  # it will be bootstrapped in the outer loop anyway
            total_sample_size=self.total_sample_size,
            alpha=config._alpha_,
            stratify=stratify,
            verbose=False
        )

        self.cv = check_cv(self.cv, self.Y, classifier=is_classifier(self.estimator))
        window_size = self.window_size

        # calculate the starting point here;
        if self.fast_mode:
            sample_sizes = np.arange(start=len(self.Y), stop=len(self.Y) - window_size, step=-self.step)[::-1]
            # adapt for classifier
        
        else:
            sample_size_multiplier = 0.2 # ToDo: Turn this into an argument;
            if is_classifier(estimator):
                
                # search for the label with the least members
                labels, counts = np.unique(self.Y, return_counts=True)
                ratio_of_smallest_class = np.min(counts)/len(self.Y)

                # make sure to start with an adequate starting_sample_size
                starting_sample_sizes = np.arange(start=len(self.Y), stop=self.cv.get_n_splits(), step=-self.step)[::-1]
                sample_sizes = starting_sample_sizes[ratio_of_smallest_class*starting_sample_sizes > self.cv.n_splits]
                
                # check if sample_sizes is calculated properly 
                # (sample_sizes might be an empty array if ratio_of_smallest_class is too small)
                if len(sample_sizes) == 0:
                    sample_sizes = starting_sample_sizes[sample_size_multiplier*starting_sample_sizes > self.cv.n_splits]

            else:
                starting_sample_sizes = np.arange(start=len(self.Y), stop=self.cv.get_n_splits(), step=-self.step)[::-1]
                sample_sizes = starting_sample_sizes[sample_size_multiplier*starting_sample_sizes > self.cv.n_splits]

        self.lc_train, self.lc_test, self.lc_dummy, self.lc_power = calculate_learning_curve(
            estimator=self.estimator,
            X=self.X,
            y=self.Y,
            cv=self.cv,
            stratify=stratify,
            power_estimator=self.power_estimator,
            scoring=self.scoring,
            sample_sizes=sample_sizes,
            num_samples=self.bootstrap_samples,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        if self.total_sample_size == len(self.Y):
            predict = False
        
        if predict == False:
            self.pred_power = None
            self.pred_score = None
            self.score = self.lc_test.stat().df['mid']
            self.score_upper = self.lc_test.stat().df['upper']
            self.score_lower = self.lc_test.stat().df['lower']
            self.power = self.lc_power.stat().df['mid']
            self.power_upper = self.lc_power.stat().df['upper']
            self.power_lower = self.lc_power.stat().df['lower']
        
        elif predict:
            pred = predict_power_curve(estimator=self.estimator,
                                       X=self.X,
                                       y=self.Y,
                                       power_estimator=self.power_estimator,
                                       total_sample_size=self.total_sample_size,
                                       step=self.step,
                                       cv=self.cv,
                                       num_samples=100,  # bootstrap
                                       scoring=self.scoring,  # todo None does not work
                                       verbose=True,
                                       n_jobs=self.n_jobs,
                                       random_state=None
                                       )
            self.score = pd.concat([self.lc_test.stat().df['mid'], pred.score.stat().df.iloc[1:, 0]])
            self.score_upper = pd.concat([self.lc_test.stat().df['upper'], pred.score.stat().df.iloc[1:, 1]])
            self.score_lower = pd.concat([self.lc_test.stat().df['lower'], pred.score.stat().df.iloc[1:, 2]])
            self.power = pd.concat([self.lc_power.stat().df['mid'], pred.power.stat().df.iloc[1:, 0]])
            self.power_upper = pd.concat([self.lc_power.stat().df['upper'], pred.power.stat().df.iloc[1:, 1]])
            self.power_lower = pd.concat([self.lc_power.stat().df['lower'], pred.power.stat().df.iloc[1:, 2]])
            self.pred_power = pred.power
            self.pred_score = pred.score

    def stopping_rule(self):
        # todo: refactor, settable as parameter or read from conf file
        min_training_sample_size = config._min_training_sample_size_  # 0 means this rule is switched off
        min_validation_sample_size = config._min_validation_sample_size_ # 0 means there is no max stopping point
        target_power = config._target_power_  # 0 mean this rule is switched off
        min_score = config._min_score_  # -np.inf means this rule is switched off
        
        window_size = self.window_size
            
        min_relevant_score = config._min_relevant_score_

        if self.predict == False:
            stop = self.total_sample_size
        else:
            stop = self.score.index.max()

        # Generate a smoother power curve using GAM; Todo: we should also smooth the prediction, if present;
        power_gam = []

        for sample_i, actual_sample_size in enumerate(self.power.index):

            if sample_i == 0: # check up gridsearch function for GAM;
                gam_X = np.array([self.power.index[sample_i]])[:, np.newaxis]
                gam_y = np.array([self.power.iloc[sample_i]])
            else:
                gam_X = np.array(self.power.index)[:sample_i][:, np.newaxis]
                gam_y = np.array(self.power)[:sample_i]

            gam = LinearGAM().gridsearch(gam_X, gam_y, n_splines=np.arange(1, 6))
            X_grid = gam.generate_X_grid(term=0, n=len(gam_X))
            power_gam.append(gam.predict(X_grid))
            
        # Prepare the power curve for rule evaluation and plotting
        mean_power_curve = pd.DataFrame(power_gam, columns=self.power.index[1:]).mean(axis=0)
        ci = calculate_ci(mean_power_curve, ci='95%')
        
        self.power = mean_power_curve
        self.power_lower = mean_power_curve - ci[0]
        self.power_upper = mean_power_curve + ci[1]

        power_slope = mean_power_curve.rolling(window_size).apply(_calc_slope, raw=False)  # returns NaN if NOT significant
        pred_final_score = self.score.rolling(window_size).apply(_pred_score, raw=False, args=(self.total_sample_size,))
        pred_actual_score = self.score.rolling(window_size).apply(_pred_score, raw=False, args=(None,))

        # Evaluate the rule
        for actual_sample_size in mean_power_curve.index:
            self.reason = []
            # max training size exceeded:
            if actual_sample_size >= self.total_sample_size - min_validation_sample_size:
                stop = actual_sample_size
                self.reason.append('max sample size reached')

            # preconditions for all other rules: min training size and min score must be exceeded
            if actual_sample_size > min_training_sample_size \
            and self.score.loc[actual_sample_size] > min_score:

                # power rule: power is already decreasing and we pass by the target power
                if power_slope[actual_sample_size] < 0 and mean_power_curve.loc[actual_sample_size] <= target_power:
                    stop = actual_sample_size
                    self.reason.append('power rule')

                # score rule: optimistic (linear) extrapolation of the power curve predicts very little gain, i.e. power curve plateaus
                if (pred_final_score[actual_sample_size] - pred_actual_score[actual_sample_size]) < min_relevant_score:
                    stop = actual_sample_size
                    self.reason.append('score rule')

            if stop == actual_sample_size:
                self.reason = ', '.join(self.reason)
                break

        # Assure that reason is not an empty list (in case no stopping point is found)
        #if len(self.reason) == 0:
        #    self.reason.append('No stopping point found')

        # FROM HERE: Modified for analyses;
        if self.plotting == True:
            fig = plot(learning_curve=self.lc_test, learning_curve_predicted=self.pred_score,
                power_curve=self.power, power_curve_lower=self.power_lower, power_curve_upper= self.power_upper, 
                power_curve_predicted=self.pred_power, training_curve=self.lc_train, dummy_curve=self.lc_dummy, 
                stop=stop, reason=self.reason)
        else:
            fig = plt.figure()

        if stop > len(self.Y):
            is_stop_predicted = True
        else:
            is_stop_predicted = False

        current = len(self.Y)

        learning_curve = np.array(self.lc_test.stat().df.loc[:, 'mid'])
        power_curve = np.array(self.lc_power.stat().df.loc[:, 'mid'])

        if self.predict == False:
            return AdaptiveSplitResults(stop <= current,
                                        is_stop_predicted,
                                        stop,
                                        current,

                                        self.score[current],
                                        (self.score_lower[current], self.score_upper[current]),
                                        self.power[current],
                                        (self.power_lower[current], self.power_upper[current]),
                                        
                                        None,
                                        (None, None),
                                        None,
                                        (None, None),
                                        ), learning_curve, power_curve, fig

        else: 
            return AdaptiveSplitResults(stop <= current,
                                        is_stop_predicted,
                                        stop,
                                        current,

                                        self.score[current],
                                        (self.score_lower[current], self.score_upper[current]),
                                        self.power[current],
                                        (self.power_lower[current], self.power_upper[current]),
                                        
                                        self.score[stop],
                                        (self.score_lower[stop], self.score_upper[stop]),
                                        self.power[stop],
                                        (self.power_lower[stop], self.power_upper[stop]),
                                        ), fig # add figure return also for adaptivesplit with prediction;

    def __call__(self, data, target, estimator, stratify=None, fast_mode=False, predict=True, random_state=None):

        self.fit(X=data, Y=target, estimator=estimator, stratify=stratify, fast_mode=fast_mode, predict=predict, 
                 random_state=random_state)
        if predict == False:
            res, lc, pc, fig = self.stopping_rule()
            return res, lc, pc, fig
        else:
            # prediction is not needed in the analysis and returns only the result and the figure;
            res, fig = self.stopping_rule()
            return res
        