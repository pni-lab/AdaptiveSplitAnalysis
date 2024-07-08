# from .base.utils import _optional_import_
# power = _optional_import_(".sklearn_interface", "pearsonr", "scipy")

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(learning_curve, learning_curve_predicted=None, power_curve=None, power_curve_lower=None, 
         power_curve_upper = None, power_curve_predicted=None, training_curve=None, 
         dummy_curve=None, stop=None, reason=None, ci='95%', grid=True,
         subplot_kw=None, gridspec_kw=None, **kwargs):
    total_sample_size = -1
    kwargs.setdefault('figsize', (8, 8))
    if learning_curve_predicted is not None:
        total_sample_size = learning_curve_predicted.df.index.max()
    if power_curve_predicted is not None:
        total_sample_size = np.max((total_sample_size, power_curve_predicted.df.index.max()))
    if total_sample_size == -1:
        total_sample_size = None

    if power_curve is not None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex='all', subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
                                       **kwargs)  # frameon=False removes frames
        plt.subplots_adjust(hspace=.05)
    else:
        fig, ax1 = plt.subplots(nrows=1, sharex='all', subplot_kw=subplot_kw, gridspec_kw=gridspec_kw,
                                **kwargs)  # todo: test me

    learning_curve.stat(ci=ci).plot(color='blue', ax=ax1)
    if total_sample_size is not None:
        learning_curve.fit(method='keeplast', extend=[total_sample_size]).plot(ax=ax1, color='blue')
        if dummy_curve is not None:
            dummy_curve.fit(method='keeplast', extend=[total_sample_size]).plot(color='gray', ax=ax1,
                                                                                linewidth=1, linestyle='--')
        if training_curve is not None:
            training_curve.fit(method='keeplast', extend=[total_sample_size]).plot(color='gray', ax=ax1,
                                                                                   linewidth=1, linestyle='--')

    if learning_curve_predicted is not None:
        learning_curve_predicted.stat().plot(ax=ax1, color='blue', alpha=0.8, linestyle='--')

    ax1.legend(['observed learning curve', 'predicted learning curve'])
    ax1.axvline(np.max(learning_curve.df.index), linestyle=':', color="black")  # current sample size

    if power_curve is not None:

        power_curve.plot(color='red', ax=ax2)
        ax2.fill_between(power_curve.index, y1=power_curve_lower, y2=power_curve_upper, alpha=0.2, color='red')
        ax2.set_ylim((0,1.2))
        #power_curve.stat(ci=ci).plot(color='red', ax=ax2)

        if power_curve_predicted is not None:
            power_curve_predicted.stat(ci=ci).plot(ax=ax2, linestyle='--', color='red')
        ax2.legend(['obs power', 'pred power'])
        ax2.axvline(np.max(learning_curve.df.index), linestyle=':', color="black")
        # ax2.text(np.max(testing_curve.df.index), np.min(power_curve.stat().df['mid'].values), 'now')
        ax2.set_title(None)

    if stop is not None:
        ax1.axvline(stop, linestyle='-', color="red")
        if power_curve is not None:
            ax2.text(stop, np.min(power_curve.values), ' STOP')
            ax2.axvline(stop, linestyle='-', color="red")
        if stop <= np.max(learning_curve.df.index):
            if reason == []:
                reason = 'Reason: None'
            elif reason is not None:
                reason = ' Reason: ' + reason[0]
            ax1.set_title('ADAPTIVESPLIT: STOP! ' + reason)
        else:
            ax1.set_title('ADAPTIVESPLIT: Continue collecting the training sample.'
                          'Earliest stopping: ' + str(stop) + '.')
    else:
        ax1.set_title("ADAPTIVESPLIT. (No stopping rule provided.)")

    if grid:
        ax1.grid(True)
        if power_curve is not None:
            ax2.grid(True)

    return fig


def estimate_sample_size(y_obs,
                         y_pred,
                         target_power,
                         power_estimator,
                         max_iter=100,
                         rel_pwr_threshold=0.001,
                         learning_rate=0.05):
    sample_size = len(y_pred)
    power_ref = power_estimator.estimate(y_obs,
                                         y_pred,
                                         sample_size=sample_size,
                                         n_jobs=-1)
    iter = 0

    while not math.isclose(power_ref, target_power, rel_tol=rel_pwr_threshold) and iter <= max_iter:
        if power_ref >= target_power:
            sample_size = sample_size * (1 - learning_rate)
        elif power_ref < target_power:
            sample_size = sample_size * (1 + learning_rate)
        iter += 1
        power_ref = power_estimator.estimate(y_obs,
                                             y_pred,
                                             sample_size=sample_size,
                                             n_jobs=-1)
    if iter == (max_iter + 1):
        warnings.warn('Iteration limit reached, check results and adjust parameters if needed')
    # ToDo: output the confidence intervals and expected score
    return round(sample_size)
