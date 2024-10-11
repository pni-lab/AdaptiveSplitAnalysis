##
# Place imports;
##

import os
import sys
sys.path.insert(0, '.')
os.chdir('.')

from Analysis import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import use
use('Agg')

##
# Load dataset;
##

### Put the ABIDE notebook in the notebooks folder and just load data from here ###
X = np.load('/groups/pni/datasets/ABIDE/abide_X.npy')
y = np.load('/groups/pni/datasets/ABIDE/abide_y_(diag).npy')

##
# Choose an estimator;
##

from sklearn.linear_model import RidgeClassifierCV

model = RidgeClassifierCV(scoring='accuracy',
                          alphas=(0.1, 1, 10)) # default alpha values;

##
# Run Analyses;
##

from adaptivesplit import config

import sys
import os


results_path = './tests/ABIDE/'
plot_path = './tests/ABIDE/plots/'

if os.path.isdir(results_path):
    print("continue")
else:
    sys.exit()

results = pd.DataFrame()
sample_sizes = np.logspace(np.log10(400), np.log10(600), 5).astype(int)
n_permutation = 100

counter = 0
for iteration in range(0, n_permutation):

    seed = iteration
    idx_set = np.arange(len(y))
    random_generator = np.random.default_rng(seed)
    perm_index_set = random_generator.permutation(idx_set)

    X_perm = X[perm_index_set, :]
    y_perm = y[perm_index_set]

    for index in range(len(sample_sizes)):
        train_pareto, test_pareto, p_vals_pareto, stops_pareto, lc_pareto, pc_pareto, plots_pareto, dscores_pareto = split_scores(
                                                                                            X_perm, y_perm, model,
                                                                                            sample_sizes=[sample_sizes[index]],
                                                                                            method='pareto', random_state=seed)

        train_halfsplit, test_halfsplit, p_vals_halfsplit, stops_halfsplit, lc_halfsplit, pc_halfsplit, plots_halfsplit, dscores_halfsplit = split_scores(
                                                                                            X_perm, y_perm, model,
                                                                                            sample_sizes=[sample_sizes[index]],
                                                                                            method='halfsplit', random_state=seed)

        train_9010, test_9010, p_vals_9010, stops_9010, lc_9010, pc_9010, plots_9010, dscores_9010 = split_scores(
                                                                                            X_perm, y_perm, model,
                                                                                            sample_sizes=[sample_sizes[index]],
                                                                                            method='90-10split', random_state=seed)

        train_75_25, test_75_25, p_vals_75_25, stops_75_25, lc_75_25, pc_75_25, plots_75_25, dscores_75_25  = split_scores(
                                                                                            X_perm, y_perm, model,
                                                                                            sample_sizes=[sample_sizes[index]],
                                                                                            method='75-25split', random_state=seed)

        train_70_30, test_70_30, p_vals_70_30, stops_70_30, lc_70_30, pc_70_30, plots_70_30, dscores_70_30 = split_scores(
                                                                                            X_perm, y_perm, model,
                                                                                            sample_sizes=[sample_sizes[index]],
                                                                                            method='70-30split', random_state=seed)

        train_ads, test_ads, p_vals_ads, stops_ads, lc_ads, pc_ads, plots_ads, dscores_ads = split_scores(X_perm, y_perm, model,
                                                                                             sample_sizes=[
                                                                                                 sample_sizes[index]],
                                                                                             method='adaptivesplit',
                                                                                             power_bootstrap_samples=1,
                                                                                             plotting=True,
                                                                                             random_state=seed)

        # Put stopping rules info inside the Dataframe:
        results.loc[counter, 'min_training_sample_size'] = config._min_training_sample_size_
        results.loc[counter, 'max_training_sample_size'] = config._max_training_sample_size_
        results.loc[counter, 'target_power'] = config._target_power_
        results.loc[counter, 'alpha'] = config._alpha_
        results.loc[counter, 'min_score'] = config._min_score_
        results.loc[counter, 'min_relevant_score'] = config._min_relevant_score_
        results.loc[counter, 'min_validation_sample_size'] = config._min_validation_sample_size_
        results.loc[counter, 'window_size'] = config._window_size_
        results.loc[counter, 'step'] = config._step_

        # Put AdaptiveSplit info inside the Dataframe:
        results.loc[counter, 'cv'] = config._cv_
        results.loc[counter, 'bootstrap_samples'] = config._bootstrap_samples_
        results.loc[counter, 'scoring'] = config._scoring_

        # Put truncation info inside the DataFrame;
        results.loc[counter, 'truncate_sample_size'] = sample_sizes[index]
        results.loc[counter, 'pareto_sample_size'] = stops_pareto[0]
        results.loc[counter, 'halfsplit_sample_size'] = stops_halfsplit[0]
        results.loc[counter, 'split90-10_sample_size'] = stops_9010[0]
        results.loc[counter, 'split75-25_sample_size'] = stops_75_25[0]
        results.loc[counter, 'split70-30_sample_size'] = stops_70_30[0]
        results.loc[counter, 'adaptivesplit_sample_size'] = sample_sizes[index] + (stops_ads - sample_sizes[index])[0]

        # Save all discovery scores (training cross-validation);
        results.loc[counter, 'pareto_discovery_scores'] = dscores_pareto.tobytes()
        results.loc[counter, 'halfsplit_discovery_scores'] = dscores_halfsplit.tobytes()
        results.loc[counter, 'split90-10_discovery_scores'] = dscores_9010.tobytes()
        results.loc[counter, 'split75-25_discovery_scores'] = dscores_75_25.tobytes()
        results.loc[counter, 'split70-30_discovery_scores'] = dscores_70_30.tobytes()
        results.loc[counter, 'adaptivesplit_discovery_scores'] = dscores_ads.tobytes()

        # Put split_scores results inside the DataFrame;
        results.loc[counter, 'pareto_train_scores'] = train_pareto[0]
        results.loc[counter, 'halfsplit_train_scores'] = train_halfsplit[0]
        results.loc[counter, 'split90-10_train_scores'] = train_9010[0]
        results.loc[counter, 'split75-25_train_scores'] = train_75_25[0]
        results.loc[counter, 'split70-30_train_scores'] = train_70_30[0]
        results.loc[counter, 'adaptivesplit_train_scores'] = train_ads[0]

        results.loc[counter, 'pareto_test_scores'] = test_pareto[0]
        results.loc[counter, 'halfsplit_test_scores'] = test_halfsplit[0]
        results.loc[counter, 'split90-10_test_scores'] = test_9010[0]
        results.loc[counter, 'split75-25_test_scores'] = test_75_25[0]
        results.loc[counter, 'split70-30_test_scores'] = test_70_30[0]
        results.loc[counter, 'adaptivesplit_test_scores'] = test_ads[0]

        results.loc[counter, 'pareto_p_values'] = p_vals_pareto[0]
        results.loc[counter, 'halfsplit_p_values'] = p_vals_halfsplit[0]
        results.loc[counter, 'split90-10_p_values'] = p_vals_9010[0]
        results.loc[counter, 'split75-25_p_values'] = p_vals_75_25[0]
        results.loc[counter, 'split70-30_p_values'] = p_vals_70_30[0]
        results.loc[counter, 'adaptivesplit_p_values'] = p_vals_ads[0]

        # Put permutation info inside the Dataframe;
        results.loc[counter, 'n_permutation'] = iteration
        results.loc[counter, 'random_seed'] = seed

        # Put learning and power curve in the Dataframe;
        results.loc[counter, 'adaptivesplit_learning_curve'] = np.nan
        results['adaptivesplit_learning_curve'] = results['adaptivesplit_learning_curve'].astype('object')
        results.at[counter, 'adaptivesplit_learning_curve'] = list(lc_ads[0])

        results.loc[counter, 'adaptivesplit_power_curve'] = np.nan
        results['adaptivesplit_power_curve'] = results['adaptivesplit_power_curve'].astype('object')
        results.at[counter, 'adaptivesplit_power_curve'] = list(pc_ads[0])

        # Save plot for the current sample_size:
        plots_ads[0].savefig(plot_path + 'adaptivesplit_' + str(counter) + '.png')
        plt.close()

        counter += 1

results.to_csv(results_path + 'results.csv')
