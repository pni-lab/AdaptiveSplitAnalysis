##
# Place imports;
##
import os
import sys

from torch import frac

sys.path.insert(0, '.')
os.chdir('.')

from Analysis import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neurocombat_sklearn import CombatModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

##
# Load a dataset;
##

data = pd.read_csv('../data_in/IXI/ixi.csv', delimiter=',', encoding='utf-8')
data.drop_duplicates(subset="ID", keep='first', inplace=True)
data.index = data['ID']

batches = pd.read_csv('../data_in/IXI/final_demo.csv', delimiter=',', encoding='utf-8')
batches.index = batches['IXI_ID']

# Using .loc instead of .ix
data = data.loc[batches['IXI_ID']]

data.insert(0, 'Batch', batches['center'])
data.dropna(inplace=True)  # If any;

X = data[data.columns[3:]]  # .sample(frac=1, random_state=42)
y = data['Age']
batch = data['Batch']

# model = CombatModel()
# model.fit(X, np.array(batch).reshape(-1, 1))
# X_combat = model.transform(X, np.array(batch).reshape(-1, 1))

##
# Choose an estimator;
##

from sklearn.linear_model import RidgeCV

# model = RidgeCV(scoring='neg_mean_absolute_error',
#                 alphas=(0.1, 1, 10))  # default alpha values;

model = Pipeline([('prep', StandardScaler()),
                  ('est', RidgeCV(scoring='neg_mean_absolute_error',
                                  alphas=(0.1, 1, 10)))])

##
# Run Analyses;
##

from adaptivesplit import config

results_path = '../tests/IXI/'
plot_path = '../tests/IXI/plots/'

results = pd.DataFrame()
sample_sizes = np.logspace(np.log10(50), np.log10(150), 5).astype(int)
n_permutation = 100

counter = 0
for iteration in range(0, n_permutation):

    seed = iteration
    idx_set = np.arange(len(y))
    random_generator = np.random.default_rng(seed)
    perm_index_set = random_generator.permutation(idx_set)

    X_perm = X.iloc[perm_index_set, :]
    y_perm = y.iloc[perm_index_set]

    for index in range(len(sample_sizes)):
        train_pareto, test_pareto, p_vals_pareto, stops_pareto, lc_pareto, pc_pareto, plots_pareto = split_scores(
            X_perm, y_perm, model,
            sample_sizes=[sample_sizes[index]],
            method='pareto', random_state=seed)

        train_halfsplit, test_halfsplit, p_vals_halfsplit, stops_halfsplit, lc_halfsplit, pc_halfsplit, plots_halfsplit = split_scores(
            X_perm, y_perm, model,
            sample_sizes=[sample_sizes[index]],
            method='halfsplit', random_state=seed)

        train_9010, test_9010, p_vals_9010, stops_9010, lc_9010, pc_9010, plots_9010 = split_scores(
            X_perm, y_perm, model,
            sample_sizes=[sample_sizes[index]],
            method='90-10split', random_state=seed)

        train_ads, test_ads, p_vals_ads, stops_ads, lc_ads, pc_ads, plots_ads = split_scores(X_perm, y_perm, model,
                                                                                             sample_sizes=[
                                                                                                 sample_sizes[index]],
                                                                                             method='adaptivesplit',
                                                                                             power_bootstrap_samples=10,
                                                                                             plotting=True,
                                                                                             random_state=seed)

        # Put stopping rules info inside the DataFrame:
        results.loc[counter, 'min_training_sample_size'] = config._min_training_sample_size_
        results.loc[counter, 'max_training_sample_size'] = config._max_training_sample_size_
        results.loc[counter, 'target_power'] = config._target_power_
        results.loc[counter, 'alpha'] = config._alpha_
        results.loc[counter, 'min_score'] = config._min_score_
        results.loc[counter, 'min_relevant_score'] = config._min_relevant_score_
        results.loc[counter, 'min_validation_sample_size'] = config._min_validation_sample_size_
        results.loc[counter, 'window_size'] = config._window_size_
        results.loc[counter, 'step'] = config._step_

        # Put AdaptiveSplit info inside the DataFrame:
        results.loc[counter, 'cv'] = config._cv_
        results.loc[counter, 'bootstrap_samples'] = config._bootstrap_samples_
        results.loc[counter, 'scoring'] = config._scoring_

        # Put truncation info inside the DataFrame;
        results.loc[counter, 'truncate_sample_size'] = sample_sizes[index]
        results.loc[counter, 'pareto_sample_size'] = stops_pareto
        results.loc[counter, 'halfsplit_sample_size'] = stops_halfsplit
        results.loc[counter, 'split90-10_sample_size'] = stops_9010
        results.loc[counter, 'adaptivesplit_sample_size'] = sample_sizes[index] + (stops_ads - sample_sizes[index])

        # Put split_scores results inside the DataFrame:
        results.loc[counter, 'pareto_train_scores'] = train_pareto
        results.loc[counter, 'halfsplit_train_scores'] = train_halfsplit
        results.loc[counter, 'split90-10_train_scores'] = train_9010
        results.loc[counter, 'adaptivesplit_train_scores'] = train_ads

        results.loc[counter, 'pareto_test_scores'] = test_pareto
        results.loc[counter, 'halfsplit_test_scores'] = test_halfsplit
        results.loc[counter, 'split90-10_test_scores'] = test_9010
        results.loc[counter, 'adaptivesplit_test_scores'] = test_ads

        results.loc[counter, 'pareto_p_values'] = p_vals_pareto
        results.loc[counter, 'halfsplit_p_values'] = p_vals_halfsplit
        results.loc[counter, 'split90-10_p_values'] = p_vals_9010
        results.loc[counter, 'adaptivesplit_p_values'] = p_vals_ads

        # Put permutation info inside the DataFrame;
        results.loc[counter, 'n_permutation'] = iteration
        results.loc[counter, 'random_seed'] = seed

        # Put learning and power curve in the DataFrame;
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
