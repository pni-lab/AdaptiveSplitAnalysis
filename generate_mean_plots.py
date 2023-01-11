import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def mean_window(arr, n_samples, n_permutations):
    windows = np.array(arr).reshape(n_permutations, n_samples)
    avg = np.mean(windows, axis=0)
    return avg

# Get average performance for all the datasets;
datasets = ['boston', 'bcw', 'banknotes']
average_data = pd.DataFrame()

for dataset in datasets:
    data = pd.read_csv('./' + dataset + '/results.csv', delimiter=',', encoding='utf-8')

    n_samples = 5
    n_permutations = 10

    # get averaged train scores;
    average_data[dataset + '_pareto_train_scores'] = mean_window(data['pareto_train_scores'], n_samples, n_permutations)
    average_data[dataset + '_adaptivesplit_train_scores'] = mean_window(data['adaptivesplit_train_scores'], n_samples, n_permutations)

    # get averaged test scores;
    average_data[dataset + '_pareto_test_scores'] = mean_window(data['pareto_test_scores'], n_samples, n_permutations)
    average_data[dataset + '_adaptivesplit_test_scores'] = mean_window(data['adaptivesplit_test_scores'], n_samples, n_permutations)

mean_pareto_train_scores = average_data[[col for col in average_data.columns.values if col.endswith('_pareto_train_scores')]].mean(axis=1)
mean_adaptivesplit_train_scores = average_data[[col for col in average_data.columns.values if col.endswith('_adaptivesplit_train_scores')]].mean(axis=1)

mean_pareto_test_scores = average_data[[col for col in average_data.columns.values if col.endswith('_pareto_test_scores')]].mean(axis=1)
mean_adaptivesplit_test_scores = average_data[[col for col in average_data.columns.values if col.endswith('_adaptivesplit_test_scores')]].mean(axis=1)

# plot train scores;
plt.figure()
plt.plot(np.arange(1, n_samples +1), mean_pareto_train_scores)
plt.plot(np.arange(1, n_samples +1), mean_adaptivesplit_train_scores)
plt.title('Training Scores')
plt.xlabel('Sample size (a.u.)')
plt.ylabel('Scores')
plt.savefig('./all_datasets_mean_train.png')
plt.close()

# plot test scores;
plt.figure()
plt.plot(np.arange(1, n_samples +1), mean_pareto_test_scores)
plt.plot(np.arange(1, n_samples +1), mean_adaptivesplit_test_scores)
plt.title('Test Scores')
plt.xlabel('Sample size (a.u.)')
plt.ylabel('Scores')
plt.savefig('./all_datasets_mean_test.png')
plt.close()
