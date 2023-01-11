import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
import numpy as np

# ToDo - Unindented; Done - Indented;
    # 50/50 split;
    # 90/10 split;
    # fix hardcoded alpha, split line 88;
    # fix step to match config file;
    # shuffle to False;
    # try with 20 > permutations;

    #data.pivot( index="random_seed", columns="truncate_sample_size", values="adaptivesplit_test_scores").sem()
#bcw: max sample size to 0.6

def mean_window(arr, n_samples, n_permutations):
    windows = np.array(arr).reshape(n_permutations, n_samples)
    avg = np.mean(windows, axis=0)
    return avg
    
dataset = 'boston_2'    
data = pd.read_csv('./' + dataset + '/results.csv', delimiter=',', encoding='utf-8')

n_samples = 5
n_permutations = 30
sample_sizes = data['truncate_sample_size'][0:n_samples]

# train scores;
pareto_train_scores = mean_window(data['pareto_train_scores'], n_samples, n_permutations)
pareto_train_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="pareto_train_scores").sem()

halfsplit_train_scores = mean_window(data['halfsplit_train_scores'], n_samples, n_permutations)
halfsplit_train_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="halfsplit_train_scores").sem()

split9010_train_scores = mean_window(data['split90-10_train_scores'], n_samples, n_permutations)
split9010_train_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="split90-10_train_scores").sem()

adaptivesplit_train_scores = mean_window(data['adaptivesplit_train_scores'], n_samples, n_permutations)
adaptivesplit_train_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="adaptivesplit_train_scores").sem()

# test scores; 
pareto_test_scores = mean_window(data['pareto_test_scores'], n_samples, n_permutations)
pareto_test_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="pareto_test_scores").sem()

halfsplit_test_scores = mean_window(data['halfsplit_test_scores'], n_samples, n_permutations)
halfsplit_test_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="halfsplit_test_scores").sem()

split9010_test_scores = mean_window(data['split90-10_test_scores'], n_samples, n_permutations)
split9010_test_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="split90-10_test_scores").sem()

adaptivesplit_test_scores = mean_window(data['adaptivesplit_test_scores'], n_samples, n_permutations)
adaptivesplit_test_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="adaptivesplit_test_scores").sem()

# p-values;
pareto_p_values = mean_window(data['pareto_p_values'], n_samples, n_permutations)
pareto_pval_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="pareto_p_values").sem()

halfsplit_p_values = mean_window(data['halfsplit_p_values'], n_samples, n_permutations)
halfsplit_pval_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="halfsplit_p_values").sem()

split9010_p_values = mean_window(data['split90-10_p_values'], n_samples, n_permutations)
split9010_pval_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="split90-10_p_values").sem()

adaptivesplit_p_values = mean_window(data['adaptivesplit_p_values'], n_samples, n_permutations)
adaptivesplit_pval_se = data.pivot( index="random_seed", columns="truncate_sample_size", values="adaptivesplit_p_values").sem()


# plot train scores;
plt.figure()

plt.plot(sample_sizes, pareto_train_scores, 'b')
plt.fill_between(sample_sizes, pareto_train_scores-(pareto_train_se/2), pareto_train_scores+(pareto_train_se/2), edgecolor='b', facecolor='b', alpha=0.1)

plt.plot(sample_sizes, halfsplit_train_scores, 'r')
plt.fill_between(sample_sizes, halfsplit_train_scores-(halfsplit_train_se/2), halfsplit_train_scores+(halfsplit_train_se/2), edgecolor='r', facecolor='r', alpha=0.1)

plt.plot(sample_sizes, split9010_train_scores, 'g')
plt.fill_between(sample_sizes, split9010_train_scores-(split9010_train_se/2), split9010_train_scores+(split9010_train_se/2), edgecolor='g', facecolor='g', alpha=0.1)

plt.plot(sample_sizes, adaptivesplit_train_scores, 'y')
plt.fill_between(sample_sizes, adaptivesplit_train_scores-(adaptivesplit_train_se/2), adaptivesplit_train_scores+(adaptivesplit_train_se/2), edgecolor='y', facecolor='y', alpha=0.1)

plt.title('Training Scores')
plt.xlabel('Sample Size')
plt.ylabel('Scores')
plt.legend(['Pareto Split', 'Half Split', '90/10 Split', 'AdaptiveSplit'])
#plt.savefig('./' + dataset + '/train.png')
#plt.close()
plt.show()

# plot test scores;
plt.figure()

plt.plot(sample_sizes, pareto_test_scores, 'b')
plt.fill_between(sample_sizes, pareto_test_scores-(pareto_test_se/2), pareto_test_scores+(pareto_test_se/2), edgecolor='b', facecolor='b', alpha=0.1)

plt.plot(sample_sizes, halfsplit_test_scores, 'r')
plt.fill_between(sample_sizes, halfsplit_test_scores-(halfsplit_test_se/2), halfsplit_test_scores+(halfsplit_test_se/2), edgecolor='r', facecolor='r', alpha=0.1)

plt.plot(sample_sizes, split9010_test_scores, 'g')
plt.fill_between(sample_sizes, split9010_test_scores-(split9010_test_se/2), split9010_test_scores+(split9010_test_se/2), edgecolor='g', facecolor='g', alpha=0.1)

plt.plot(sample_sizes, adaptivesplit_test_scores, 'y')
plt.fill_between(sample_sizes, adaptivesplit_test_scores-(adaptivesplit_test_se/2), adaptivesplit_test_scores+(adaptivesplit_test_se/2), edgecolor='y', facecolor='y', alpha=0.1)

plt.title('Test Scores')
plt.xlabel('Sample Size')
plt.ylabel('Scores')
plt.legend(['Pareto Split', 'Half Split', '90/10 Split', 'AdaptiveSplit'])
#plt.savefig('./' + dataset + '/test.png')
#plt.close()
plt.show()

# plot p-values;
plt.figure()

plt.plot(sample_sizes, pareto_p_values, 'b')
plt.fill_between(sample_sizes, pareto_p_values-(pareto_pval_se/2), pareto_p_values+(pareto_pval_se/2), edgecolor='b', facecolor='b', alpha=0.1)

plt.plot(sample_sizes, halfsplit_p_values, 'r')
plt.fill_between(sample_sizes, halfsplit_p_values-(halfsplit_pval_se/2), halfsplit_p_values+(halfsplit_pval_se/2), edgecolor='r', facecolor='r', alpha=0.1)

plt.plot(sample_sizes, split9010_p_values, 'g')
plt.fill_between(sample_sizes, split9010_p_values-(split9010_pval_se/2), split9010_p_values+(split9010_pval_se/2), edgecolor='g', facecolor='g', alpha=0.1)

plt.plot(sample_sizes, adaptivesplit_p_values, 'y')
plt.fill_between(sample_sizes, adaptivesplit_p_values-(adaptivesplit_pval_se/2), adaptivesplit_p_values+(adaptivesplit_pval_se/2), edgecolor='y', facecolor='y', alpha=0.1)

plt.title('P-Values')
plt.xlabel('Sample Size')
plt.ylabel('Scores')
plt.legend(['Pareto Split', 'Half Split', '90/10 Split', 'AdaptiveSplit'])
#plt.savefig('./' + dataset + '/pvals.png')
#plt.close()
plt.show()
