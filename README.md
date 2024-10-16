External Validation of Machine Learning Models - Registered Models and Adaptive Sample Splitting
==========================

Giuseppe Gallitto, Robert Englert, Balint Kincses, Raviteja Kotikalapudi, Jialin Li, Kevin Hoffschlag, Ulrike Bingel, Tamas Spisak

### Purpose of this Repository

This repository provides the reproducible analysis code for our manuscript and a copy of its [preprint version](https://pni-lab.github.io/AdaptiveSplitAnalysis/).

**Content Description**:

The `adaptivesplit` package in this repository includes additional code to support the analyses presented in our 
manuscript.

Modifications are located in the "adaptivesplit/sklearn_interface/split.py" file, starting at line 265.

This version of the package is designed to reproduce the analyses shown in the manuscript. For general use, please 
refer to the maintained version of the package available at [this repository](https://github.com/pni-lab/adaptivesplit).

**Analysis Scripts and Outputs**:

To reproduce the analyses, refer to the code in the "scripts" folder. Each dataset has a dedicated analysis script 
(e.g., "analysis_abide.py", "analysis_hcp.py", etc.).

Due to the size and licensing restrictions of some datasets, this repository only includes data for the preprocessed 
IXI dataset (under the CC BY-SA 3.0 license). This data is available in the "data_in" folder. You can also find our 
repository containing the same preprocessed IXI data [here](https://zenodo.org/records/11635168). 
Information and data for the original IXI dataset can be found following [this link](https://brain-development.org/ixi-dataset/).

Running an analysis script will generate plots with the learning curve and power curve for each permutation of the 
data (100 total permutations) at each "sample size budget" (depending on the dataset). The plots also show the optimal
splitting point (training / external validation) found by the adaptivesplit algorithm.

A CSV file (results.csv) containing the results for each permutation will also be created. All output generated by the 
analysis scripts is saved in the "./tests" folder.

**Access to the Results for Each Dataset**:

You can access the output data generated by all the analysis scripts (for all datasets) in the "./data_out" folder. 
Each "results.csv" file created in the respective dataset directory is used as input for the "./notebooks/results.ipynb"
Python notebook. This notebook demonstrates the procedure to generate the plots with the main results, shown in 
**Figure 3** of the manuscript and the plots for the **discovery phase**, **additional splits** and the demonstration of the **stopping rule's
performance mode** that can be found in the supplementary materials.
