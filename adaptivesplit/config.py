import os
import configparser
import warnings
import numpy as np
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomConfigParser(configparser.ConfigParser):
    def getlist(self, section, option):
        return json.loads(self.get(section, option))


settings = CustomConfigParser(allow_no_value=True)
settings.read(os.path.join(ROOT_DIR, 'settings.conf'))

# Stopping rule settings
_min_training_sample_size_ = settings.getint('STOPPING_RULE', 'min_training_sample_size')
_target_power_ = settings.getfloat('STOPPING_RULE', 'target_power')
_alpha_ = settings.getfloat('STOPPING_RULE', 'alpha')
_min_relevant_score_ = settings.getfloat('STOPPING_RULE', 'min_relevant_score')
_min_validation_sample_size_ = settings.getfloat('STOPPING_RULE', 'min_validation_sample_size')
_window_size_ = settings.getint('STOPPING_RULE', 'window_size')
_step_ = settings.getint('STOPPING_RULE', 'step')

try:
    _max_training_sample_size_ = settings.getint('STOPPING_RULE', 'max_training_sample_size')
except:
    warnings.warn("max_training_sample_size expected as int, is set to np.inf ")
    _max_training_sample_size_ = np.inf

try:
    _min_score_ = settings.getfloat('STOPPING_RULE', 'min_score')
except:
    warnings.warn("min_score expected as float, is set to -np.inf")
    _min_score_ = -np.inf

# ------------------------------------------------
# AdaptiveSplit settings
_cv_ = settings.getint('AdaptiveSplit', 'cv')
_bootstrap_samples_ = settings.getint('AdaptiveSplit', 'bootstrap_samples')
_power_bootstrap_samples_ = settings.getint('AdaptiveSplit', 'power_bootstrap_samples')
_n_jobs_ = settings.getint('AdaptiveSplit', 'n_jobs')
_scoring_ = settings['AdaptiveSplit']['scoring']
_total_sample_size_ = settings.getint('AdaptiveSplit', 'total_sample_size')