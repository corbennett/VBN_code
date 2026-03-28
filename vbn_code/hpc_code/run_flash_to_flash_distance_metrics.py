import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import argparse
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
from scipy.spatial import distance
from scipy.linalg import norm
from utilities import *

import warnings
warnings.filterwarnings('ignore')
save_dir = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/logistic_regression_model'
stimtable_with_flash_metrics = pd.read_csv('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/stimtable_with_flash_metrics.csv')
areas_to_run = ('VISall', 'VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP', 'SCMRN')

def run_logit_model(session, save_dir=save_dir):
    session_id = session.metadata['ecephys_session_id']
    stim = stimtable_with_flash_metrics[stimtable_with_flash_metrics['session_id']==session_id]
    
    curated_table = stim[stim['engaged'] & \
                    stim['no_abnorm'] & \
                    ~stim['grace_period_after_hit'] & \
                    #~stim['omitted'] & \
                    #~stim['previous_omitted'] & \
                    #~(stim['is_change'] | stim['is_sham_change'] | stim['is_prechange']) & \
                    #~(stim['is_change'] | stim['is_sham_change']) & \
                    ~(stim['is_change'])]

    column_to_predict = 'lickbout_for_flash_during_response_window'
    for area in areas_to_run:
        for metrics in metrics_to_run:
            if isinstance(metrics, str):
                metrics = metrics + '_' + area if not 'lick' in metrics else metrics
                metrics_name = metrics
            else:
                metrics = [m+'_'+area if not 'lick' in m else m for m in metrics]
                metrics_name = 'combo'
                for m in metrics_to_run:
                    metrics_name = metrics_name + '__' + m
            
            X = curated_table[metrics].to_numpy().astype(float)
            no_nan_inds = [irow for irow, row in enumerate(X) if not np.any(np.isnan(row))]
            #X_nonan = X[no_nan_inds]
            X_nonan = X[no_nan_inds].reshape(-1,1)

            X_nonan = (X_nonan - X_nonan.mean(axis=0))/X_nonan.std(axis=0)
            y = curated_table[column_to_predict].values
            y_nonan = y[no_nan_inds]
            model = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced')
            res = trainModel(model, X_nonan, y_nonan, 5,)

            session_model_results[session].update({'train_balanced_accuracy' + '_' + metrics_name: np.mean(res['train_balanced_accuracy']),
                                                    'test_balanced_accuracy' + '_' + metrics_name: np.mean(res['test_balanced_accuracy'])})
    


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=int)
    parser.add_argument('--cache_dir', type=str)
    args = parser.parse_args()
    
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=args.cache_dir)

    session = cache.get_ecephys_session(
           ecephys_session_id=args.session_id)
    # call the plotting function
    run_decoding(
        session,
    )