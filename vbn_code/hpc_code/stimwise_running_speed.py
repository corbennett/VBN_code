import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import argparse
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
from brain_observatory_utilities.datasets.electrophysiology.\
    receptive_field_mapping import ReceptiveFieldMapping_VBN

from utilities import *

save_dir = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/stimwise_running'

def compute_stimwise_running(session, save_location=save_dir):
    
    sess_id = session.metadata['ecephys_session_id']

    running_speed = session.running_speed
    # master_stim = pd.read_csv(os.path.join('/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables', 'master_stim_table_no_filter.csv'))
    # session_stim = master_stim[master_stim['session_id']==sess_id]
    session_stim = session.stimulus_presentations


    stim_speeds = []
    baseline_speeds = []
    stim_indices = []
    for irow, row in session_stim.iterrows():
        start_t = row['start_time']
        end_t = row['end_time']
        stim_running = running_speed[(running_speed['timestamps']>=start_t)&
                                (running_speed['timestamps']< end_t)]
        if len(stim_running)>0:
            stim_speeds.append(np.nanmean(stim_running.speed.values))
        else:
            stim_speeds.append(np.nan)

        baseline_running = running_speed[(running_speed['timestamps']>=start_t-0.2)&
                                (running_speed['timestamps']< start_t)]
        if len(baseline_running)>0:
            baseline_speeds.append(np.nanmean(baseline_running.speed.values))
        else:
            baseline_speeds.append(np.nan)
        
        stim_indices.append(irow)
    
    df = pd.DataFrame({'stim_index': stim_indices,
                        'stim_speed': stim_speeds,
                        'baseline_speed': baseline_speeds,
                        'session_id': sess_id,
                        'start_time': session_stim['start_time'].values,
                        'block': session_stim['stimulus_block'].values})
    
    df.to_csv(os.path.join(save_dir, str(sess_id) + '_stimwise_running_with_baseline.csv'))


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
    compute_stimwise_running(
        session,
    )