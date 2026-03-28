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

save_dir = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/stim_trials"

def get_stim_trials(session, save_location=save_dir):
    
    sess_id = session.metadata['ecephys_session_id']
    stim = session.stimulus_presentations
    trials = session.trials
    licks = session.licks
    
    col_names = ['hit', 'false_alarm', 'miss', 'correct_reject', 'aborted']
    trials['trial_type'] = trials[col_names].idxmax(axis=1)
    trials['previous_trial_type'] = np.insert(trials.trial_type.values[:-1], 0, 'none')

    lick_times = licks.timestamps.values
    lick_diffs = np.diff(lick_times)
    lick_bout_starts = lick_times[np.insert(lick_diffs, 0, 1) > 0.5]
    lick_bout_starts = lick_bout_starts[lick_bout_starts>stim.start_time[0]]
    lick_bout_ends = lick_times[np.append(lick_diffs, 1) > 0.5]
    lick_bout_ends = lick_bout_ends[lick_bout_ends>=lick_bout_starts[0]]
    time_from_last_bout = lick_bout_starts[1:] - lick_bout_ends[:-1]

    lick_flashes = np.searchsorted(stim.start_time, lick_bout_starts) - 1 
    stim.loc[lick_flashes, 'lick_time'] = lick_bout_starts
    stim.loc[lick_flashes, 'time_from_last_lick_bout'] = np.insert(time_from_last_bout, 0, -1)
    stim['previous_omitted'] = np.insert(stim['omitted'].values[:-1], 0, False)
    stim['lick_for_flash'] = stim.lick_time.notna()

    trial_start_times = trials['start_time'] #get the start times for every behavior trial
    stim_presentation_starts = stim.start_time #and the start times for every stim presentation

    #Assign every stim to a trial based on when it was shown
    stim_trial_assignments = np.searchsorted(trial_start_times, stim_presentation_starts) - 1
    stim['behavior_trial_id'] = stim_trial_assignments

    #Clean up result
    #First don't assign trials to stims that occurred outside the behavior
    stim.loc[stim.start_time>trials.iloc[-1]['stop_time'], 'behavior_trial_id'] = np.nan

    #Second copy the trial assignments from the active block to the passive block
    stim.loc[stim.stimulus_block==5, 'behavior_trial_id'] = stim[stim.active]['behavior_trial_id'].values

    #Now users can merge the two tables with the following line:
    stim_trials = stim.merge(trials, left_on='behavior_trial_id', right_index=True, how='left')

    flashes_since_trial_began = []
    counter = -1
    last_trial_id = stim['behavior_trial_id'].iloc[0]
    for tid in stim['behavior_trial_id'].values:
        if last_trial_id == tid:
            counter+=1
        else: 
            counter = 0
        flashes_since_trial_began.append(counter)
        last_trial_id = tid

    stim_trials['trial_flash'] = flashes_since_trial_began

    #Find flashes since last lick bout start
    flashes_since_last_lick = []
    all_lick_flashes = np.searchsorted(stim.start_time, lick_bout_starts) - 1 
    counter = 0
    for lt in stim_trials.lick_time.values:
        counter +=1
        flashes_since_last_lick.append(counter)
        if ~np.isnan(lt):
            counter = 0

    stim_trials['flashes_since_last_lick_bout_start'] = flashes_since_last_lick

    #Find flashes since last lick (end of last lick bout)
    all_lick_flashes = np.append(np.searchsorted(stim.start_time, lick_times) - 1, -1)
    flashes_since_lick = [np.min(flash - all_lick_flashes[all_lick_flashes<flash]) for flash in stim_trials.index.values]
    stim_trials['flashes_since_last_lick'] = flashes_since_lick
    first_lick_stim = stim_trials['lick_time'].idxmin()
    stim_trials.loc[:first_lick_stim, 'flashes_since_last_lick'] = np.nan #column doesn't make sense before first lick

    #Make column denoting the first lick in each trial
    first_lick_in_trial = stim_trials[['lick_time', 'behavior_trial_id']].groupby('behavior_trial_id').transform('min')
    stim_trials['first_lick_in_trial']  = np.isin(stim_trials['lick_time'], first_lick_in_trial)

    stim_trials = stim_trials.rename(columns={'start_time_x':'start_time', 
                                        'start_time_y': 'trial_start_time', 
                                        'stop_time_x': 'stop_time', 
                                        'stop_time_y': 'trial_stop_time'})
    
    stim_trials.to_csv(os.path.join(save_location, str(sess_id) + '_stim_trials.csv'))

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
    get_stim_trials(
        session,
    )