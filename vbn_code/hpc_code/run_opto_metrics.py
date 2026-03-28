import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import argparse
from vbn_utils import *
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

save_dir = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/opto_metrics"

def get_opto_metrics(session, save_location=save_dir):
    
    sess_id = session.metadata['ecephys_session_id']
    spike_times = session.spike_times
    units = session.get_units()

    start_times = session.optotagging_table.groupby(by=['stimulus_name', 'level'])['start_time'].apply(list)
    conditions = start_times.index.get_level_values(0)
    levels = start_times.index.get_level_values(1)

    censor_period = 0.005
    baseline_starts = session.optotagging_table['stop_time'].values[:-1] + censor_period
    baseline_ends = session.optotagging_table['start_time'].values[1:] - censor_period
    min_gap = np.min(baseline_ends - baseline_starts)
    baseline_starts = baseline_starts + min_gap/2
    total_baseline_time = np.sum(baseline_ends-baseline_starts)

    min_iti = session.optotagging_table['start_time'].diff().min()
    csalt_baseline_starts = session.optotagging_table['start_time'] + min_iti-0.3
    #csalt_baseline_starts = np.random.choice(baseline_starts, 200, replace=False)
    #csalt_baseline_starts = np.copy(baseline_starts)

    start_times = session.optotagging_table.groupby(by=['stimulus_name', 'level'])['start_time'].apply(list)
    conditions = start_times.index.get_level_values(0)
    levels = start_times.index.get_level_values(1)

    metrics = [mean_trial_spike_rate, cv_trial_spike_rate, csalt, 
            first_spike_jitter, first_spike_latency, 
            fraction_time_responsive, fraction_trials_responsive]
    metric_dict = {cond+'_'+str(lev)+'_'+met.__name__:[] for cond in conditions for lev in levels for met in metrics}
    metric_dict.update({'uid':[], 'pulse_baseline_mean' :[], 'raised_cosine_baseline_mean': [],
                    'pulse_baseline_std': [], 'raised_cosine_baseline_std': []})
    censor_period=0.0015
    durations = {'pulse': 0.010-2*censor_period, 'raised_cosine': 1-2*censor_period}
    binsizes = {'pulse': 0.001, 'raised_cosine': 0.01}

    for counter, (u, _) in enumerate(units.iterrows()):
        #print(f'running {u}, {counter+1} of {len(cortical_units)}')
        spikes = spike_times[u]
        metric_dict['uid'].append(u)
        baseline_mean_rates = {c: get_baseline_bin_rates(spikes, baseline_starts, baseline_ends, binsize=durations[c]) for c in conditions}
        metric_dict['pulse_baseline_mean'].append(np.mean(baseline_mean_rates['pulse']))
        metric_dict['pulse_baseline_std'].append(np.std(baseline_mean_rates['pulse']))
        metric_dict['raised_cosine_baseline_mean'].append(np.mean(baseline_mean_rates['raised_cosine']))
        metric_dict['raised_cosine_baseline_std'].append(np.std(baseline_mean_rates['raised_cosine']))
        
        baseline_bin_rates = {c: get_baseline_bin_rates(spikes, baseline_starts, baseline_ends, binsize=binsizes[c]) for c in conditions}

        for starts, condition, level in zip(start_times, conditions, levels):
            duration = durations[condition]
            starts=np.array(starts)
            
            for m in metrics:
                name = m.__name__
                col_name = condition+'_'+str(level)+'_'+name
                if 'csalt' == name:
                    metric_dict[col_name].append(m(spikes, starts, csalt_baseline_starts))
                elif 'fraction' in name:
                    metric_dict[col_name].append(m(spikes, starts, 0, duration, 
                                                baseline_bin_rates[condition], binsizes[condition]))
                else:
                    metric_dict[col_name].append(m(spikes, starts+censor_period, duration))
            

    metric_df = pd.DataFrame(metric_dict)
    
    metric_df.to_csv(os.path.join(save_location, str(sess_id) + '_opto_metrics.csv'))

if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=int)
    parser.add_argument('--cache_dir', type=str)
    args = parser.parse_args()
    
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=args.cache_dir)
    print('getting session')
    session = cache.get_ecephys_session(
           ecephys_session_id=args.session_id)
    # call the plotting function
    get_opto_metrics(
        session,
    )