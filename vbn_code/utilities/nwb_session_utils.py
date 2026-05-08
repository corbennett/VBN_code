import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures
import tqdm
import pdb
from analysis_utils import makePSTH_numba, exponential_convolve
from scipy.interpolate import interp1d
from functools import partial

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache


cache_dir = '/data/'

cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(
            cache_dir=cache_dir, use_static_cache=True)


def get_session(session_id):

    return cache.get_ecephys_session(session_id)


def find_running_acceleration_deceleration_times(session, stimulus_block=5):
    running = session.running_speed
    running.loc[running['speed']<0, 'speed'] = 0
    
    # Calculate rolling means for the speed column
    rolling_mean_before = running['speed'].rolling(window=30).mean().shift(1)
    rolling_mean_after = running['speed'].rolling(window=30).mean().shift(-29)

    # Find indices where the acceleration conditions are met
    condition = (rolling_mean_before < 1) & (rolling_mean_after > 5)
    indices = np.where(condition)[0]

    acceleration_times = []
    for ir in indices:
        if (ir > 30) and (ir < len(running) - 31):
            window_indices = running.iloc[ir-30:ir+30].index.values
            min_index = running.loc[window_indices].idxmin()['speed']
            window_diffs = running.loc[min_index:window_indices[-1]].diff()
            try:
                max_diff_index = window_diffs.idxmax()['speed']
                if running.loc[max_diff_index]['speed'] < 1:
                    last_point_below_threshold = running.loc[max_diff_index]['timestamps']
                else:
                    last_point_below_threshold = np.where(running.loc[min_index:max_diff_index]<1)[0][-1]
                    last_point_below_threshold = running.loc[min_index+last_point_below_threshold]['timestamps']
                if len(acceleration_times) > 0:
                    if last_point_below_threshold - acceleration_times[-1] < 0.5:
                        continue
                acceleration_times.append(last_point_below_threshold)
            except Exception as exc:
                sess_id = session.metadata['ecephys_session_id']
                print(f'{sess_id} generated an exception: {exc}')
                continue

    stims = session.stimulus_presentations
    passive_start = stims[stims['stimulus_block']==stimulus_block]['start_time'].iloc[0]
    passive_end = stims[stims['stimulus_block']==stimulus_block]['end_time'].iloc[-1]

    acceleration_times=np.array(acceleration_times)
    passive_acceleration_times = acceleration_times[(acceleration_times>passive_start)&(acceleration_times<passive_end)]


    # Find indices where the deceleration conditions are met
    condition = (rolling_mean_before > 5) & (rolling_mean_after < 1)
    indices = np.where(condition)[0]

    deceleration_times = []
    for ir in indices:
        if (ir > 30) and (ir < len(running) - 31):
            window_indices = running.iloc[ir-30:ir+30].index.values
            max_index = running.loc[window_indices].idxmax()['speed']
            min_index = running.loc[max_index:window_indices[-1]].idxmin()['speed']
            # window_diffs = running.loc[max_index:window_indices[-1]].diff()
            window_diffs = running.loc[window_indices[0]:min_index].diff()
            min_diff_index = window_diffs.idxmin()['speed']
            max_decel_point = running.loc[min_diff_index]['timestamps']
            # if running.loc[min_diff_index]['speed'] > 5:
            #     last_point_above_threshold = running.loc[min_diff_index]['timestamps']
            # else:
            #     last_point_above_threshold = np.where(running.loc[max_index:min_diff_index]>5)[0][-1]
            #     last_point_above_threshold = running.loc[max_index+last_point_above_threshold]['timestamps']
            if len(deceleration_times) > 0:
                if max_decel_point - deceleration_times[-1] < 0.5:
                    continue
            deceleration_times.append(max_decel_point)

    deceleration_times=np.array(deceleration_times)
    passive_deceleration_times = deceleration_times[(deceleration_times>passive_start)&(deceleration_times<passive_end)]


    return passive_acceleration_times, passive_deceleration_times


def unit_peth(session_id, alignment_func, time_before, time_after, binsize):

    session = cache.get_ecephys_session(session_id)
    alignment_times1, alignment_times2 = alignment_func(session)

    units = session.get_units()
    units = units[units['quality']=='good']

    total_time = time_before+time_after
    time_bins = int(total_time/binsize)

    condition_peths = []
    for alignment_times in [alignment_times1, alignment_times2]:
        if len(alignment_times)<5:
            condition_peths.append(np.full((len(units), time_bins), np.nan))
            unit_ids = units.index.values
            continue

        peths = []
        unit_ids = []
        for unit in units.index.values:

            spike_times = session.spike_times[unit]
            peth, time = makePSTH_numba(spike_times, alignment_times-time_before, 
                                    total_time, binSize=binsize)
            
            peths.append(peth[:time_bins])
            unit_ids.append(unit)
    
        condition_peths.append(peths)

    return np.array(condition_peths[0]), np.array(condition_peths[1]), unit_ids


def unit_averaged_psth_time_aligned(session_list, alignment_time_func = 'running', 
                                    time_before=0.5, time_after=0.5, binsize=0.001, stimulus_block=None):

    if alignment_time_func == 'running':
        if stimulus_block is None:
            stimulus_block = 5
        alignment_time_func = partial(find_running_acceleration_deceleration_times, stimulus_block=stimulus_block)

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=20)        
    future_to_session = {}
    for session in session_list:

        fut = pool.submit(unit_peth,
                            session,
                            alignment_time_func,
                            time_before,
                            time_after,
                            binsize)

        future_to_session[fut] = session

    session_data_1 = []
    session_data_2 = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            session_data_1.append(data[0])
            session_data_2.append(data[1])
            unit_ids.append(data[2])
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return session_data_1, session_data_2, unit_ids


def unit_averaged_psth_time_aligned(session_list, alignment_time_func = 'running', 
                                    time_before=0.5, time_after=0.5, binsize=0.001, stimulus_block=None):

    if alignment_time_func == 'running':
        if stimulus_block is None:
            stimulus_block = 5
        alignment_time_func = partial(find_running_acceleration_deceleration_times, stimulus_block=stimulus_block)

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=20)        
    future_to_session = {}
    for session in session_list:

        fut = pool.submit(unit_peth,
                            session,
                            alignment_time_func,
                            time_before,
                            time_after,
                            binsize)

        future_to_session[fut] = session

    session_data_1 = []
    session_data_2 = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            session_data_1.append(data[0])
            session_data_2.append(data[1])
            unit_ids.append(data[2])
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return session_data_1, session_data_2, unit_ids


def get_session_running_df(session_id):

    session = cache.get_ecephys_session(session_id)
    running = session.running_speed

    return running


def get_running_dfs(session_list):

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=30)        
    future_to_session = {}
    for session in session_list:

        fut = pool.submit(get_session_running_df,
                            session,
                            )

        future_to_session[fut] = session

    running_data = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            running_data.append(data)
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return running_data, session_ids


def resample_df(df, start_time, end_time, samplerate):

    ms_timestep = int(1000/samplerate)
    df_subset = df[(df['timestamps']>start_time) & (df['timestamps']<=end_time)]

    df_subset['timestamps_date'] = pd.to_datetime(df_subset['timestamps'], unit='s', origin='unix')
    df_resampled = df_subset.set_index('timestamps_date').resample(f'{ms_timestep}L').mean().interpolate()

    return df_resampled


def resample_vector(vector, resample_factor):
    
    original_time = np.linspace(0, len(vector) - 1, len(vector))
    interpolator = interp1d(original_time, vector, kind='linear')

    new_time = np.linspace(0, len(vector) - 1, len(vector) * resample_factor)
    
    return interpolator(new_time)


def resample_df_to_times(df, time_column, val_column, new_times):

    timestamps = df[time_column].values*1000
    vals = df[val_column].values
    interpolator = interp1d(timestamps, vals, kind='linear', bounds_error=False)#, fill_value=np.nan)
    new_values = interpolator(new_times)

    return new_values, new_times


def get_kernel_subtracted_firing_rate(session_id, unit_ids, stim_filters):

    stim_table = pd.read_csv('/root/capsule/data/supplemental_tables/master_stim_table_no_filter.csv')
    weights_df = pd.read_pickle('/root/capsule/data/supplemental_tables/weights_df.pkl')

    session = get_session(session_id)
    session_running = session.running_speed

    running_resampled = resample_df(session_running[['timestamps', 'speed']], 0, 10000, 1000)
    running_resampled['timestamps'] *= 1000
    running_resampled['timestamps'] = running_resampled['timestamps'].astype(int)

    session_stim = stim_table[stim_table['session_id']==session_id].reset_index()
    behavior_end_time = int(1000*session_stim[session_stim['active']].iloc[-1]['stop_time'])

    trace_length = 2000

    raw_firing_rates = np.full((len(unit_ids), len(stim_filters), trace_length), np.nan)
    kernel_subtracted_firing_rates = np.full((len(unit_ids), len(stim_filters), trace_length), np.nan)
    stim_running = np.full((len(stim_filters), trace_length), np.nan)
    for iunit, unit_id in enumerate(unit_ids):

        #Get running kernel, resample to 1 ms intervals and convolve with running speed
        running_kernel = weights_df.set_index('unit_id').loc[unit_id]['running_weights']
        running_kernel_resampled = resample_vector(running_kernel, 25)
        running_resampled['kernel_convolution'] = np.convolve(running_resampled['speed'].values, running_kernel_resampled, 'same')/25**2

        #Get unit spikes and make into binary vector
        unit_spike_times = session.spike_times[unit_id]
        unit_spike_vector, _ = np.histogram(unit_spike_times*1000, bins=np.arange(behavior_end_time))
        unit_spike_vector = unit_spike_vector.astype(bool)

        for istim, stim_filter in enumerate(stim_filters):
            chained_query = ' & '.join(stim_filter)
            stims_subset = session_stim.query(chained_query)
            
            raw_fr = []
            sub_fr = []
            running = []
            for ind, stim in stims_subset.iterrows():

                start_time = int(stim['start_time']*1000 - 1000)
                stop_time = int(stim['stop_time']*1000 + 1000)

                trial_fr = exponential_convolve(unit_spike_vector[start_time:stop_time]*1000, 5, symmetrical=True)
                trial_kernel_conv = running_resampled[(running_resampled['timestamps']>=start_time)& \
                    (running_resampled['timestamps']<=stop_time)]['kernel_convolution'].values
                trial_running = running_resampled[(running_resampled['timestamps']>=start_time)& \
                    (running_resampled['timestamps']<=stop_time)]['speed'].values
                
                if np.any([len(x)<trace_length for x in [trial_fr, trial_kernel_conv]]):
                    continue

                raw_fr.append(trial_fr[:trace_length])
                sub_fr.append(trial_fr[:trace_length] - trial_kernel_conv[:trace_length])
                running.append(trial_running[:trace_length])

            if len(raw_fr)>0:
                raw_firing_rates[iunit, istim] = np.nanmean(raw_fr, axis=0)
                kernel_subtracted_firing_rates[iunit, istim] = np.nanmean(sub_fr, axis=0)
                stim_running[istim] = np.nanmean(running, axis=0)

    return raw_firing_rates, kernel_subtracted_firing_rates, stim_running, unit_ids


def get_kernel_subtracted_firing_rate_2(session_id, unit_ids, stim_filters):

    stim_table = pd.read_csv('/root/capsule/data/supplemental_tables/master_stim_table_no_filter.csv')
    weights_df = pd.read_pickle('/root/capsule/data/supplemental_tables/weights_df.pkl')

    session = get_session(session_id)
    session_running = session.running_speed

    session_stim = stim_table[stim_table['session_id']==session_id].reset_index()
    behavior_end_time = int(1000*session_stim[session_stim['active']].iloc[-1]['stop_time'])

    trace_length = 160

    running_start = session_running.iloc[0]['timestamps']*1000
    runvals, runtimes = resample_df_to_times(session_running, 'timestamps', 'speed', np.arange(12.5, behavior_end_time+12.5, 25))
    runvals -= np.nanmean(runvals)
    runvals /= np.nanstd(runvals)

    raw_firing_rates = np.full((len(unit_ids), len(stim_filters), trace_length), np.nan)
    kernel_subtracted_firing_rates = np.full((len(unit_ids), len(stim_filters), trace_length), np.nan)
    kernel_conv = np.full((len(unit_ids), len(stim_filters), trace_length), np.nan)
    stim_running = np.full((len(stim_filters), trace_length), np.nan)
    for iunit, unit_id in enumerate(unit_ids):

        #Get running kernel and convolve with running speed
        running_kernel = weights_df.set_index('unit_id').loc[unit_id]['running_weights']
        run_conv = np.convolve(runvals, running_kernel, 'same')

        #Get unit spikes and make into binary vector
        unit_spike_times = session.spike_times[unit_id]
        unit_spike_vector, _ = np.histogram(unit_spike_times*1000, bins=np.arange(0, behavior_end_time, 25))
        unit_spike_vector *= 40 #make Hz

        for istim, stim_filter in enumerate(stim_filters):
            chained_query = ' & '.join(stim_filter)
            stims_subset = session_stim.query(chained_query)
            
            raw_fr = []
            sub_fr = []
            k_conv = []
            running = []
            for ind, stim in stims_subset.iterrows():

                start_time = int(round((stim['start_time']*1000 - 1000)/25))
                stop_time = start_time + trace_length
                
                trial_fr = unit_spike_vector[start_time:stop_time]
                trial_kernel_conv = run_conv[start_time:stop_time]
                trial_running = runvals[start_time:stop_time]

                if np.any([len(x)<trace_length for x in [trial_fr, trial_kernel_conv]]):
                    continue

                raw_fr.append(trial_fr[:trace_length])
                sub_fr.append(trial_fr[:trace_length] - trial_kernel_conv[:trace_length])
                k_conv.append(trial_kernel_conv[:trace_length])
                running.append(trial_running[:trace_length])

            if len(raw_fr)>0:
                raw_firing_rates[iunit, istim] = np.nanmean(raw_fr, axis=0)
                kernel_subtracted_firing_rates[iunit, istim] = np.nanmean(sub_fr, axis=0)
                stim_running[istim] = np.nanmean(running, axis=0)
                kernel_conv[iunit, istim] = np.nanmean(k_conv,axis=0)

    return raw_firing_rates, kernel_subtracted_firing_rates, stim_running, unit_ids


def unit_kernel_subtracted_peth(session_list, unit_ids, stim_filters):

    units = pd.read_csv('/root/capsule/data/supplemental_tables/master_units_with_responsiveness.csv')

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=40)        
    future_to_session = {}
    for session in session_list:
        session_units = units[units['ecephys_session_id']==session]['unit_id'].values
        session_units_to_use = [u for u in unit_ids if u in session_units]
        if len(session_units_to_use)==0:
            continue
        fut = pool.submit(get_kernel_subtracted_firing_rate_2,
                            session,
                            session_units_to_use,
                            stim_filters)

        future_to_session[fut] = session

    raw_firing_rates = []
    kernel_subtracted_firing_rates = []
    running_traces = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            raw_firing_rates.append(data[0])
            kernel_subtracted_firing_rates.append(data[1])
            running_traces.append(data[2])
            unit_ids.append(data[3])
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return raw_firing_rates, kernel_subtracted_firing_rates, running_traces, unit_ids
