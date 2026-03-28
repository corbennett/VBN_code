import numpy as np
from numba import njit
from scipy.stats import mannwhitneyu
from scipy.stats import kstest
import scipy.stats
import h5py
import tqdm
import concurrent.futures
from vbn_utils import makePSTH
import pandas as pd
import decoding_utils as du
import os
from functools import partial

from allensdk.brain_observatory.ecephys.behavior_ecephys_session import BehaviorEcephysSession

sessions = pd.read_csv("/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/VBN_four_day_experiment_nwbs/sessions.csv")
all_probes = pd.read_csv("/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/VBN_four_day_experiment_nwbs/probes.csv")

def get_session_by_id(session_id):
    session_path = sessions[sessions['exp_id']==session_id]['nwb_path'].values[0]
    session = BehaviorEcephysSession.from_nwb_path(session_path)
    return session

def psth_from_nwb(session_id, unit_ids, stim_filter, win_before=1, win_after=1, bin_size=0.01):
    session_path = sessions[sessions['exp_id']==session_id]['nwb_path'].values[0]
    session = BehaviorEcephysSession.from_nwb_path(session_path)

    #stims = session.stimulus_presentations #pd.read_csv("/Volumes/programs/mindscope/workgroups/np-exp/1380255150_729286_20240715/SDK_outputs/VBN_stimulus_table.csv")    
    stims = du.annotate_stimulus_table(session)
    stims = stims.query(' & '.join(stim_filter))

    align_times = stims['start_time'].values - win_before
    win_duration = win_before + win_after

    session_unit_ids = [u for u in unit_ids if u in session.spike_times.keys()]
    unitpsths = []
    bins = None
    for unit_id in session_unit_ids:
        spikes = session.spike_times[unit_id]
        psth, bins = makePSTH(spikes, align_times, win_duration, bin_size)
        unitpsths.append(psth)
    
    return unitpsths, bins


def psth_from_nwb_cut(session_id, unit_ids, stim_filter, win_before=1, win_after=1, bin_size=0.01, num_cuts = 5):
    session_path = sessions[sessions['exp_id']==session_id]['nwb_path'].values[0]
    session = BehaviorEcephysSession.from_nwb_path(session_path)  
    stims = du.annotate_stimulus_table(session)
    
    #stims = stims.query(' & '.join(stim_filter))

    win_duration = win_before + win_after
    session_unit_ids = [u for u in unit_ids if u in session.spike_times.keys()]

    epoch_psths = []
    for epoch in ['active', '~active']:
        epoch_psths.append([])
        stims_to_use = stims.query(epoch).reset_index()
        stims_to_use['cut_index'] = pd.qcut(stims_to_use.index, num_cuts, labels=False)
        stims_to_use = stims_to_use.query(' & '.join(stim_filter))
        for cut in range(num_cuts):
            align_times = stims_to_use[stims_to_use['cut_index']==cut]['start_time'].values - win_before
            
            unitpsths = []
            bins = None
            for unit_id in session_unit_ids:
                spikes = session.spike_times[unit_id]
                psth, bins = makePSTH(spikes, align_times, win_duration, bin_size)
                unitpsths.append(psth)
            epoch_psths[-1].append(unitpsths)
    
    return epoch_psths, bins


def unit_averaged_psth_from_nwb(session_list, unit_ids, stim_filter, cut=False, win_before=1, win_after=1, bin_size = 0.01, num_cuts=5):
    
    func = partial(psth_from_nwb_cut, num_cuts=num_cuts) if cut else psth_from_nwb

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=7)        
    future_to_session = {}
    for session_id in session_list:
    
        fut = pool.submit(func, 
                            session_id, 
                            unit_ids, 
                            stim_filter,
                            win_before,
                            win_after,
                            bin_size,
                            )

        future_to_session[fut] = session_id

    session_data = []
    session_ids = []
    bins = None
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session)):

        session = future_to_session[future]
        try:
            data = future.result()
            session_data.append(data[0])
            # unit_ids.append(data[1])
            bins = data[1]
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return session_data, bins

def save_out_units_table_for_session(session_id):
    session_path = sessions[sessions['exp_id']==session_id]['nwb_path'].values[0]
    session = BehaviorEcephysSession.from_nwb_path(session_path)
    units = session.get_units()
    units.reset_index(inplace=True)
    units.rename(columns={'id': 'unit_id'}, inplace=True)
    channels = session.get_channels()
    probes = session.probes
    units = units.merge(channels, left_on='peak_channel_id', right_on='id')
    units = units.merge(probes, left_on='probe_id', right_on='id')

    quality = du.apply_unit_quality_filter(units, no_abnorm=False)
    units['quality'] = quality

    def get_in_ctx(row):
        ctx_start = all_probes.set_index('probe_id').loc[row['probe_id']]['ctx_start']
        ctx_end = all_probes.set_index('probe_id').loc[row['probe_id']]['ctx_end']
        return row['probe_channel_number'] <= ctx_start and row['probe_channel_number'] >= ctx_end
    
    units['in_cortex'] = units.apply(get_in_ctx, axis=1)
    units = units.merge(all_probes[['probe_id', 'area']], on='probe_id', suffixes=('', '_allprobes'))

    units.to_csv(os.path.join("/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/VBN_four_day_experiment_nwbs/unit_csvs",
                              f'{session_id}_units.csv'), index=False)
    return session_id    

def save_out_units_table(session_list):
    
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=6)        
    future_to_session = {}
    for session_id in session_list:
    
        fut = pool.submit(save_out_units_table_for_session, 
                            session_id, 
                            )

        future_to_session[fut] = session_id

    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session)):

        session_id = future_to_session[future]
        try:
            data = future.result()
            print(f'completed session {data}')

        except Exception as exc:
            print(f'{session_id} generated an exception: {exc}')
