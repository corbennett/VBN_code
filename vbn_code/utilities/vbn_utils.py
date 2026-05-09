from typing import Any
import numpy as np
#from numba import njit
from scipy.stats import mannwhitneyu
from scipy.stats import kstest
import h5py
import tqdm
import pandas as pd
import concurrent.futures
from statsmodels.stats.multitest import multipletests
import scipy.stats
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pdb
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PolyCollection
import decoding_utils as du
from analysis_utils import exponential_convolve

#@njit     
def makePSTH_numba(spikes, startTimes, windowDur, binSize=0.001, convolution_kernel=0.05, avg=True):
    spikes = spikes.flatten()
    startTimes = startTimes - convolution_kernel/2
    windowDur = windowDur + convolution_kernel
    bins = np.arange(0,windowDur+binSize,binSize)
    convkernel = np.ones(int(convolution_kernel/binSize))
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/startTimes.size
    counts = np.convolve(counts, convkernel)/(binSize*convkernel.size)
    return counts[convkernel.size-1:-convkernel.size], bins[:-convkernel.size-1]

#@njit
def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    '''
    Convenience function to compute a peri-stimulus-time histogram
    (see section 7.2.2 here: https://neuronaldynamics.epfl.ch/online/Ch7.S2.html)
    INPUTS:
        spikes: spike times in seconds for one unit
        startTimes: trial start times in seconds; the first spike count 
            bin will be aligned to these times
        windowDur: trial duration in seconds
        binSize: size of spike count bins in seconds
    OUTPUTS:
        Tuple of (PSTH, bins), where:
            PSTH gives the trial-averaged spike rate for 
                each time bin aligned to the start times;
            bins are the bin edges as defined by numpy histogram
    '''
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for start in startTimes:
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins[:-1]

#@njit
def make_time_trials_array(spike_times, start_times, 
                            time_before, trial_duration, 
                            bin_size=0.001):

    num_time_bins = int(trial_duration/bin_size)
    
    # Initialize array
    trial_array = np.zeros((num_time_bins, len(start_times)))
    
    # now loop through trials and make a PSTH for this unit for every trial
    for it, trial_start in enumerate(start_times):
        trial_start = trial_start - time_before
        trial_array[:, it] = makePSTH(spike_times, 
                                        [trial_start], 
                                        trial_duration, 
                                        binSize=bin_size)[0][:num_time_bins]
    
    # Make the time vector that will label the time axis
    time_vector = np.arange(num_time_bins)*bin_size - time_before

    return trial_array, time_vector

def make_neuron_time_trials_array(units, spike_times, stim_table, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a 3D array with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        units: dataframe with unit info (same form as session.units table)
        spike_times: dictionary with spike times for each unit (ie session.spike_times)
        stim_table: dataframe whose indices are trial ids and containing a
            'start_time' column indicating when each trial began
        time_before: seconds to take before each start_time in the stim_table
        trial_duration: total time in seconds to take for each trial
        bin_size: bin_size in seconds used to bin spike counts 
    OUTPUTS:
        unit_array: 3D array storing spike counts. The value in [i,j,k] 
            is the spike count for neuron i at time bin j in the kth trial.
        time_vector: vector storing the trial timestamps for the time bins
    '''
    # Get dimensions of output array
    neuron_number = len(units)
    trial_number = len(stim_table)
    num_time_bins = int(trial_duration/bin_size)
    
    # Initialize array
    unit_array = np.zeros((neuron_number, num_time_bins, trial_number))
    
    # Loop through units and trials and store spike counts for every time bin
    for u_counter, (iu, unit) in enumerate(units.iterrows()):
        
        # grab spike times for this unit
        unit_spike_times = spike_times[iu]
        
        # now loop through trials and make a PSTH for this unit for every trial
        for t_counter, (it, trial) in enumerate(stim_table.iterrows()):
            trial_start = trial.start_time - time_before
            unit_array[u_counter, :, t_counter] = makePSTH(unit_spike_times, 
                                                            [trial_start], 
                                                            trial_duration, 
                                                            binSize=bin_size)[0]
    
    # Make the time vector that will label the time axis
    time_vector = np.arange(num_time_bins)*bin_size - time_before
    
    return unit_array, time_vector


def first_spikes_after_onset(spikes, start_times, duration='', censor_period = 0.0015):
    
    start_times = start_times + censor_period
    start_times = start_times[start_times < spikes.max()]
    
    first_spike_inds = np.searchsorted(spikes, start_times)
    first_spike_times = spikes[first_spike_inds] - start_times

    return first_spike_times + censor_period


def first_spike_jitter(spikes, start_times, duration='', censor_period=0.0015):
    
    first_spike_times = first_spikes_after_onset(spikes, start_times, censor_period)
    
    return np.std(first_spike_times)
    

def first_spike_latency(spikes, start_times, duration='', censor_period=0.0015):
    
    return np.mean(first_spikes_after_onset(spikes, start_times, censor_period))
    

def trial_spike_rates(spikes, start_times, duration):
    
    spike_counts = []
    for start in start_times:
        count = len(spikes[(spikes > start) & (spikes <= start+duration)])
        spike_counts.append(count)
    
    return (np.array(spike_counts)/duration)


def baseline_spike_rates(spikes, baseline_starts, baseline_ends):
    baseline_counts = []
    for bs, be in zip(baseline_starts, baseline_ends):
        count = len(spikes[(spikes > bs) & (spikes <= be)])
        baseline_counts.append(count)
    
    total_baseline_time = np.sum(baseline_ends-baseline_starts)
    baseline_rate = np.sum(baseline_counts)/total_baseline_time
    return baseline_rate


def mean_trial_spike_rate(spikes, start_times, duration, baseline_starts, baseline_ends):
    return np.mean(trial_spike_rates(spikes, start_times, duration))-baseline_spike_rates(spikes, baseline_starts, baseline_ends)


def cv_trial_spike_rate(spikes, start_times, duration, baseline_starts, baseline_ends):
    spike_rates = trial_spike_rates(spikes, start_times, duration)-baseline_spike_rates(spikes, baseline_starts, baseline_ends)
    return np.std(spike_rates)/np.mean(spike_rates)


def csalt(spikes, start_times, baseline_start_times):
    
    first_spikes = first_spikes_after_onset(spikes, start_times)
    baseline_spikes = first_spikes_after_onset(spikes, baseline_start_times)
    
    try:
        p = kstest(first_spikes, baseline_spikes)[1]
    except:
        p = np.nan
    return p


def calcDprime(hits,misses,falseAlarms,correctRejects):
    hitRate = calcHitRate(hits,misses,adjusted=True)
    falseAlarmRate = calcHitRate(falseAlarms,correctRejects,adjusted=True)
    z = [scipy.stats.norm.ppf(r) for r in (hitRate,falseAlarmRate)]
    return z[0]-z[1]


def calcHitRate(hits,misses,adjusted=False):
    n = hits+misses
    if n==0:
        return np.nan
    hitRate = hits/n
    if adjusted:
        if hitRate==0:
            hitRate = 0.5/n
        elif hitRate==1:
            hitRate = 1-0.5/n
    return hitRate


def get_baseline_bins(baseline_starts, baseline_ends, binssize=0.001):
    
    all_binstarts = []
    for bs, be in zip(baseline_starts, baseline_ends):
        
        binstarts = np.arange(bs, be, binssize)
        binstarts = binstarts[binstarts+binssize<be] #make sure you don't go beyond end
        
        all_binstarts = all_binstarts + list(binstarts)
        
    return all_binstarts
    

def count_spikes_in_bin(spikes, binstart, binend):
    
    return len(spikes[(spikes>binstart)&(spikes<=binend)])



def fraction_time_responsive(spikes, start_times, time_before, duration, 
                          baseline_starts, baseline_ends, binsize=0.001):
   
    trial_array, time = make_time_trials_array(spikes, start_times, time_before, duration, binsize)
    
    baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
    baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
    baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs+binsize) for bs in baseline_bins_to_use]
    baseline_bin_rates = np.array(baseline_bin_counts)/binsize

    pvals = []
    above_baseline = []
    for timebin in trial_array:
        p = mannwhitneyu(baseline_bin_rates, timebin)
        pvals.append(p[1])
        above_baseline.append(np.mean(timebin)>np.mean(baseline_bin_rates))
    
    above_baseline = np.array(above_baseline)
    
    num_sig_bins = np.sum((np.array(pvals)<0.01)&above_baseline)
    fraction_sig_bins = num_sig_bins/len(time)
    return fraction_sig_bins    


def fraction_trials_responsive(spikes, start_times, time_before, duration, 
                          baseline_starts, baseline_ends, binsize=0.001):
   
    trial_array, time = make_time_trials_array(spikes, start_times, time_before, duration, binsize)
    
    baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
    baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
    baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs+binsize) for bs in baseline_bins_to_use]
    baseline_bin_rates = np.array(baseline_bin_counts)/binsize

    pvals = []
    above_baseline = []
    for trial in trial_array.T:
        p = mannwhitneyu(baseline_bin_rates, trial)
        pvals.append(p[1])
        above_baseline.append(np.mean(trial)>np.mean(baseline_bin_rates))
    
    above_baseline = np.array(above_baseline)
    
    num_sig_bins = np.sum((np.array(pvals)<0.01)&above_baseline)
    fraction_sig_bins = num_sig_bins/len(time)
    return fraction_sig_bins    


def formatFigure(fig, ax, title=None, xLabel=None, yLabel=None, xTickLabels=None, yTickLabels=None,
                blackBackground=False, saveName=None, no_spines=False, yaxis_side='left', axis_lw=1.5, fontsize=None):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42

    if fontsize is not None:
        mpl.rcParams['font.size'] = fontsize

    fig.set_facecolor('w')
    
    to_hide = 'right' if yaxis_side=='left' else 'left'
    spinesToHide = ['right', 'top', 'left', 'bottom'] if no_spines else [to_hide, 'top']
    for spines in spinesToHide:
        ax.spines[spines].set_visible(False)

    if yaxis_side=='left':
        ax.tick_params(direction='out',top=False,right=False)
    else:
        ax.tick_params(direction='out',top=False,left=False)
    
    if title is not None:
        ax.set_title(title)
    if xLabel is not None:
        ax.set_xlabel(xLabel)
    if yLabel is not None:
        ax.set_ylabel(yLabel)
        
    if blackBackground:
        ax.set_axis_bgcolor('k')
        ax.tick_params(labelcolor='w', color='w')
        ax.xaxis.label.set_color('w')
        ax.yaxis.label.set_color('w')
        for side in ('left','bottom'):
            ax.spines[side].set_color('w')

        fig.set_facecolor('k')
        fig.patch.set_facecolor('k')

    # Set the linewidth of all spines
    for spine in ax.spines.values():
        spine.set_linewidth(axis_lw)  # Set spine linewidth to 2 points

    # Set the linewidth of ticks
    ax.tick_params(width=axis_lw)  # Set tick line width to 2 points

    if saveName is not None:
        fig.savefig(saveName, facecolor=fig.get_facecolor())

def get_baseline_over_cuts(session_id, active_tensor_file, unit_ids, flash_indices, baseline_length = 50, resp_window_length=10):
    
    tensor = h5py.File(active_tensor_file, 'r')
    
    n_time_bins = baseline_length + resp_window_length

    session_tensor = tensor[session_id]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids]    
    
    flash_cut_edges = np.percentile(flash_indices, [20, 40, 60, 80])
    flash_cut_index = np.digitize(flash_indices, flash_cut_edges, right=True)
    cut_baselines = []
    for cut in range(5):
        cut_flash_indices = flash_indices[flash_cut_index==cut]
        unit_resp = np.full((len(unit_indices), n_time_bins), np.nan)
        for ucount, uind in enumerate(unit_indices):
            resp = np.mean(session_tensor['spikes'][uind][cut_flash_indices, :resp_window_length], axis=0)

            if baseline_length>0:
                baseline = np.mean(session_tensor['spikes'][uind][cut_flash_indices-1, -baseline_length:], axis=0)
                unit_resp[ucount] = np.concatenate([baseline, resp])
            else:
                unit_resp[ucount] = resp

        cut_baselines.append(np.nanmean(unit_resp))
    
    return cut_baselines, session_id


def get_matched_change_prechange(session_id, tensor_file, unit_ids, stims, 
                                baseline_length = 50, resp_window_length=750, shared='~is_shared',
                                match_running_speed=False, match_running_col_prefix='active_'):


    hit_stim_filter = ['is_change', 'hit', shared, 'engaged']
    hit_query = ' & '.join(hit_stim_filter)
    image_ids = stims.query(hit_query)['image_name'].unique()

    unit_ids_to_return = None
    change_psths = []
    pre_change_psths = []
    for imid in image_ids:

        im_filter = [f'image_name=="{imid}"']
        change_filter = hit_stim_filter + im_filter
        if match_running_speed:
            change_filter = change_filter + [f'{match_running_col_prefix}change_baseline_running_matched']
        change_query = ' & '.join(change_filter)
        
        change_inds = stims.query(change_query).index.values

        pre_change_filter = ['is_prechange', 'engaged', 'hit'] + im_filter
        if match_running_speed:
            pre_change_filter = pre_change_filter + [f'{match_running_col_prefix}prechange_baseline_running_matched']
        pre_change_query = ' & '.join(pre_change_filter)
        prechange_inds = stims.query(pre_change_query).index.values

        if len(change_inds)>0 and len(prechange_inds)>0:
            change_psth = get_unit_psth_for_session_2(str(session_id), tensor_file, unit_ids, change_inds, baseline_length = 50, resp_window_length=750)
            pre_change_psth = get_unit_psth_for_session_2(str(session_id), tensor_file, unit_ids, prechange_inds, baseline_length = 50, resp_window_length=750)
            change_psths.append(change_psth[0])
            pre_change_psths.append(pre_change_psth[0])
            
            unit_ids_to_return = change_psth[1]
    
    return np.mean(change_psths, axis=0), np.mean(pre_change_psths, axis=0), unit_ids_to_return


def get_stim_filters():

    filters = {}
    filters['nonchange_nolick'] = ['~is_change', '~omitted', '~previous_omitted', 'flashes_since_change>5',
                        'flashes_since_last_lick>1', 'engaged', '~lickbout_for_flash_during_response_window']
    filters['nonchange_lick'] = ['~is_change', '~omitted', '~previous_omitted', 'flashes_since_change>5',
                        'flashes_since_last_lick>1', 'engaged', 'lickbout_for_flash_during_response_window']
    filters['hit'] = ['engaged', 'is_change', 'hit']

    return filters


def get_shared_nonshared_nonchange(session_id, tensor_file, unit_ids, stims, 
                        baseline_length = 50, resp_window_length=750,
                        match_running_speed=False, match_running_col_prefix='active_'):


    nonchange_filter = ['~is_change', '~omitted', '~previous_omitted', 'flashes_since_change>5',
                        'flashes_since_last_lick>1', 'engaged', '~lickbout_for_flash_during_response_window']
    
    nonchange_query = ' & '.join(nonchange_filter)

    shared_filter = nonchange_filter + ['is_shared']
    shared_query = ' & '.join(shared_filter)
    shared_inds = stims.query(shared_query).index.values

    nonshared_filter = nonchange_filter + ['~is_shared']
    nonshared_query = ' & '.join(nonshared_filter)
    nonshared_inds = stims.query(nonshared_query).index.values


    if len(shared_inds)>0 and len(nonshared_inds)>0:
        shared_psths = get_unit_psth_for_session_2(str(session_id), tensor_file, unit_ids, shared_inds, baseline_length = 50, resp_window_length=750)
        nonshared_psths = get_unit_psth_for_session_2(str(session_id), tensor_file, unit_ids, nonshared_inds, baseline_length = 50, resp_window_length=750)
        
        unit_ids_to_return = shared_psths[1]
    
        return shared_psths[0], nonshared_psths[0], unit_ids_to_return
    else:
        return np.full((1,800), np.nan), np.full((1,800), np.nan), [-1]
    

def change_prechange_matched_psth(active_tensor_file, stim_file, session_list, unit_ids, 
                                baseline_length=50, resp_window_length=750, comparison='change_prechange',
                                match_running_speed=False, match_running_col_prefix='active_'):

    func = get_matched_change_prechange if comparison=='change_prechange' else get_shared_nonshared_nonchange
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=None)        
    future_to_session = {}
    for session in session_list:
        stims = stim_file[stim_file['session_id']==int(session)].reset_index()
        fut = pool.submit(func, 
                            session, 
                            active_tensor_file,
                            unit_ids, 
                            stims, 
                            baseline_length = baseline_length, 
                            resp_window_length = resp_window_length,
                            match_running_speed=match_running_speed,
                            match_running_col_prefix=match_running_col_prefix
                            )

        future_to_session[fut] = session

    changes = []
    prechanges = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            changes.append(data[0])
            prechanges.append(data[1])
            unit_ids.append(data[2])
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return changes, prechanges, unit_ids


def change_prechange_matched_unit_response_over_trials(active_tensor_file, stim_file, session_list, 
                                unit_ids, baseline_length=50, resp_window_length=750, 
                                response_slice=slice(20,150), baseline_subtract=True, shared='~is_shared'):

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=None)        
    future_to_session = {}
    for session in session_list:
        stims = stim_file[stim_file['session_id']==int(session)].reset_index()
        fut = pool.submit(get_change_prechange_matched_responses_over_trials, 
                            session, 
                            active_tensor_file,
                            unit_ids, 
                            stims, 
                            baseline_length = baseline_length, 
                            resp_window_length = resp_window_length,
                            response_slice=response_slice,
                            baseline_subtract=baseline_subtract,
                            shared=shared)

        future_to_session[fut] = session

    changes = []
    prechanges = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            changes.append(data[0])
            prechanges.append(data[1])
            unit_ids.append(data[2])
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return changes, prechanges, unit_ids


def get_change_prechange_matched_responses_over_trials(session_id, tensor_file, unit_ids, stims, 
                                baseline_length = 50, resp_window_length=750, baseline_subtract=True,
                                response_slice=slice(0,100), shared = '~is_shared'):
    
    #First compute for nonshared images
    hit_stim_filter = ['is_change', 'hit', shared, 'engaged']
    hit_query = ' & '.join(hit_stim_filter)
    image_ids = stims.query(hit_query)['image_name'].unique()

    unit_ids_to_return = None
    change_resps = []
    prechange_resps = []
    for imid in image_ids:

        im_filter = [f'image_name=="{imid}"']
        change_filter = hit_stim_filter + im_filter
        change_query = ' & '.join(change_filter)
        
        change_inds = stims.query(change_query).index.values

        pre_change_filter = ['is_prechange', 'engaged', 'hit'] + im_filter
        pre_change_query = ' & '.join(pre_change_filter)
        prechange_inds = stims.query(pre_change_query).index.values

        min_inds = np.min([len(change_inds), len(prechange_inds)])

        change_inds = np.random.choice(change_inds, min_inds, replace=False)
        prechange_inds = np.random.choice(prechange_inds, min_inds, replace=False)

        if len(change_inds)>0 and len(prechange_inds)>0:
            change_responses = get_unit_response_in_window(str(session_id), tensor_file, unit_ids, change_inds, 
                                                baseline_length = 50, resp_window_length=750,
                                                baseline_subtract=baseline_subtract, response_slice=response_slice)
            prechange_responses = get_unit_response_in_window(str(session_id), tensor_file, unit_ids, prechange_inds, 
                                                baseline_length = 50, resp_window_length=750,
                                                baseline_subtract=baseline_subtract, response_slice=response_slice)
            
            change_resps.append(change_responses[0])
            prechange_resps.append(prechange_responses[0])
            unit_ids_to_return = change_responses[1]
    
    change_nonshared = np.concatenate(change_resps, axis=1)
    prechange_nonshared = np.concatenate(prechange_resps, axis=1)

    
    return change_nonshared, prechange_nonshared, unit_ids_to_return


def get_unit_response_in_window(session_id, active_tensor_file, unit_ids, flash_indices, 
                                baseline_length = 50, resp_window_length=750, baseline_subtract=True,
                                response_slice=slice(0,100)):

    tensor = h5py.File(active_tensor_file, 'r')
    # stims = pd.read_csv(stim_file)
    # #stims = stim_file[stim_file['session_id']==session_id].reset_index()
    # stims = stims[stims['session_id']==session_id].reset_index()

    # chained_query = ' & '.join(stim_filter)
    # stims_subset = stims.query(chained_query)
    # flash_indices = stims_subset.index.values
    
    nflashes = len(flash_indices)

    session_tensor = tensor[session_id]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids]    
    unit_resp = np.full((len(unit_indices), nflashes), np.nan)
    for ucount, uind in enumerate(unit_indices):
        resp = np.mean(session_tensor['spikes'][uind][flash_indices, response_slice], axis=1)

        if baseline_subtract:
            baseline = np.mean(session_tensor['spikes'][uind][flash_indices-1, -baseline_length:], axis=1)
            unit_resp[ucount] = resp-baseline
        else:
            unit_resp[ucount] = resp
    
    return unit_resp, session_tensor['unitIds'][unit_indices]


def get_time_to_first_spike(session_id, active_tensor_file, unit_ids, flash_indices, baseline_length = 50, resp_window_length=750):
    tensor = h5py.File(active_tensor_file, 'r')

    session_tensor = tensor[session_id]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids]    
    unit_ttfs = []
    for ucount, uind in enumerate(unit_indices):
        resp = session_tensor['spikes'][uind][flash_indices, :resp_window_length]
        first_spike_times = np.argmax(resp==1, axis=1)
        first_spike_times = first_spike_times[first_spike_times>0]

        if len(first_spike_times)>0:
            unit_ttfs.append(np.median(first_spike_times))
        else:
            unit_ttfs.append(np.nan)

    return unit_ttfs, session_tensor['unitIds'][unit_indices]


def get_unit_psth_for_session_2(session_id, active_tensor_file, unit_ids, flash_indices, baseline_length = 50, resp_window_length=750):

    tensor = h5py.File(active_tensor_file, 'r')
    # stims = pd.read_csv(stim_file)
    # #stims = stim_file[stim_file['session_id']==session_id].reset_index()
    # stims = stims[stims['session_id']==session_id].reset_index()

    # chained_query = ' & '.join(stim_filter)
    # stims_subset = stims.query(chained_query)
    # flash_indices = stims_subset.index.values
    
    n_time_bins = baseline_length + resp_window_length

    session_tensor = tensor[session_id]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids]    
    unit_resp = np.full((len(unit_indices), n_time_bins), np.nan)
    for ucount, uind in enumerate(unit_indices):
        resp = np.mean(session_tensor['spikes'][uind][flash_indices, :resp_window_length], axis=0)

        if baseline_length>0:
            baseline = np.mean(session_tensor['spikes'][uind][flash_indices-1, -baseline_length:], axis=0)
            unit_resp[ucount] = np.concatenate([baseline, resp])
        else:
            unit_resp[ucount] = resp
    
    return unit_resp, session_tensor['unitIds'][unit_indices]


def get_unit_psth_for_session_3(session_id, tensor_file, unit_ids, flash_indices, baseline_length = 50, resp_window_length=750,
                alignment_times=None, alignment_before=500, alignment_after=500, shuffle=False):
    
    tensor = h5py.File(tensor_file)

    n_time_bins = baseline_length + resp_window_length if alignment_times is None else alignment_before + alignment_after

    session_tensor = tensor[str(session_id)]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids] 
    #grab time from previous flash if necessary
    if baseline_length > 0:
        from_previous_flash = slice(-baseline_length, None)
    else:
        from_previous_flash = slice(0)
    
    #grab time from next flash if necessary
    if resp_window_length > 750:
        from_next_flash = slice(0, resp_window_length-750)
    else:
        from_next_flash = slice(0)
    
    unit_resp = np.full((len(unit_indices), n_time_bins), np.nan)
    # print(unit_resp.shape)
    # shuffle_unit_resp = np.full_like(unit_resp, np.nan)
    shuffle_unit_resp = np.full((len(unit_indices), 3, n_time_bins), np.nan)
    for ucount, uind in enumerate(unit_indices):
        # resp = np.mean(session_tensor['spikes'][uind][flash_indices, :resp_window_length], axis=0)
        # baseline = np.mean(session_tensor['spikes'][uind][flash_indices-1, -baseline_length:], axis=0)
        # unit_resp[ucount] = np.concatenate([baseline, resp])
        flash_indices = flash_indices[(flash_indices>0)&(flash_indices<len(session_tensor['spikes'][uind])-1)]
        uresp = np.concatenate([session_tensor['spikes'][uind][flash_indices-1, from_previous_flash], 
                               session_tensor['spikes'][uind][flash_indices, :resp_window_length],
                               session_tensor['spikes'][uind][flash_indices+1, from_next_flash]], axis=1)
        if alignment_times is not None:
            aligned_resp = []
            for atind, at in enumerate(alignment_times):
                t = np.round(at).astype(int)
                try:
                    aligned_resp.append(uresp[atind, t+baseline_length-alignment_before:t+baseline_length+alignment_after])
                except:
                    #catch out of bounds errors
                    pass
            resp = np.array([a for a in aligned_resp if len(a)==n_time_bins])
            #pdb.set_trace()
            if shuffle:
                shuffle_responses = []
                for iter in range(1000):
                    shuffle_times = np.random.permutation(alignment_times)
                    shuffle_aligned_resp = []
                    for atind, at in enumerate(shuffle_times):
                        t = np.round(at).astype(int)
                        try:
                            shuffle_aligned_resp.append(uresp[atind, t+baseline_length-alignment_before:t+baseline_length+alignment_after])
                        except:
                            #catch out of bounds errors
                            pass
                    shuffle_aligned_resp = np.array([a for a in shuffle_aligned_resp if len(a)==n_time_bins])
                    shuffle_responses.append(np.mean(shuffle_aligned_resp, axis=0))
        
                shuff_convolved = np.array([exponential_convolve(s, 3, symmetrical=True) for s in shuffle_responses])
                shuffle_unit_resp[ucount, 0] = np.mean(shuff_convolved, axis=0)
                shuffle_unit_resp[ucount, 1] = np.percentile(shuff_convolved, 0.05, axis=0)
                shuffle_unit_resp[ucount, 2] = np.percentile(shuff_convolved, 99.95, axis=0)

        # print(f'{ucount} {np.mean(resp, axis=0).shape}')
        unit_resp[ucount] = np.mean(resp, axis=0)
        
    return unit_resp, shuffle_unit_resp, session_tensor['unitIds'][unit_indices]


def match_conditions_on_column(stims, cond_filters, col_to_match):

    col_values = stims[col_to_match].dropna().unique()
    col_value_counts = []
    for stim_filter in cond_filters:

        chained_query = ' & '.join(stim_filter)
        stims_subset = stims.query(chained_query)

        col_counts = []
        for val in col_values:
            col_counts.append(np.sum(stims_subset[col_to_match]==val))
        col_value_counts.append(col_counts)

    cond_indices_to_keep = []
    col_value_counts = np.array(col_value_counts)
    min_counts = col_value_counts.min(axis=0)
    for stim_filter in cond_filters:
        
        chained_query = ' & '.join(stim_filter)
        stims_subset = stims.query(chained_query)

        indices_to_keep = []
        for val, count in zip(col_values, min_counts):
            bin_subset = stims_subset[stims_subset[col_to_match]==val]
            bin_inds = np.random.choice(bin_subset.index.values, count, replace=False)
            indices_to_keep.append(bin_inds)
        
        indices_to_keep = [val for sub in indices_to_keep for val in sub]
        cond_indices_to_keep.append(indices_to_keep)
    return cond_indices_to_keep


def get_unit_psth_column_matched(session_id, active_tensor_file, unit_ids, stims, stim_filters, col_to_match, 
                                baseline_length = 50, resp_window_length=750, num_iterations=100):

    tensor = h5py.File(active_tensor_file, 'r')
    
    n_time_bins = baseline_length + resp_window_length

    session_tensor = tensor[session_id]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids]    

    cond_indices = match_conditions_on_column(stims, stim_filters, col_to_match)
    iteration_data = np.full((num_iterations, len(cond_indices), len(unit_indices), n_time_bins), np.nan)
    iteration_cond_indices = []
    for iter in range(num_iterations):

        cond_indices = match_conditions_on_column(stims, stim_filters, col_to_match)
        cond_unit_resps = np.full((len(cond_indices), len(unit_indices), n_time_bins), np.nan)
        for icond, cond_inds in enumerate(cond_indices):
            cond_inds = np.array(cond_inds)
            unit_resp = np.full((len(unit_indices), n_time_bins), np.nan)
            if len(cond_inds)>0:
                for ucount, uind in enumerate(unit_indices):
                    resp = np.mean(session_tensor['spikes'][uind][cond_inds, :resp_window_length], axis=0)

                    if baseline_length>0:
                        baseline = np.mean(session_tensor['spikes'][uind][cond_inds-1, -baseline_length:], axis=0)
                        unit_resp[ucount] = np.concatenate([baseline, resp])
                    else:
                        unit_resp[ucount] = resp
            cond_unit_resps[icond] = unit_resp
    
        iteration_data[iter] = cond_unit_resps
        iteration_cond_indices.append(cond_indices)

    return np.nanmean(iteration_data, axis=0), iteration_cond_indices, session_tensor['unitIds'][unit_indices]


def unit_averaged_psth(active_tensor_file, stim_file, session_list, unit_ids, *stim_filter, baseline_length=50, resp_window_length=750,
                        just_get_baselines=False, time_to_first_spike=False):

    func = get_unit_psth_for_session_2
    if just_get_baselines:
        func = get_baseline_over_cuts
    elif time_to_first_spike:
        func = get_time_to_first_spike
    
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=70)        
    future_to_session = {}
    for session in session_list:
        stims = stim_file[stim_file['session_id']==int(session)].reset_index()
        chained_query = ' & '.join(stim_filter)
        stims_subset = stims.query(chained_query)
        flash_indices = stims_subset.index.values
        
        fut = pool.submit(func, 
                            str(session), 
                            active_tensor_file,
                            unit_ids, 
                            flash_indices, 
                            baseline_length = baseline_length, 
                            resp_window_length = resp_window_length)

        future_to_session[fut] = session

    session_data = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            session_data.append(data[0])
            unit_ids.append(data[1])
            session_ids.append(session)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return session_data, unit_ids


def unit_averaged_psth_col_matched(tensor_file, stim_file, session_list, unit_ids, stim_filters, col_to_match, 
                        baseline_length=50, resp_window_length=750, num_iterations=100):

    func = get_unit_psth_column_matched
    
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=45)        
    future_to_session = {}
    for session in session_list:
        stims = stim_file[stim_file['session_id']==int(session)].reset_index()
        stims['binned_to_deciles'] = pd.qcut(stims[col_to_match], q=10, labels=False)
        
        fut = pool.submit(func, 
                            str(session), 
                            tensor_file,
                            unit_ids, 
                            stims,
                            stim_filters,
                            'binned_to_deciles',
                            baseline_length = baseline_length, 
                            resp_window_length = resp_window_length,
                            num_iterations=num_iterations)

        future_to_session[fut] = session

    session_data = []
    stim_indices = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        # try:
        data = future.result()
        session_data.append(data[0])
        stim_indices.append(data[1])
        unit_ids.append(data[2])
        session_ids.append(session)

        # except Exception as exc:
        #     print(f'{session} generated an exception: {exc}')

    return session_data, stim_indices, unit_ids, session_ids

    
def unit_averaged_psth_lick_aligned(active_tensor_file, stim_file, session_list, unit_ids, *stim_filter, baseline_length=50, resp_window_length=750,
                        ):

    func = get_unit_psth_for_session_3
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=None)        
    future_to_session = {}
    for session in session_list:
        stims = stim_file[stim_file['session_id']==int(session)].reset_index()
        chained_query = ' & '.join(stim_filter)
        stims_subset = stims.query(chained_query)
        flash_indices = stims_subset.index.values
        
        lick_times = np.array(stims_subset['lick_time'])
        flash_times = np.array(stims_subset['start_time'])
        relative_lickTimes = 1000*(lick_times - flash_times)

        fut = pool.submit(func, 
                            session, 
                            active_tensor_file,
                            unit_ids, 
                            flash_indices, 
                            baseline_length = baseline_length, 
                            resp_window_length = resp_window_length,
                            alignment_times=relative_lickTimes,
                            alignment_before = 500,
                            alignment_after = 500, 
                            shuffle=True)

        future_to_session[fut] = session

    session_data = []
    shuffle_session_data = []
    unit_ids = []
    session_ids = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        # try:
        data = future.result()
        session_data.append(data[0])
        shuffle_session_data.append(data[1])
        unit_ids.append(data[2])
        session_ids.append(session)

        # except Exception as exc:
        #     print(f'{session} generated an exception: {exc}')

    return session_data, shuffle_session_data, unit_ids

def findResponsiveUnits_overtime(basePsth, respPsth, window_duration=50):
    #hasSpikes = ((basePsth[:,:,baseWin].mean(axis=(1,2)) / 0.001) > 0.1) | ((respPsth[:,:,respWin].mean(axis=(1,2))/0.001)>0.1)
    # print(basePsth.shape)
    num_neurons = basePsth.shape[0]
    time = basePsth.shape[2]

    window_starts = np.arange(0, 250)
    window_ends = window_starts + window_duration
    baseWin = slice(time-window_duration, time)
    base = basePsth[:,:,baseWin].mean(axis=2)
    pvals = np.ones((num_neurons, len(window_starts)))
    for iw, (wstart, wend) in enumerate(zip(window_starts, window_ends)):
        respWin = slice(wstart,wend)
        resp = respPsth[:,:,respWin].mean(axis=2)
        pval = np.array([1 if np.sum(r-b)<=0 else scipy.stats.wilcoxon(b,r)[1] for b,r in zip(base,resp)])
        pvals[:, iw] = pval

    return pvals

def get_nonchange_flashes(stim_table, image_id=None):

    stim_table_filter = ((~stim_table['lickbout_for_flash_during_response_window'])&
                        (stim_table['flashes_since_last_lick']>1)&
                        (~stim_table['is_change'])&
                        (stim_table['engaged'])&
                        (~stim_table['previous_omitted'])&
                        (stim_table['flashes_since_change']>5))
                        
    if image_id is not None:
        filter_table = stim_table[stim_table_filter & (stim_table['image_name']==image_id)]
    else:
        filter_table = stim_table[(stim_table_filter)&(~stim_table['omitted'])]
    
    return filter_table.index.values


def calculate_imagewise_stats(session_id, active_tensor_file, unit_ids,
                            baseline_slice=slice(700, 750), response_slice=slice(20,100)):

    active_tensor = h5py.File(active_tensor_file)    
    session_tensor = active_tensor[str(session_id)]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids]    

    stim_table_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
    stim_table = pd.read_csv(stim_table_file)
    stim_table = stim_table.drop(columns='Unnamed: 0') #drop redundant column

    g_images = ['omitted',
                'im012_r',
                'im036_r',
                'im044_r',
                'im047_r',
                'im078_r',
                'im115_r',
                'im083_r',
                'im111_r']

    h_images = ['omitted',
                'im005_r',
                'im024_r',
                'im034_r',
                'im087_r',
                'im104_r',
                'im114_r',
                'im083_r',
                'im111_r']

    session_stim_table = stim_table[stim_table['session_id']==int(session_id)].reset_index()

    image_set = g_images if '_G_' in session_stim_table['stimulus_name'].iloc[0] else h_images
    data_dict = {u:{} for u in session_tensor['unitIds'][unit_indices]}
    spikes = session_tensor['spikes']
    for image in image_set:
        # print(image)
        filter_stims = get_nonchange_flashes(session_stim_table, image_id=image)
        filter_stims = filter_stims[filter_stims>0]

        for iu, uind in enumerate(unit_indices):
            uresp = spikes[uind, filter_stims, :]
            ubaseline = spikes[uind, filter_stims-1, :]

            mean_resp = np.nanmean(uresp[:, response_slice], axis=0) - np.nanmean(ubaseline[:, baseline_slice])
            uid = session_tensor['unitIds'][uind]
            data_dict[uid][image] = {'peak': np.nanmax(mean_resp), 'mean': np.nanmean(mean_resp)}

    other_image_set = h_images if '_G_' in session_stim_table['stimulus_name'].iloc[0] else g_images
    other_image_set = [im for im in other_image_set if im not in ['omitted', 'im083_r', 'im111_r']]
    for image in other_image_set:
        for iu, uind in enumerate(unit_indices):
            uid = session_tensor['unitIds'][uind]
            data_dict[uid][image] = {'peak': np.nan, 'mean': np.nan}


    active_tensor.close()
    return data_dict

def unit_imagewise_stats(active_tensor_file, session_list, unit_ids, baseline_slice=slice(700, 750), response_slice=slice(20,100)):

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=None)        
    future_to_session = {}
    for session in session_list:

        fut = pool.submit(calculate_imagewise_stats, 
                            session, 
                            active_tensor_file,
                            unit_ids, 
                            baseline_slice = baseline_slice,
                            response_slice = response_slice)

        future_to_session[fut] = session

    session_data = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            session_data.append(data)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return session_data

def calculate_responsiveness_over_time(session_id, active_tensor_file, unit_ids, window_size=40):

    active_tensor = h5py.File(active_tensor_file)    
    session_tensor = active_tensor[str(session_id)]
    unit_indices = [ind for ind, u in enumerate(session_tensor['unitIds'][()]) if u in unit_ids]    

    stim_table_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
    stim_table = pd.read_csv(stim_table_file)
    stim_table = stim_table.drop(columns='Unnamed: 0') #drop redundant column

    g_images = ['omitted',
                'im012_r',
                'im036_r',
                'im044_r',
                'im047_r',
                'im078_r',
                'im115_r',
                'im083_r',
                'im111_r']

    h_images = ['omitted',
                'im005_r',
                'im024_r',
                'im034_r',
                'im087_r',
                'im104_r',
                'im114_r',
                'im083_r',
                'im111_r']

    session_stim_table = stim_table[stim_table['session_id']==int(session_id)].reset_index()
    
    image_set = g_images if '_G_' in session_stim_table['stimulus_name'].iloc[0] else h_images
    data_dict = {'unit_ids':session_tensor['unitIds'][unit_indices]}
    spikes = session_tensor['spikes']
    for image in image_set:
        # print(image)
        filter_stims = get_nonchange_flashes(session_stim_table, image_id=image)
        stim_sp = np.full((len(unit_indices), len(filter_stims), 750), np.nan)
        pre_stim_sp = np.full((len(unit_indices), len(filter_stims), 750), np.nan)
        for iu, uind in enumerate(unit_indices):
            stim_sp[iu] = spikes[uind, filter_stims, :]
            pre_stim_sp[iu] = spikes[uind, filter_stims-1, :]
        
        pos_resp_pval = findResponsiveUnits_overtime(pre_stim_sp, stim_sp, window_duration=window_size)

        data_dict[image] = pos_resp_pval

    active_tensor.close()
    return data_dict

def unit_responsiveness_over_time(active_tensor_file, session_list, unit_ids, window_length=40):

    pool = concurrent.futures.ProcessPoolExecutor(max_workers=20)        
    future_to_session = {}
    for session in session_list:

        fut = pool.submit(calculate_responsiveness_over_time, 
                            session, 
                            active_tensor_file,
                            unit_ids, 
                            window_size = window_length)

        future_to_session[fut] = session

    session_data = []
    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_session), total=len(future_to_session), leave=True):
    #for future in concurrent.futures.as_completed(future_to_session):

        session = future_to_session[future]
        try:
            data = future.result()
            session_data.append(data)

        except Exception as exc:
            print(f'{session} generated an exception: {exc}')

    return session_data

def bin_array(array, dim_to_bin, bin_size):
    
    current_size = array.shape
    new_size = np.copy(current_size)
    new_size[dim_to_bin] = -1
    new_size = np.insert(new_size, dim_to_bin+1, bin_size)

    return np.reshape(array, new_size).mean(dim_to_bin+1)


def cumulative_hist(data, include_nan=False):

    data = np.array(data)

    if include_nan:
        total = len(data)
    else:
        total = np.sum(~np.isnan(data))

    data = data[~np.isnan(data)]
    sorted_unique = np.unique(data)
    
    hist = np.array([np.sum(data<=val) for val in sorted_unique])/total
    return sorted_unique, hist


def mean_sem_plot(data, ax, x=None, color='b', alpha=1, ls='solid', label='', facecolor=None, edgecolor=None):
    '''
    Assumes data should be averaged along axis = 0

    '''
    nonnancount = len([d for d in data if not np.any(np.isnan(d))])
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0)/nonnancount**0.5

    if x is None:
        x = np.arange(len(mean))

    if facecolor is None:
        facecolor = color
    
    ax.plot(x, mean, color=color, alpha=alpha, ls=ls, label=label)
    ax.fill_between(x, mean+sem, mean-sem, color=facecolor, alpha=0.5*alpha, lw=0)
    if edgecolor is not None:
        ax.plot(x, mean+sem, color=edgecolor, alpha=alpha, ls=ls, label=label)
        ax.plot(x, mean-sem, color=edgecolor, alpha=alpha, ls=ls, label=label)

def mean_CI_plot(data, ax, x=None, color='b', alpha=1, ls='solid', label='', facecolor=None, edgecolor=None):
    '''
    Assumes data should be averaged along axis = 0

    '''
    nonnancount = len([d for d in data if not np.any(np.isnan(d))])
    mean = np.nanmean(data, axis=0)
    
    ci_high = np.nanpercentile(data, 97.5, axis=0)
    ci_low = np.nanpercentile(data, 2.5, axis=0)

    if x is None:
        x = np.arange(len(mean))

    if facecolor is None:
        facecolor = color
    
    ax.plot(x, mean, color=color, alpha=alpha, ls=ls, label=label)
    ax.fill_between(x, ci_high, ci_low, color=facecolor, alpha=0.25*alpha, lw=0)
    if edgecolor is not None:
        ax.plot(x, ci_high, color=edgecolor, alpha=alpha, ls=ls, label=label)
        ax.plot(x,ci_low, color=edgecolor, alpha=alpha, ls=ls, label=label)

def norm_novel_modulation_ind(fam_resp, nov_resp):
    fmed = np.nanmean(fam_resp)
    nmed = np.nanmean(nov_resp)
    max_abs_dev = np.max([np.abs(v) for v in [fmed,nmed]])
    return ((nmed-fmed)/max_abs_dev)

class Result:
    def __init__(self, pvalue):
        self.pvalue = pvalue

def hybrid_permutation_test_2(vals1, vals2, num_iterations=10000, func=np.nanmedian, nan_policy='omit'):
    
    vals1 = np.array(vals1)
    vals2 = np.array(vals2)
    
    # 1. Identify indices for the two blocks
    mask_paired = ~np.isnan(vals1) & ~np.isnan(vals2)
    mask_u1 = ~np.isnan(vals1) & np.isnan(vals2)  # Unpaired in Group 1
    mask_u2 = np.isnan(vals1) & ~np.isnan(vals2)  # Unpaired in Group 2
    
    # Extract data blocks
    paired_v1 = vals1[mask_paired]
    paired_v2 = vals2[mask_paired]
    unpaired_v1 = vals1[mask_u1]
    unpaired_v2 = vals2[mask_u2]
    
    # Count of unpaired samples to maintain group sizes
    n_u1 = len(unpaired_v1)
    all_unpaired = np.concatenate([unpaired_v1, unpaired_v2])
    
    # Calculate Observed Statistic
    obs_diff = func(np.concatenate([paired_v1, unpaired_v1])) - \
               func(np.concatenate([paired_v2, unpaired_v2]))
    
    if np.isnan(obs_diff):
        return Result(np.nan)

    null_diffs = []
    
    for _ in range(num_iterations):
        # --- PERMUTE PAIRED BLOCK ---
        # For each pair, randomly decide to swap or keep
        swaps = np.random.random(len(paired_v1)) < 0.5
        v1_p_null = np.where(swaps, paired_v2, paired_v1)
        v2_p_null = np.where(swaps, paired_v1, paired_v2)
        
        # --- PERMUTE UNPAIRED BLOCK ---
        # Shuffle all unpaired values and re-split into original sizes
        shuffled_unpaired = np.random.permutation(all_unpaired)
        v1_u_null = shuffled_unpaired[:n_u1]
        v2_u_null = shuffled_unpaired[n_u1:]
        
        # Combine and compute null statistic
        null_v1 = np.concatenate([v1_p_null, v1_u_null])
        null_v2 = np.concatenate([v2_p_null, v2_u_null])
        
        null_diffs.append(func(null_v1) - func(null_v2))

    null_diffs = np.array(null_diffs)
    
    # Two-tailed P-value calculation (including the observed value)
    # Formula: (count(abs(null) >= abs(obs)) + 1) / (N + 1)
    hits = np.sum(np.abs(null_diffs) >= np.abs(obs_diff))
    p_val = (hits + 1) / (num_iterations + 1)
    
    return Result(p_val)


def permutation_test(vals1, vals2, num_iterations=10000, func=norm_novel_modulation_ind, nan_policy='omit'):
    
    a1f = np.array(vals1[0])
    a1n = np.array(vals1[1])

    a2f = np.array(vals2[0])
    a2n = np.array(vals2[1])

    if nan_policy == 'omit':
        a1f = a1f[~np.isnan(a1f)]
        a1n = a1n[~np.isnan(a1n)]
        a2f = a2f[~np.isnan(a2f)]
        a2n = a2n[~np.isnan(a2n)]

    observed_result = func(a1f, a1n) - func(a2f, a2n)
    fs_combined = np.concatenate([a1f, a2f])
    ns_combined = np.concatenate([a1n, a2n])
    null_results = []
    for iter in range(num_iterations):
        np.random.shuffle(fs_combined)
        np.random.shuffle(ns_combined)
        new_null = func(fs_combined[:len(a1f)], ns_combined[:len(a1n)]) - func(fs_combined[len(a1f):], ns_combined[len(a1n):])
        null_results.append(new_null)

    p = Result((1 + np.sum(np.abs(null_results) >= abs(observed_result))) / (num_iterations + 1))

    return p


def multiple_comparisons(pvalues):
    if isinstance(pvalues, dict):
        old_values = list(pvalues.values())
    else:
        old_values = pvalues
    
    reject, corrected_pvalues, _, _ = multipletests(old_values, alpha=0.05, method='fdr_bh')

    if isinstance(pvalues, dict):
        return {list(pvalues.keys())[i]: corrected_pvalues[i] for i in range(len(pvalues))}

    return reject, corrected_pvalues


def plot_comparison_matrix(*values, test_func=scipy.stats.ranksums, ax=None, labels=None, cmap='bwr', colorbar=False, binarize=False, corrected=True, nan_color='gray', return_matrix=False):
    if corrected: 
        comp_matrix = comparison_matrix(*values, test_func=test_func)[0]
    else:
        comp_matrix = comparison_matrix(*values, test_func=test_func)[2]

    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color=nan_color, alpha=1.0) 

    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.05, vmax=1)

    if ax is None:
        fig, ax = plt.subplots()

    if labels is None:
        labels = np.arange(len(values))

    comp_matrix[np.diag_indices_from(comp_matrix)] = 1

    if binarize:
        # comp_matrix = comp_matrix<=0.05
        sig_inds = list(np.where(comp_matrix<=0.05))
        # sig_inds[0] = comp_matrix.shape[0] - sig_inds[0]
        # sig_inds[1] = comp_matrix.shape[1] - sig_inds[1]


    im = ax.imshow(comp_matrix, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation = 90)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    if binarize:
        ax.plot(sig_inds[1], sig_inds[0], 'ko', mec='w')
    if colorbar:
        plt.colorbar(im, ticks=[0, 0.05, 1])
    
    if return_matrix:
        return comp_matrix


def comparison_matrix(*values, test_func=scipy.stats.ranksums):

    pvalue_matrix = np.full((len(values), len(values)), np.nan)
    for ind1, vals1 in enumerate(values):
        for ind2, vals2 in enumerate(values):
            if ind1==ind2:
                continue
            
            p = test_func(vals1, vals2, nan_policy='omit')
            pvalue_matrix[ind1, ind2] = p.pvalue
    
    lower_triang_mask = np.tril(np.ones(pvalue_matrix.shape, dtype=bool), k=-1)

    # diag_mask = np.ones(pvalue_matrix.shape, dtype=bool)
    # diag_mask[np.diag_indices_from(diag_mask)] = False
    pvals = np.copy(pvalue_matrix[np.where(lower_triang_mask)])
    non_nan_mask = ~np.isnan(pvals)
    corrected_pvals = multipletests(pvals[non_nan_mask], method='fdr_bh')
    pvals[non_nan_mask] = corrected_pvals[1]

    corrected_pvalue_matrix = np.full_like(pvalue_matrix, np.nan)
    corrected_pvalue_matrix[np.where(lower_triang_mask)] = pvals
    corrected_pvalue_matrix[np.where(~lower_triang_mask)] = np.nan
    sig_matrix = corrected_pvalue_matrix<0.05
    pvalue_matrix[np.where(~lower_triang_mask)] = np.nan

    return corrected_pvalue_matrix, sig_matrix, pvalue_matrix


def bootstrap_ci(vals, func, num_iterations = 10000, ci=[2.5,97.5], nan_policy='ignore'):
    
    if nan_policy == 'ignore':
        vals = np.array(vals)
        vals = vals[~np.isnan(vals)]
    
    func_vals = []
    for iteration in range(num_iterations):
        iter_vals = np.random.choice(vals, len(vals), replace=True)
        func_vals.append(func(iter_vals))
    
    func_ci = np.percentile(func_vals, ci)
    return func_ci


def bootstrap_ci_vectorized(vals, func=np.median, num_iterations=10000, ci=(2.5, 97.5),
                            nan_policy='ignore', random_state=None, chunk_size=None):
    
    rng = np.random.default_rng(random_state)

    vals = np.asarray(vals)
    if nan_policy == 'ignore':
        vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)

    n = vals.size
    B = num_iterations

    # If memory allows: one big take
    if chunk_size is None:
        idx = rng.integers(0, n, size=(B, n), endpoint=False)   # (B, n)
        samples = vals[idx]                                     # (B, n)
        boot_stats = func(samples, axis=1)                   # (B,)
        return np.percentile(boot_stats, ci)

    # Memory‑friendly: process in chunks
    out = np.empty(B, dtype=float)
    wrote = 0
    while wrote < B:
        m = min(chunk_size, B - wrote)
        idx = rng.integers(0, n, size=(m, n), endpoint=False)
        samples = vals[idx]
        out[wrote:wrote+m] = func(samples, axis=1)
        wrote += m
    return np.percentile(out, ci)


def annotate_plot_with_sig_comparisons(vals, ax, stats_test_func):
    pvalues = []
    comparisons = []
    for ip1, p1 in enumerate(vals):
        for ip2, p2 in enumerate(vals[ip1+1:]):
            print(f'{ip1} vs {ip2+ip1+1}: {kstest(p1, p2)}')
            pvalues.append(stats_test_func(p1, p2).pvalue)
            comparisons.append((ip1, ip2+ip1+1))

    corrected_pvalues = multipletests(pvalues, method='fdr_bh')[1]
    ax_max = ax.get_ylim()[1]
    y_increment = ax_max/10
    for i, (start, end) in enumerate(comparisons):
        if corrected_pvalues[i]<0.05:
            y = ax_max + y_increment*i
            ax.plot([start, start, end, end], [y, y + y_increment/2, y + y_increment/2, y], lw=1.5, c='k')
            ax.text((start + end) * .5, y+ y_increment/2, f"p = {corrected_pvalues[i]:.1e}", ha='center', va='bottom', color='k')

# Function to copy axis properties and data
def copy_axis(source_ax, target_ax):
    # Copy lines
    for line in source_ax.get_lines():
        target_ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(),
                       color=line.get_color(), linestyle=line.get_linestyle(),
                       marker=line.get_marker(), linewidth=line.get_linewidth())
    
    # Copy fill_between (shading)
    for collection in source_ax.collections:
        if isinstance(collection, PolyCollection):
            new_collection = PolyCollection([collection.get_paths()[0].vertices], 
                                            facecolors=collection.get_facecolors(),
                                            edgecolors=collection.get_edgecolors(),
                                            alpha=collection.get_alpha())
            target_ax.add_collection(new_collection)
    
    # Copy labels and title
    target_ax.set_title(source_ax.get_title())
    target_ax.set_xlabel(source_ax.get_xlabel())
    target_ax.set_ylabel(source_ax.get_ylabel())
    
    # Copy legend
    if source_ax.get_legend() is not None:
        target_ax.legend()
        
    # Copy ticks and tick labels
    target_ax.set_xticks(source_ax.get_xticks())
    target_ax.set_xticklabels(source_ax.get_xticklabels())
    target_ax.set_yticks(source_ax.get_yticks())
    target_ax.set_yticklabels(source_ax.get_yticklabels())

    for patch in source_ax.patches:
        if isinstance(patch, Polygon):
            new_patch = Polygon(patch.xy,
                                facecolor=patch.get_facecolor(),
                                alpha=patch.get_alpha())
            target_ax.add_patch(new_patch)

    # Copy limits
    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())


def get_unit_ids(units, areas, cell_types='all', layers='all', clusters='all', clustering = 'new', 
                experience='all', responsive=False, session_id='all', training_trajectory='all'):

    if clustering == 'old':
        if clusters == 'sensory':
            clusters = np.arange(6)
        elif clusters == 'action':
            clusters = [6,7,9,10,11,12]
    else: 
        if clusters == 'sensory':
            clusters = np.arange(1,6)
        elif clusters == 'action':
            clusters = [6,7,8]
            
    if experience == 'all':
        experience_filter = [True]*len(units)
        if training_trajectory != 'all':
            experience_filter = du.apply_condition_filter(units, 'Familiar', training_trajectory) | du.apply_condition_filter(units, 'Novel', training_trajectory)
    else:
        experience_filter = du.apply_condition_filter(units, experience, training_trajectory)

    if responsive:
        response_filter = units['responds_to_at_least_one_nonchange_image'].values
    else:
        response_filter = [True]*len(units)

    if session_id == 'all':
        session_filter = [True]*len(units)
    else:
        # session_filter = units['ecephys_session_id'].values == session_id
        session_filter = units['ecephys_session_id'].isin(make_iterable(session_id))


    final_filter = np.array([False]*len(units))
    for area in make_iterable(areas):
        for cell_type in make_iterable(cell_types):
            for layer in make_iterable(layers):
                final_filter = final_filter | (du.getUnitsInRegion(units, area, layer=layer, cell_type=cell_type) & \
                                                du.get_units_in_cluster(units, *make_iterable(clusters), clustering=clustering) & \
                                                du.apply_unit_quality_filter(units) & experience_filter & response_filter & session_filter)
    return units.loc[final_filter]['unit_id'].values


from collections.abc import Iterable
def make_iterable(arg):
    if not isinstance(arg, Iterable) or isinstance(arg, str):
        arg = [arg]
    return arg


def make_nov_mod_psth_figure(areas, layers, cell_types, clusters, units_subsample, flash_data,
                            flashes=['prechange', 'change'], state='active', training_trajectory='all', 
                            resp_slice=slice(50,170), display_slice=slice(50, 170), norm=True,
                            plot_pvalues=False, smoothing_kernel_width=3, return_fig = False):
    
    base_slice = slice(0, 50)
    base_sub = lambda x: x - x[:, base_slice].mean(axis=1)[:, None]
    no_base_sub = lambda x: x

    total_splits = np.prod([len(make_iterable(split)) for split in [areas, layers, cell_types]])
    fig = plt.figure(figsize=(8, 4*total_splits))
    numcols = len(flashes)
    
    gs = gridspec.GridSpec(total_splits, numcols, figure=fig)
    axes = []

    colors = ['b', 'r']
    alphas = {'active': 1, 'passive': 0.5}
    for istate, state_cond in enumerate(make_iterable(state)):
        for iflash, flash in enumerate(flashes):
            
            axcol = iflash
            counter = 0
            labels = []
            for ia, area in enumerate(make_iterable(areas)):
                for il, layer in enumerate(make_iterable(layers)):
                    for ic, cell_type in enumerate(make_iterable(cell_types)):
                        
                        axrow = counter
                        if istate==0:
                            ax = fig.add_subplot(gs[axrow, axcol])
                            axes.append(ax)
                        else:
                            axind = counter + total_splits*iflash
                            ax = axes[axind]
                        ns = []
                        exp_data = []
                        for iexp, experience in enumerate(['Familiar', 'Novel']):
                            units_to_use = get_unit_ids(units_subsample, area, layers=layer, 
                                                            cell_types=cell_type, clusters=clusters, 
                                                            experience=experience, training_trajectory=training_trajectory,)

                            if norm:
                                resps_norm, _ = get_norm_pop_response(units_to_use, state, state, flash, flash, norm_slice=resp_slice)
                            else:
                                resps_norm = base_sub(np.array([flash_data[u][state_cond][flash] for u in units_to_use]))
                                if smoothing_kernel_width>0:
                                    resps_norm = np.array([exponential_convolve(r, smoothing_kernel_width, symmetrical=True) for r in resps_norm])


                            mean_sem_plot(resps_norm[:, display_slice]*1000, ax, color=colors[iexp], 
                                                    alpha=alphas[state_cond], label=len(resps_norm))
                            exp_data.append(resps_norm[:, display_slice])
                            ns.append(len(resps_norm))
                        
                        if plot_pvalues:
                            ps = [scipy.stats.ranksums(exp_data[0][:, t], exp_data[1][:,t])[1] for t in range(exp_data[0].shape[1])]
                            sigs, corrected_ps = multiple_comparisons(ps)
                            ax2 = ax.twinx()
                            ax2.plot(corrected_ps<0.01)

                        # if istate==0:
                            # ax.axvspan(20, 100, color='k', alpha=0.2)
                        counter += 1
                        layer_str = '_' + layer if layer != 'all' else ''
                        cell_type_str = '_' + cell_type if cell_type != 'all' else ''
                        labels.append(f'{area}{layer_str}{cell_type_str}')
                        ax.set_title(f'{area}{layer_str}{cell_type_str} \n {flash}')
                        formatFigure(fig, ax)

                        ax.spines['left'].set_bounds(0, 5)
                        ax.set_yticks([0,5])
                        ax.spines['bottom'].set_bounds(20, 100)
                        ax.set_xticks([20,100])

                        ax.legend(frameon=False, handlelength=1,)
                        ax.set_xlabel('Time from stim start (ms)')
                        ax.set_ylabel('Firing rate (Hz)')

    plt.tight_layout()
    if return_fig:
        return fig