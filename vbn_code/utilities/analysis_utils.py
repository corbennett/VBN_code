import scipy.signal
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal.windows import exponential
from scipy.ndimage.filters import convolve1d
from statsmodels.stats.multitest import multipletests
import scipy.stats
from numba import njit

@njit
def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins

@njit
def makePSTH_numba(spikes, startTimes, windowDur, binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins

def make_neuron_time_trials_tensor(unit_ids, spike_times, stim_start_time, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a tensor with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        unit_ids: unit_id, i.e. index from units table (same form as session.units table)
        spike_times: spike times corresponding to each unit (spike_times column from units table)
        stim_start_time: the time the stimulus started for each trial
        time_before: seconds to take before each start_time in the stim_table
        trial_duration: total time in seconds to take for each trial
        bin_size: bin_size in seconds used to bin spike counts 
    OUTPUTS:
        unit_tensor: tensor storing spike counts. The value in [i,j,k] 
            is the spike count for neuron i at time bin j in the kth trial.
    '''
    neuron_number = len(unit_ids)
    trial_number = len(stim_start_time)
    unit_tensor = np.zeros((neuron_number, int(trial_duration/bin_size), trial_number))
    
    for iu,unit_id in enumerate(unit_ids):
        unit_spike_times = spike_times[unit_id]
        for tt, trial_stim_start in enumerate(stim_start_time):
            unit_tensor[iu, :, tt] = makePSTH(unit_spike_times, 
                                                [trial_stim_start-time_before], 
                                                trial_duration, 
                                                binSize=bin_size)[0]
    return unit_tensor



def make_data_array(unit_ids, spike_times, stim_start_time, time_before_flash = 0.5, trial_duration = 2, bin_size = 0.001):
    '''
    
    '''

    # Make tensor (3-D matrix [units,time,trials])
    trial_tensor = make_neuron_time_trials_tensor(unit_ids, spike_times, stim_start_time, 
                                                  time_before_flash, trial_duration, 
                                                  bin_size)
    # make xarray data array
    trial_da = xr.DataArray(trial_tensor, dims=("unit_id", "time", "trials"), 
                               coords={
                                   "unit_id": unit_ids,
                                   "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                                   "trials": stim_start_time.index.values
                                   })
    return trial_da


def region_psth(area, units, session, starttime, duration):
    
    area_units = units[units['structure_acronym'].str.contains(area)]
    
    pop_spike_times = []
    for u, _ in area_units.iterrows():
        times = session.spike_times[u]
        pop_spike_times.extend(times)

    pop_spike_times = np.sort(np.array(pop_spike_times))
    
    poppsth, time = makePSTH(pop_spike_times, [starttime], duration, 0.01)
    time = time + starttime

    return poppsth/len(area_units), time


def exponential_convolve(response_vector, tau=1, symmetrical=False):
    
    center = 0 if not symmetrical else None
    exp_filter = exponential(10*tau, center=center, tau=tau, sym=symmetrical)
    exp_filter = exp_filter/exp_filter.sum()
    filtered = convolve1d(response_vector, exp_filter[::-1])
    
    return filtered



def add_block_id_to_trials_table(trials):
    
    first_no_reward, last_no_reward = trials[trials['no_reward_epoch']].index.values[[0, -1]]
    trials['behavior_block'] = 0
    trials.loc[first_no_reward:last_no_reward, 'behavior_block'] = 1
    trials.loc[last_no_reward:, 'behavior_block'] = 2
    
    return trials
    

def blockwise_hit_rates(trials):
    
    hit_rates = []
    for block in [0,1,2]:
        
        bt = trials[trials['behavior_block']==block]
        
        num_go = len(bt[(bt['go']) &
                        (~bt['aborted'])&
                        (~bt['auto_rewarded'])])
        
        num_hits = np.sum(bt['hit'])
        
        hit_rates.append(num_hits/num_go)

    return hit_rates
        

def pass_block_criterion(trials, on_criterion=0.4, off_factor = 2):
    
    block_hit_rates = blockwise_hit_rates(trials)
    on_passes = block_hit_rates[0] > on_criterion and block_hit_rates[2] > on_criterion
    off_passes = block_hit_rates[1] < min([block_hit_rates[0], block_hit_rates[2]])/off_factor
    
    return on_passes and off_passes


def get_aligned_trials(trials, alignment_trial, trials_before=10, trials_after=10):
    
    go_trials = trials[trials['go']]
    
    
    go_trial_inds = go_trials.index.get_level_values(0)
    go_trial_inds = go_trial_inds[go_trial_inds<alignment_trial][-trials_before:]
    go_before = go_trials.loc[go_trial_inds]
    
    go_trial_inds = go_trials.index.get_level_values(0)
    go_trial_inds = go_trial_inds[go_trial_inds>=alignment_trial][:trials_after]
    go_after = go_trials.loc[go_trial_inds]
    
    return pd.concat([go_before, go_after])


def get_reward_off_start(trials):
    
    return trials[trials['behavior_block']==1].iloc[0].name


def get_reward_off_end(trials):
    
    return trials[(trials['behavior_block']==2)&(trials['auto_rewarded'])].iloc[0].name


def parse_lick_times(lick_time_str):
    
    lick_times = lick_time_str.replace('[', '').replace(']', '').replace('\n', '').split(' ')
    lick_times = [float(l) for l in lick_times if l!='']
    return lick_times


def get_response_latency(trial):
        
    lick_times = parse_lick_times(trial['lick_times'])
    change_time = trial['change_time']

    if np.isnan(change_time):
        return np.nan

    licks_after_change = [l for l in lick_times if l>change_time]

    if len(licks_after_change)==0:
        return np.nan

    return licks_after_change[0]-change_time
    

def multiple_comparisons(pvalues):
    if isinstance(pvalues, dict):
        old_values = list(pvalues.values())
    else:
        old_values = pvalues
    
    reject, corrected_pvalues, _, _ = multipletests(old_values, alpha=0.05, method='fdr_bh')

    if isinstance(pvalues, dict):
        return {list(pvalues.keys())[i]: corrected_pvalues[i] for i in range(len(pvalues))}

    return reject, corrected_pvalues


def comparison_matrix(*values, test_func=scipy.stats.wilcoxon):

    pvalue_matrix = np.full((len(values), len(values)), np.nan)
    for ind1, vals1 in enumerate(values):
        for ind2, vals2 in enumerate(values):
            if ind1==ind2:
                continue
            
            p = test_func(vals1, vals2, nan_policy='omit')
            pvalue_matrix[ind1, ind2] = p.pvalue
    
    diag_mask = np.ones(pvalue_matrix.shape, dtype=bool)
    diag_mask[np.diag_indices_from(diag_mask)] = False
    corrected_pvals = multipletests(pvalue_matrix[np.where(diag_mask)], method='fdr_bh')

    pvalue_matrix[np.where(diag_mask)] = corrected_pvals[1]
    sig_matrix = pvalue_matrix<0.05


    return pvalue_matrix, sig_matrix

import scipy.optimize
def fitCurve(func,x,y,initGuess=None, bounds=None):
    if bounds is None:
        fit = scipy.optimize.curve_fit(func,x,y,p0=initGuess, maxfev=100000)[0]
    else:
        fit = scipy.optimize.curve_fit(func,x,y,p0=initGuess,bounds=bounds,maxfev=100000)[0]
    return fit

def calcLogisticDistrib(x,a,b,m,s):
    # a: amplitude, b: offset, m: x at 50% max y, s: scale
    return a * (1 / (1 + np.exp(-(x - m) / s))) + b

def calc_gompertz(x, a, b, c, d):
    return d + (a-d)*np.exp(-np.exp(-b*(x-c)))

def invert_gompertz(y, xs, a, b, c, d):

    vals = np.array([calc_gompertz(x, a, b, c, d) for x in xs])
    yind = np.where(vals<=y)[0]
    if len(yind)==0:
        return np.nan, np.nan
    else:
        return xs[yind[-1]], vals[yind[-1]]

def find_midpoint_raw(x, y):
    y = np.array(y)
    maxd = max(y)
    mind = min(y)
    midpoint = (maxd-mind)/2 + mind

    after_ind = np.where(y>midpoint)[0][0]
    before_ind = np.where(y[:after_ind]<midpoint)[0][-1]

    xbefore = x[before_ind]
    xafter = x[after_ind]

    ybefore = y[before_ind]
    yafter  = y[after_ind]

    slope = (yafter-ybefore)/(xafter-xbefore)

    rise_to_midpoint = midpoint-ybefore
    run_to_midpoint = rise_to_midpoint/slope

    return xbefore+run_to_midpoint, midpoint
