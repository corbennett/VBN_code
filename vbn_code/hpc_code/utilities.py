import numpy as np
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt
import json
import scipy.stats


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

def make_neuron_time_trials_array(units, spike_times, stim_table, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a 3D array with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        units: dataframe with unit info (same form as session.units table)
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
    #unit_array = np.zeros((neuron_number, num_time_bins, trial_number))
    unit_array = np.zeros((trial_number, neuron_number, num_time_bins))
    
    # Loop through units and trials and store spike counts for every time bin
    for u_counter, (iu, unit) in enumerate(units.iterrows()):
        
        # grab spike times for this unit
        unit_spike_times = spike_times[iu]
        
        # now loop through trials and make a PSTH for this unit for every trial
        for t_counter, (it, trial) in enumerate(stim_table.iterrows()):
            trial_start = trial.start_time - time_before
            unit_array[t_counter, u_counter, :] = makePSTH(unit_spike_times, 
                                                            [trial_start], 
                                                            trial_duration, 
                                                            binSize=bin_size)[0]
    
    # Make the time vector that will label the time axis
    time_vector = np.arange(num_time_bins)*bin_size - time_before
    
    return unit_array, time_vector


def triggered_average(data, times, time_before=0.5, time_after=0.5, sampling_rate=60):
    mean = []
    expected_samples = int((time_before+time_after)*sampling_rate)
    for t in times:
        trial_data = data[(data['timestamps']>=t-time_before-1)&
                              (data['timestamps']< t+time_after +1)]
        
        x = trial_data.timestamps.values
        y = trial_data.speed.values
        f = interpolate.interp1d(x, y)
        
        trial_running = f(np.linspace(t-time_before, t+time_after, expected_samples))
        
        mean.append(trial_running)
    
    return mean, np.linspace(-time_before, time_after, expected_samples)


def align_lfp(lfp, trial_window, alignment_times, trial_ids = None):
    time_selection = np.concatenate([trial_window + t for t in alignment_times])
    
    if trial_ids is None:
        trial_ids = np.arange(len(alignment_times))
        
    inds = pd.MultiIndex.from_product((trial_ids, trial_window), 
                                      names=('presentation_id', 'time_from_presentation_onset'))

    ds = lfp.sel(time = time_selection, method='nearest').to_dataset(name = 'aligned_lfp')
    ds = ds.assign(time=inds).unstack('time')

    return ds['aligned_lfp']


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def get_first_lick_in_response_window(row, frame_length=1/60.):
    
    licktimes = row['lick_times']
    changetime = row['change_time']
    
    if len(licktimes)==0 or np.isnan(changetime):
        return np.nan
    
    licktimes = licktimes[licktimes>changetime+0.15-frame_length]
    if len(licktimes)==0:
        return np.nan
    
    return np.min(licktimes - changetime)

def get_first_lick(licktimes):
    if len(licktimes)==0:
        fl = np.nan
    else:
        fl = min(licktimes)
    return fl


def get_nonchange_flashes(stim_table, image_id=None):

    stim_table_filter = ((stim_table['lick_for_flash']==False)&
                        (stim_table['flashes_since_last_lick']>=2)&
                        (stim_table['is_change']==False)&
                        (stim_table['reward_rate']>=2)&
                        (stim_table['previous_omitted']==False)&
                        #(~stim_table['omitted'])&
                        (stim_table['flashes_since_change']>5))
                        
    if image_id is not None:
        filter_table = stim_table[stim_table_filter & (stim_table['image_name']==image_id)]
    else:
        filter_table = stim_table[(stim_table_filter)&(~stim_table['omitted'])]
    
    return filter_table.index.values


def get_change_flashes(stim_table, image_id=None):

    stim_table_filter = (
                        (stim_table['is_change']==True)&
                        (stim_table['reward_rate']>=2)
                        )
    if image_id is not None:
        filter_table = stim_table[stim_table_filter & (stim_table['image_name']==image_id)]
    else:
        filter_table = stim_table[stim_table_filter]

    return filter_table.index.values


def get_hit_flashes(stim_table, image_id):

    stim_table_filter = (
                        (stim_table['is_change']==True)&
                        (stim_table['hit'])&
                        (stim_table['reward_rate']>=2)&
                        (stim_table['image_name']==image_id)
                        )

    return stim_table[stim_table_filter].index.values


def get_miss_flashes(stim_table, image_id):

    stim_table_filter = (
                        (stim_table['is_change']==True)&
                        (stim_table['miss'])&
                        (stim_table['reward_rate']>=2)&
                        (stim_table['image_name']==image_id)
                        )

    return stim_table[stim_table_filter].index.values


def findResponsiveUnits_nopeak(basePsth, respPsth, baseWin = slice(500,750), respWin = slice(30,280)):
    #hasSpikes = ((basePsth[:,:,baseWin].mean(axis=(1,2)) / 0.001) > 0.1) | ((respPsth[:,:,respWin].mean(axis=(1,2))/0.001)>0.1)
    
    base = basePsth[:,:,baseWin].mean(axis=1)
    resp = respPsth[:,:,respWin].mean(axis=1)
    peak_evoked = np.max(resp-base.mean(axis=1)[:,None],axis=1)
    # hasPeakResp = peak > 5 * base.std(axis=1)
    
    base = basePsth[:,:,baseWin].mean(axis=2)
    resp = respPsth[:,:,respWin].mean(axis=2)
    pval = np.array([1 if np.sum(r-b)==0 else scipy.stats.wilcoxon(b,r)[1] for b,r in zip(base,resp)])
    
    mean_evoked = resp.mean(axis=1) - base.mean(axis=1)
    positive_modulation = mean_evoked>0

    return pval, positive_modulation, mean_evoked, peak_evoked


def findResponsiveUnits_overtime(basePsth, respPsth, window_duration=50):
    #hasSpikes = ((basePsth[:,:,baseWin].mean(axis=(1,2)) / 0.001) > 0.1) | ((respPsth[:,:,respWin].mean(axis=(1,2))/0.001)>0.1)
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


def save_json(to_save, save_path):
        
    with open(save_path, 'w') as f:
        json.dump(to_save, f, indent=2)