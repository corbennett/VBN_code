import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import median_abs_deviation
from scipy.stats import kstest
from matplotlib import pyplot as plt
import os
from scipy.signal.windows import exponential
from scipy.ndimage.filters import convolve1d
import decoding_utils as du


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
    start_times = start_times[start_times<spikes.max()]
    
    first_spike_inds = np.searchsorted(spikes, start_times)
    first_spike_times = spikes[first_spike_inds] - start_times

    return first_spike_times + censor_period


def first_spike_jitter(spikes, start_times, duration='', censor_period=0.0015):
    
    first_spike_times = first_spikes_after_onset(spikes, start_times, censor_period)
    
    return median_abs_deviation(first_spike_times)
    

def first_spike_latency(spikes, start_times, duration='', censor_period=0.0015):
    
    return np.median(first_spikes_after_onset(spikes, start_times, censor_period))
    

def trial_spike_rates(spikes, start_times, duration):
    
    spike_counts = []
    for start in start_times:
        count = len(spikes[(spikes>start) & (spikes<=start+duration)])
        spike_counts.append(count)
    
    return (np.array(spike_counts)/duration)


def baseline_spike_rate(spikes, baseline_starts, baseline_ends, binsize=1):
    baseline_counts = []
    for bs, be in zip(baseline_starts, baseline_ends):
        count = len(spikes[(spikes>bs) & (spikes<=be)])
        baseline_counts.append(count)
    
    baseline_rate = np.sum(baseline_counts)/total_baseline_time
    return baseline_rate


def mean_trial_spike_rate(spikes, start_times, duration):
    return np.mean(trial_spike_rates(spikes,start_times,duration))


def cv_trial_spike_rate(spikes, start_times, duration):
    spike_rates = trial_spike_rates(spikes,start_times,duration)
    return np.std(spike_rates)/np.mean(spike_rates)


def csalt(spikes, start_times, baseline_start_times):
    
    first_spikes = first_spikes_after_onset(spikes, start_times)
    baseline_spikes = first_spikes_after_onset(spikes, baseline_start_times)
    
    try:
        #p = wilcoxon(first_spikes, baseline_spikes[:len(first_spikes)])[1]
        p = kstest(first_spikes, baseline_spikes)[1]
    except:
        p = np.nan
    return p


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
                          baseline_bin_rates, binsize=0.001):
   
    trial_array, time = make_time_trials_array(spikes, start_times, time_before, duration, binsize)
    
#     baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
#     baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
#     baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs+binsize) for bs in baseline_bins_to_use]
#     baseline_bin_rates = np.array(baseline_bin_counts)/binsize

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
                          baseline_bin_rates, binsize=0.001):
   
    trial_array, time = make_time_trials_array(spikes, start_times, time_before, duration, binsize)
    
#     baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
#     baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
#     baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs+binsize) for bs in baseline_bins_to_use]
#     baseline_bin_rates = np.array(baseline_bin_counts)/binsize

    pvals = []
    above_baseline = []
    for trial in trial_array.T:
        p = mannwhitneyu(baseline_bin_rates, trial)
        pvals.append(p[1])
        above_baseline.append(np.mean(trial)>np.mean(baseline_bin_rates))
    
    above_baseline = np.array(above_baseline)
    
    num_sig_bins = np.sum((np.array(pvals)<0.01)&above_baseline)
    fraction_sig_bins = num_sig_bins/len(pvals)
    return fraction_sig_bins    


def get_baseline_bin_rates(spikes, baseline_starts, baseline_ends, binsize=0.001):
    
    baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
    baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
    baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs+binsize) for bs in baseline_bins_to_use]
    baseline_bin_rates = np.array(baseline_bin_counts)/binsize
    
    return baseline_bin_rates

def plot_raster(ax, spikes, start_times, duration=0.03):
    
    raster = []
    for start in start_times:
        r = spikes[(spikes>=start)&(spikes<=start+duration)]
        if len(r)>0:
            raster.append(r-start)
#         else:
#             raster.append([np.nan])
    
    ax.eventplot(raster)
    ax.set_xlim([0, duration])

def get_unit_psth_for_session_2(session_id, tensor, unit_indices, flash_indices, alignment_times=None, baseline_length = 50, resp_window_len=750):
    
    #tensor = h5py.File(active_tensor_file)

    n_time_bins = baseline_length + resp_window_len if alignment_times is None else 1000

    session_tensor = tensor[str(session_id)]
    #grab time from previous flash if necessary
    if baseline_length > 0:
        from_previous_flash = slice(-baseline_length, None)
    else:
        from_previous_flash = slice(0)
    
    #grab time from next flash if necessary
    if resp_window_len > 750:
        from_next_flash = slice(0, resp_window_len-750)
    else:
        from_next_flash = slice(0)
    
    unit_resp = np.full((len(unit_indices), n_time_bins), np.nan)
    for ucount, uind in enumerate(unit_indices):
        # resp = np.mean(session_tensor['spikes'][uind][flash_indices, :resp_window_len], axis=0)
        # baseline = np.mean(session_tensor['spikes'][uind][flash_indices-1, -baseline_length:], axis=0)
        # unit_resp[ucount] = np.concatenate([baseline, resp])
        flash_indices = flash_indices[(flash_indices>0)&(flash_indices<len(session_tensor['spikes'][uind])-1)]
        resp = np.concatenate([session_tensor['spikes'][uind][flash_indices-1, from_previous_flash], 
                               session_tensor['spikes'][uind][flash_indices, :resp_window_len],
                               session_tensor['spikes'][uind][flash_indices+1, from_next_flash]], axis=1)
        if alignment_times is not None:
            aligned_resp = []
            for atind, at in enumerate(alignment_times):
                t = np.round(at).astype(int)
                try:
                    aligned_resp.append(resp[atind, t+baseline_length-500:t+baseline_length+500])
                except:
                    #catch out of bounds errors
                    pass
            resp = np.array(aligned_resp)
        unit_resp[ucount] = np.mean(resp, axis=0)
        
    return unit_resp


def plot_raster2(trial_spike_times, flash_indices, win_before, win_after, annotation_times=None, orderby=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if orderby is not None:
        order = np.argsort(orderby)
        flash_indices = flash_indices[order]
        annotation_times = annotation_times[order]

    #grab time from previous flash if necessary
    if win_before > 0:
        from_previous_flash = slice(-win_before, None)
    else:
        from_previous_flash = slice(0)
    
    #grab time from next flash if necessary
    if win_after > 750:
        from_next_flash = slice(0, win_after-750)
    else:
        from_next_flash = slice(0)

    if annotation_times is not None:
        for anno_ind, anno in enumerate(annotation_times):
            if anno:
                ax.plot(anno+win_before, anno_ind, 'r.', alpha=0.25)


    #build the raster
    for flash_num, flash in enumerate(flash_indices):
        if flash == 0 or flash == len(trial_spike_times)-1:
            continue 
        
        trial_spikes = np.concatenate((trial_spike_times[flash-1, from_previous_flash], 
                                       trial_spike_times[flash],
                                       trial_spike_times[flash+1, from_next_flash]))
        
        raster_positions = np.where(trial_spikes)[0]
        if len(raster_positions)>0:
            ax.eventplot(raster_positions, lineoffsets=flash_num, colors='k')
    ax.set_xticks(np.arange(0, win_before+win_after, 150))
    ax.set_xticklabels(np.arange(-win_before, win_after, 150), rotation=90)
    ax.axvline(win_before, color='k', ls='dotted')

def exponential_convolve(response_vector, tau=1, symmetrical=False):
    
    center = 0 if not symmetrical else None
    exp_filter = exponential(10*tau, center=center, tau=tau, sym=symmetrical)
    exp_filter = exp_filter/exp_filter.sum()
    filtered = convolve1d(response_vector, exp_filter[::-1])
    
    return filtered   


def make_unit_figure(unit_id, unit_table, stim_table, active_tensor, passive_tensor, save=True):
    cluster_col_to_use = 'cluster_labels_new'
    uinfo = unit_table.set_index('unit_id').loc[unit_id]
    if np.isnan(uinfo[cluster_col_to_use]):
        return
    
    unit_session = unit_table.set_index('unit_id').loc[unit_id]['ecephys_session_id']
    session_stim = stim_table[stim_table['session_id']==unit_session]
    session_tensor = active_tensor[str(unit_session)]

    flashTimes,image_ids,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = du.getBehavData(session_stim)
    relative_lickTimes = 1000*(lickTimes - flashTimes)

    unit_index = np.where(session_tensor['unitIds'][()]==unit_id)[0][0]
    unit_spikes = session_tensor['spikes'][unit_index]
    
    fig, axes = plt.subplots(2,3, figsize=(25, 6))
    fig.suptitle(f'Unit {unit_id}, {uinfo["structure_acronym"]}')
    fig.patch.set_facecolor('white')

    
    #plot raster
    for filter, ax in zip([changeFlashes, nonChangeFlashes], [axes[0][0], axes[1][0]]):
        mid_color = 'gold' if filter is changeFlashes else 'gray'
        ax.axvspan(0, 250, color='gray', alpha=0.3)
        ax.axvspan(750, 1000, color=mid_color, alpha=0.5)
        ax.axvspan(1500, 1750, color='gray', alpha=0.3)
        
        stim_filter = filter&lick
        plot_raster2(unit_spikes, 
                np.where(stim_filter)[0], 750, 1500, 
                relative_lickTimes[stim_filter],
                orderby=relative_lickTimes[stim_filter], ax=ax)
        ax.set_title('Change lick' if filter is changeFlashes else 'Nonchange lick')

    
    axes[0][0].set_xticks([])

    
    #plot change licks vs nonchangelicks psth
    axes[0][1].axvspan(-750, -500, color='gray', alpha=0.3)
    axes[0][1].axvspan(0, 250, color='gold', alpha=0.5)
    axes[0][1].axvspan(750, 1000, color='gray', alpha=0.3)

    for state_tensor, colors, state in zip([active_tensor, passive_tensor], [['k', 'gray'], ['darkviolet', 'violet']], ['active', 'passive']):
        for filter, color, condition in zip([changeFlashes, nonChangeFlashes], colors, ['change lick', 'nonchange lick']):
            lick_filter = lick & filter
            lick_psth = get_unit_psth_for_session_2(unit_session, state_tensor, 
                                                    [unit_index,], np.where(lick_filter)[0],
                                                    baseline_length=750, resp_window_len=1500)
            lick_psth = exponential_convolve(lick_psth, 10, symmetrical=True)
            time = np.arange(-750, 1500)
            axes[0][1].plot(time, np.mean(lick_psth, axis=0), color=color, label=f'{state} {condition}')
            axes[0][1].axvline(0, color='k', ls='dotted')
            
            time = np.arange(-500,500)
            lick_psth = get_unit_psth_for_session_2(unit_session, state_tensor, 
                                                    [unit_index,], np.where(lick_filter)[0],
                                                    baseline_length=750, resp_window_len=1500, alignment_times=relative_lickTimes[lick_filter])
            lick_psth = exponential_convolve(lick_psth, 10, symmetrical=True)
            axes[0][2].plot(time, np.mean(lick_psth, axis=0), color=color, label=f'{state} {condition}')
            axes[0][2].axvline(0, color='k', ls='dotted')

    #plot non change licks vs non change no licks psth
    axes[1][1].axvspan(-750, -500, color='gray', alpha=0.3)
    axes[1][1].axvspan(0, 250, color='gray', alpha=0.5)
    axes[1][1].axvspan(750, 1000, color='gray', alpha=0.3)
    for filter, tensor, color, condition in zip([nonChangeFlashes & lick, nonChangeFlashes & ~lick, nonChangeFlashes & ~lick], 
                                        [active_tensor, active_tensor, passive_tensor],
                                        ['k','gray','violet'], 
                                        ['active nonchange lick', 'active nonchange no lick', 'passive nonchange no lick']):
        lick_psth = get_unit_psth_for_session_2(unit_session, tensor, 
                                                [unit_index,], np.where(filter)[0],
                                                baseline_length=750, resp_window_len=1500)
        lick_psth = exponential_convolve(lick_psth, 10, symmetrical=True)
        time = np.arange(-750, 1500)
        axes[1][1].plot(time, np.mean(lick_psth, axis=0), color=color, label=f'{condition}')
        axes[1][1].axvline(0, color='k', ls='dotted')
    
    axes[1][1].legend()

    axes[0][1].legend()
    axes[1][0].set_xlabel('Time from flash (ms)')
    axes[1][1].set_xlabel('Time from flash (ms)')

    axes[0][2].set_xlabel('Time from lick (ms)')
    axes[1,2].axis('off')
    if save:
        save_dir = '/Volumes/programs/mindscope/workgroups/np-behavior/VBN_spike_rasters'
        if np.isnan(uinfo[cluster_col_to_use]):
            cluster = 'unlabeled'
        else:
            cluster = str(uinfo[cluster_col_to_use])
        
        u_save_dir = os.path.join(save_dir, cluster + '_new')
        if not os.path.exists(u_save_dir):
            os.mkdir(u_save_dir)

        structure = uinfo['structure_acronym']
        fig.savefig(os.path.join(u_save_dir, f'unit_{unit_id}_cluster_{cluster}_structure_{structure}.png'))
        plt.close('all')