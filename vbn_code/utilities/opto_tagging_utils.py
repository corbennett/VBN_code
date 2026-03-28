import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import median_abs_deviation


# ---------------------------------------------------------------------------
# Spike-count helpers
# ---------------------------------------------------------------------------

def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    bins = np.arange(0, windowDur + binSize, binSize)
    counts = np.zeros(bins.size - 1)
    for start in startTimes:
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start + windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd] - start, bins)[0]
    counts = counts / len(startTimes)
    return counts / binSize, bins[:-1]


def make_time_trials_array(spike_times, start_times, time_before, trial_duration,
                           bin_size=0.001):
    num_time_bins = int(trial_duration / bin_size)
    trial_array = np.zeros((num_time_bins, len(start_times)))
    for it, trial_start in enumerate(start_times):
        trial_array[:, it] = makePSTH(spike_times,
                                      [trial_start - time_before],
                                      trial_duration,
                                      binSize=bin_size)[0][:num_time_bins]
    return trial_array, np.arange(num_time_bins) * bin_size - time_before


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------

def get_baseline_bins(baseline_starts, baseline_ends, binssize=0.001):
    all_binstarts = []
    for bs, be in zip(baseline_starts, baseline_ends):
        binstarts = np.arange(bs, be, binssize)
        binstarts = binstarts[binstarts + binssize < be]
        all_binstarts = all_binstarts + list(binstarts)
    return all_binstarts


def count_spikes_in_bin(spikes, binstart, binend):
    return len(spikes[(spikes > binstart) & (spikes <= binend)])


def get_baseline_bin_rates(spikes, baseline_starts, baseline_ends, binsize=0.001):
    baseline_bin_starts = get_baseline_bins(baseline_starts, baseline_ends, binsize)
    baseline_bins_to_use = np.random.choice(baseline_bin_starts, 10000, replace=True)
    baseline_bin_counts = [count_spikes_in_bin(spikes, bs, bs + binsize)
                           for bs in baseline_bins_to_use]
    return np.array(baseline_bin_counts) / binsize


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def first_spikes_after_onset(spikes, start_times, censor_period=0.0015):
    start_times = start_times + censor_period
    start_times = start_times[start_times < spikes.max()]
    first_spike_inds = np.searchsorted(spikes, start_times)
    first_spike_times = spikes[first_spike_inds] - start_times
    return first_spike_times + censor_period


def first_spike_jitter(spikes, start_times, duration='', censor_period=0.0015):
    return median_abs_deviation(first_spikes_after_onset(spikes, start_times, censor_period))


def first_spike_latency(spikes, start_times, duration='', censor_period=0.0015):
    return np.median(first_spikes_after_onset(spikes, start_times, censor_period))


def trial_spike_rates(spikes, start_times, duration):
    spike_counts = [len(spikes[(spikes > start) & (spikes <= start + duration)])
                    for start in start_times]
    return np.array(spike_counts) / duration


def mean_trial_spike_rate(spikes, start_times, duration):
    return np.mean(trial_spike_rates(spikes, start_times, duration))


def fraction_time_responsive(spikes, start_times, time_before, duration,
                             baseline_bin_rates, binsize=0.001):
    trial_array, time = make_time_trials_array(spikes, start_times, time_before,
                                               duration, binsize)
    pvals = []
    above_baseline = []
    for timebin in trial_array:
        p = mannwhitneyu(baseline_bin_rates, timebin)
        pvals.append(p[1])
        above_baseline.append(np.mean(timebin) > np.mean(baseline_bin_rates))
    above_baseline = np.array(above_baseline)
    num_sig_bins = np.sum((np.array(pvals) < 0.01) & above_baseline)
    return num_sig_bins / len(time)


# ---------------------------------------------------------------------------
# Post-processing: rename numeric power levels to low/med/high, compute
# evoked-rate z-score columns
# ---------------------------------------------------------------------------

def _get_column_level(col):
    if 'pulse' in col:
        return col.split('_')[1]
    elif 'raised_cosine' in col:
        return col.split('_')[2]
    return 'nolevel'


def _get_levels_from_metrics_df(metrics_df):
    cols = [c for c in metrics_df.columns
            if 'mean_trial_spike_rate' in c and 'pulse' in c]
    levels = [float(c.split('_')[1]) for c in cols]
    return np.sort(levels)


def rename_levels_in_metrics_df(metrics_df):
    """Rename numeric power-level labels in column names to low/med/high."""
    levels = _get_levels_from_metrics_df(metrics_df)
    level_old_labels = [str(l) for l in levels[[0, -2, -1]]]
    level_new_labels = ['low', 'med', 'high']

    for old, new in zip(level_old_labels, level_new_labels):
        col_labels = [c for c in metrics_df.columns if old == _get_column_level(c)]
        metrics_df = metrics_df.rename(
            columns={c: c.replace(old, new) for c in col_labels})

    metrics_df['levels'] = str(levels)
    return metrics_df


def get_evoked_rates(metric_df):
    """Add mean_evoked_rate and mean_evoked_rate_zscored columns for pulse stimulus."""
    cols = [c for c in metric_df.columns if 'mean_trial_spike_rate' in c]

    for col in [c for c in cols if 'pulse' in c]:
        level = col.split('_')[1]
        prefix = f'pulse_{level}'
        metric_df[f'{prefix}_mean_evoked_rate'] = (
            metric_df[col] - metric_df['pulse_baseline_mean'])
        metric_df[f'{prefix}_mean_evoked_rate_zscored'] = (
            (metric_df[col] - metric_df['pulse_baseline_mean'])
            / metric_df['pulse_baseline_std'].replace(0, float('nan')))

    return metric_df
