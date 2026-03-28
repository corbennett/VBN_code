"""
Utility functions shared across VBN figure notebooks.
"""

import numpy as np
import scipy.stats
import scipy.optimize
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Population RF fitting
# ---------------------------------------------------------------------------

def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = -(np.sin(2*theta)) / (4*sigma_x**2) + (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    g = offset + amplitude * np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def bootstrap_CI(data, n_samples=1000, alpha=0.05):
    ntrials = data.shape[0]
    ntime = data.shape[1]
    lower_bounds = np.zeros(ntime)
    upper_bounds = np.zeros(ntime)
    for time in range(ntime):
        sample_means = np.zeros(n_samples)
        for i in range(n_samples):
            sample = np.random.choice(np.arange(ntrials), ntrials, replace=True)
            sample_means[i] = data[sample, time].mean()
        lower_bounds[time] = np.percentile(sample_means, 100*alpha/2)
        upper_bounds[time] = np.percentile(sample_means, 100*(1-alpha/2))
    return lower_bounds, upper_bounds


def normalize(x, chance, norm_slice=slice(0, 75), sub='chance'):
    x = x[:, norm_slice]
    if sub == 'min':
        x = x - x.min(axis=1)[:, None]
    else:
        x = x - chance
    x = x / x.max(axis=1)[:, None]
    return x


def upsample(x, factor):
    new_time = np.arange(0, len(x), 1/factor)
    interpolator = interp1d(np.arange(len(x)), x, kind='cubic', bounds_error=False)
    upsampled = interpolator(new_time)
    return upsampled


def plot_decoding_results(decoding_dict, label, area, cluster, unitSampleSize, nPsuedoFlashes, nUnitSamples, time,
                          condition='active', bootstrap_iterations=1000, color=None, ax=None, norm=False,
                          norm_slice=slice(0, 75), return_mean=False, ls='-', upsample_factor=1,
                          plotlabel=None, **plotkwargs):
    from matplotlib import pyplot as plt
    from ccf_utils import get_area_color

    if condition in decoding_dict:
        results = decoding_dict[condition][label][area][cluster][unitSampleSize][nPsuedoFlashes][nUnitSamples]
    else:
        results = decoding_dict[label][area][cluster][unitSampleSize][nPsuedoFlashes][nUnitSamples]

    results = np.array([upsample(r, upsample_factor) for r in results])
    time = np.linspace(time[0], time[-1], len(time)*upsample_factor)

    if len(results) == 0:
        return
    if norm:
        if label in ['lick', 'change']:
            chance = 0.5
        elif label == 'image':
            chance = 1/8
        elif label == 'reaction_time':
            chance = 0.2
        else:
            chance = 0
        results = normalize(results, chance, norm_slice)
    mean = results.mean(axis=0)
    error = results.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots()

    if color is None:
        import pandas as pd
        structure_tree = pd.read_csv("/Volumes/programs/mindscope/workgroups/np-behavior/ccf_structure_tree_2017.csv")
        color = get_area_color(area, structure_tree)

    ax.plot(time, mean, color=color, ls=ls, label=plotlabel, **plotkwargs)
    ax.fill_between(time, mean-error, mean+error, color=color, alpha=0.3, lw=0)

    if return_mean:
        return mean


def plot_facemap_decoding(data, time, ax=None, norm=False, norm_slice=slice(0, 75),
                          bootstrap_iterations=100, color=None, return_mean=False):
    from matplotlib import pyplot as plt
    from ccf_utils import get_area_color

    data = np.array(data)
    if norm:
        data = normalize(data, 0.5, norm_slice)
    mean = data.mean(axis=0)
    p2pt5, p97pt5 = bootstrap_CI(data, n_samples=bootstrap_iterations)
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(time, mean, color=color)
    ax.fill_between(time, p2pt5, p97pt5, color=color, alpha=0.3, lw=0)

    if return_mean:
        return mean


def get_latency2(results, chance, time, norm_slice=slice(0, 75), threshold=0.5, upsample_factor=1, sub='min', norm=True):
    if norm:
        results_to_use = normalize(results, chance, norm_slice, sub=sub)
    else:
        results_to_use = results[:, norm_slice]
    latencies = np.zeros(results_to_use.shape[0])
    time_upsampled = np.linspace(time[0], time[-1], len(time)*upsample_factor)
    for ires, res in enumerate(results_to_use):
        if np.max(results[ires][norm_slice]) < chance+0.1:
            latencies[ires] = np.nan
            continue
        res = upsample(res, upsample_factor)
        l = np.where(res <= threshold)[0]
        if len(l) == 0:
            latencies[ires] = np.nan
        else:
            latencies[ires] = time_upsampled[l[-1]]

    error_bounds = [np.nanstd(latencies), np.nanstd(latencies)]
    return time_upsampled, latencies, np.nanmean(latencies), error_bounds[0], error_bounds[1]


def set_spine_linewidth(ax, linewidth=2):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def fitCurve(func, x, y, initGuess=None, bounds=None):
    if bounds is None:
        fit = scipy.optimize.curve_fit(func, x, y, p0=initGuess, maxfev=100000)[0]
    else:
        fit = scipy.optimize.curve_fit(func, x, y, p0=initGuess, bounds=bounds, maxfev=100000)[0]
    return fit


def calc_gompertz(x, a, b, c, d):
    return d + (a-d)*np.exp(-np.exp(-b*(x-c)))


def invert_gompertz(y, xs, a, b, c, d):
    vals = np.array([calc_gompertz(x, a, b, c, d) for x in xs])
    yind = np.where(vals <= y)[0]
    if len(yind) == 0:
        return np.nan, np.nan
    else:
        return xs[yind[-1]], vals[yind[-1]]


def get_sigmoidfit_midpoint(x, y, ythresh=None):
    """
    Fit a Gompertz sigmoid to data and return the midpoint.
    x and y should be 1D arrays. Returns (xmid, ymid, fit_data).
    For notebooks that pass decode_windows as x, call with decode_windows explicitly.
    """
    try:
        ymidpoint = (y.max() - y.min())/2 + y.min()
        shift_estimate = np.where(y > ymidpoint)[0][0]
        fit = fitCurve(calc_gompertz, x, y, [y.max(), 0.1, shift_estimate, y.min()])
        fit_data = [calc_gompertz(xi, *fit) for xi in x]
        rval = np.corrcoef(fit_data, y)[0, 1]
        if rval < 0.8:
            xmid = ymid = np.nan
        elif ythresh is not None and (np.max(y) < ythresh):
            xmid = ymid = np.nan
        else:
            mid_range_y = (fit[0]-fit[-1])/2 + fit[-1]
            xmid, ymid = invert_gompertz(mid_range_y, np.arange(0, x.max(), 0.1), *fit)

        if len(fit_data) == len(x):
            return xmid, ymid, fit_data
        else:
            return xmid, ymid
    except Exception as e:
        print(e)
        if ythresh is not None:
            return np.nan, np.nan, np.full(len(x), np.nan)
        return np.nan, np.nan


def get_sigmoidfit_midpoint_2val(x, y):
    """2-value wrapper around get_sigmoidfit_midpoint for callers that only need xmid, ymid."""
    result = get_sigmoidfit_midpoint(x, y)
    return result[0], result[1]


def get_latency_sigmoid_fit(results, time, norm_slice=slice(0, 12), threshold=0.5):
    fits = np.array([get_sigmoidfit_midpoint(time[norm_slice], r[norm_slice], ythresh=threshold) for r in results])
    latencies = fits[:, 0]
    error_bounds = [np.nanstd(latencies), np.nanstd(latencies)]
    return latencies, np.nanmean(latencies), error_bounds[0], error_bounds[1]


def find_midpoint_raw(x, y):
    y = np.array(y)
    maxd = max(y)
    mind = min(y)
    midpoint = (maxd-mind)/2 + mind

    after_ind = np.where(y > midpoint)[0][0]
    before_ind = np.where(y[:after_ind] < midpoint)[0][-1]

    xbefore = x[before_ind]
    xafter = x[after_ind]
    ybefore = y[before_ind]
    yafter = y[after_ind]

    slope = (yafter-ybefore)/(xafter-xbefore)
    rise_to_midpoint = midpoint-ybefore
    run_to_midpoint = rise_to_midpoint/slope

    return xbefore+run_to_midpoint, midpoint


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def bh_multitest(pvals, alpha=0.05):
    return multipletests(pvals, alpha=alpha, method='fdr_bh')


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
            if ind1 == ind2:
                continue
            p = test_func(vals1, vals2, nan_policy='omit')
            pvalue_matrix[ind1, ind2] = p.pvalue

    diag_mask = np.ones(pvalue_matrix.shape, dtype=bool)
    diag_mask[np.diag_indices_from(diag_mask)] = False
    corrected_pvals = multipletests(pvalue_matrix[np.where(diag_mask)], method='fdr_bh')
    pvalue_matrix[np.where(diag_mask)] = corrected_pvals[1]
    sig_matrix = pvalue_matrix < 0.05

    return pvalue_matrix, sig_matrix


def DiD_test(fsub, fset, nsub, nset, metric=np.mean):
    """Difference of differences permutation test."""
    observed = (metric(fsub) - metric(fset)) - (metric(nsub) - metric(nset))

    concatenated_subs = np.concatenate((fsub, nsub))
    concatenated_sets = np.concatenate((fset, nset))
    perms = []
    for iteration in range(1000):
        np.random.shuffle(concatenated_subs)
        np.random.shuffle(concatenated_sets)

        permf_sub = concatenated_subs[:len(fsub)]
        permf_set = concatenated_sets[:len(fset)]
        permn_sub = concatenated_subs[len(fsub):]
        permn_set = concatenated_sets[len(fset):]

        perms.append((metric(permf_sub) - metric(permf_set)) - (metric(permn_sub) - metric(permn_set)))

    p = (1 + np.sum(np.abs(perms) >= abs(observed))) / (1000 + 1)
    return p


def bootstrapped_diff_ci(data1, data2, metric=np.mean):
    bootstrapped_means = []
    for _ in range(1000):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        bootstrapped_means.append(metric(sample1) - metric(sample2))

    lower = np.percentile(bootstrapped_means, 2.5)
    upper = np.percentile(bootstrapped_means, 97.5)
    return lower, upper


# ---------------------------------------------------------------------------
# Population sparseness
# ---------------------------------------------------------------------------

def calc_pop_sparseness(resp_vector):
    N = len(resp_vector)
    numerator = (np.sum(resp_vector)/N)**2
    denom = np.sum(resp_vector**2/N)
    A = numerator/denom
    return (1 - A)/(1 - 1/N)


def calc_pop_sparseness_kurtosis(resp_vector):
    N = len(resp_vector)
    r_mean = np.mean(resp_vector)
    r_sigma = np.std(resp_vector)
    sum_term = np.sum(((resp_vector - r_mean)/r_sigma)**4)
    return (1/N)*sum_term - 3


def get_nonshared_images_from_imageids(datadict):
    shared_images = [b'im083_r', b'im111_r']
    image_keys = list(datadict.keys())
    return np.setdiff1d(image_keys, shared_images)


# ---------------------------------------------------------------------------
# Response averaging helpers (Figure 6)
# ---------------------------------------------------------------------------

def just_mean(x, baseline):
    return np.mean(x, axis=1)


def mean_with_base_sub(x, baseline):
    return np.mean(x, axis=1) - baseline


def get_decoding_results_files(session_list, file_list, image_set='change'):
    import os
    file_sessions = np.array([int(os.path.basename(f).split('_')[0]) for f in file_list])
    file_change = np.array([os.path.basename(f).split('_')[1].split('.npy')[0] for f in file_list])
    file_inds = (np.isin(file_sessions, session_list) & (file_change == image_set))
    return np.array(file_list)[file_inds]


def get_mouse_from_session(sessions_table, session_id):
    return sessions_table[sessions_table['ecephys_session_id'] == session_id]['mouse_id'].values[0]


def get_mouse_paired_indices(session_id_dict, sessions_table, region, n):
    sessions = [session_id_dict[region][c][n] for c in ['Familiar', 'Novel']]
    mouse_ids = [[get_mouse_from_session(sessions_table, id) for id in slist] for slist in sessions]
    matching_mice = np.intersect1d(*mouse_ids)

    matching_inds = {c: [] for c in ['Familiar', 'Novel']}
    for m in matching_mice:
        fam_ind = np.where(mouse_ids[0] == m)[0][0]
        nov_ind = np.where(mouse_ids[1] == m)[0][0]
        matching_inds['Familiar'].append(fam_ind)
        matching_inds['Novel'].append(nov_ind)

    return matching_inds


# ---------------------------------------------------------------------------
# Behavioral matrix helpers (Figure 7)
# ---------------------------------------------------------------------------

image_name_dict = {
    'H': ['im005_r', 'im024_r', 'im034_r', 'im087_r', 'im104_r', 'im114_r', 'im083_r', 'im111_r'],
    'G': ['im012_r', 'im036_r', 'im044_r', 'im047_r', 'im078_r', 'im115_r', 'im083_r', 'im111_r']
}


def get_contingent_engaged_trials(trials):
    return trials[(~trials['aborted'].astype(bool)) & (~trials['auto_rewarded'].astype(bool)) & (trials['reward_rate'] >= 2)]


def beh_mat_from_stim_table(stim_table, session_id=None):
    if session_id is not None:
        trials = get_contingent_engaged_trials(stim_table[stim_table['session_id'] == session_id].groupby('behavior_trial_id').head(1))
    else:
        trials = get_contingent_engaged_trials(stim_table.groupby('behavior_trial_id').head(1))

    images = image_name_dict[trials['image_set'].iloc[0]]

    beh_mat = np.full((8, 8), np.nan)
    count_mat = np.full((8, 8), np.nan)
    for pre_ind, pre_image in enumerate(images):
        for ch_ind, change_image in enumerate(images):
            condition_trials = trials[(trials['initial_image_name'] == pre_image) & (trials['change_image_name'] == change_image)]
            total_in_condition = len(condition_trials)
            total_responses = np.sum((condition_trials['hit'].astype(bool)) | (condition_trials['false_alarm'].astype(bool)))
            response_rate = total_responses/total_in_condition
            beh_mat[pre_ind, ch_ind] = response_rate
            count_mat[pre_ind, ch_ind] = total_in_condition

    return images, count_mat, beh_mat


def mean_paired_image_mat_from_stim_table(stim_table, col_to_average, session_id=None,
                                          only_changes_shams=True, only_engaged=True, experience='Novel', image_set='H'):
    engaged_filter = (stim_table['engaged']).astype(bool) if only_engaged else [True]*len(stim_table)
    experience_filter = (stim_table['experience_level'] == experience).astype(bool)
    change_filter = (stim_table['is_change'].astype(bool) | stim_table['is_sham_change'].astype(bool)) if only_changes_shams else [True]*len(stim_table)

    if session_id is not None:
        stim = stim_table[(stim_table['session_id'] == session_id) & engaged_filter & experience_filter & change_filter]
    else:
        stim = stim_table[engaged_filter & experience_filter & change_filter]

    images = image_name_dict[image_set]

    mean_mat = np.full((8, 8), np.nan)
    count_mat = np.full((8, 8), np.nan)
    for pre_ind, pre_image in enumerate(images):
        for ch_ind, change_image in enumerate(images):
            condition_trials = stim[(stim['initial_image_name'] == pre_image) & (stim['change_image_name'] == change_image)]
            total_in_condition = len(condition_trials)
            column_mean = np.nanmean(condition_trials[col_to_average])
            mean_mat[pre_ind, ch_ind] = column_mean
            count_mat[pre_ind, ch_ind] = total_in_condition

    return images, count_mat, mean_mat


def get_omission_response_rate(trials):
    omissions = trials[trials['omitted'] & trials['change_eligible_window'] & (trials['reward_rate'] >= 2) & (~trials['grace_period_after_hit'])]
    return np.sum(omissions['lick_for_flash_during_response_window'])/len(omissions)


def get_post_omission_response_rate(trials):
    omissions = trials[trials['previous_omitted'] & trials['change_eligible_window'] & (trials['reward_rate'] >= 2) & (~trials['grace_period_after_hit'])]
    return np.sum(omissions['lick_for_flash_during_response_window'])/len(omissions)


def get_shared_hit_rate(trials):
    """Hit rate for shared images (im083_r and im111_r)."""
    good_trials = get_contingent_engaged_trials(trials)
    image_trials = good_trials[good_trials['change_image_name'].isin(['im083_r', 'im111_r'])]
    return np.sum(image_trials['hit'].astype(bool)) / np.sum(image_trials['go'].astype(bool))


def get_private_hit_rate(trials):
    """Hit rate for private (non-shared) images."""
    good_trials = get_contingent_engaged_trials(trials)
    image_trials = good_trials[~good_trials['change_image_name'].isin(['im083_r', 'im111_r'])]
    return np.sum(image_trials['hit'].astype(bool)) / np.sum(image_trials['go'].astype(bool))


def get_shared_fa_rate(trials):
    """False alarm rate for shared images (im083_r and im111_r)."""
    good_trials = get_contingent_engaged_trials(trials)
    image_trials = good_trials[good_trials['change_image_name'].isin(['im083_r', 'im111_r'])]
    return np.sum(image_trials['false_alarm'].astype(bool)) / np.sum(image_trials['catch'].astype(bool))


def get_private_fa_rate(trials):
    """False alarm rate for private (non-shared) images."""
    good_trials = get_contingent_engaged_trials(trials)
    image_trials = good_trials[~good_trials['change_image_name'].isin(['im083_r', 'im111_r'])]
    return np.sum(image_trials['false_alarm'].astype(bool)) / np.sum(image_trials['catch'].astype(bool))


def get_shared_nonchange_response_rate(stim_trials):
    """Non-change response rate for shared images (im083_r and im111_r)."""
    image_trials = stim_trials[
        stim_trials['image_name'].isin(['im083_r', 'im111_r']) &
        stim_trials['change_eligible_window'] &
        ~stim_trials['is_change'].astype(bool) &
        (stim_trials['reward_rate'] >= 2) &
        (~stim_trials['grace_period_after_hit'])
    ]
    return np.sum(image_trials['lick_for_flash_during_response_window'])/len(image_trials)


def get_private_nonchange_response_rate(stim_trials):
    """Non-change response rate for private (non-shared) images."""
    image_trials = stim_trials[
        ~stim_trials['image_name'].isin(['im083_r', 'im111_r', 'omitted']) &
        stim_trials['change_eligible_window'] &
        ~stim_trials['is_change'].astype(bool) &
        (stim_trials['reward_rate'] >= 2) &
        (~stim_trials['grace_period_after_hit'])
    ]
    return np.sum(image_trials['lick_for_flash_during_response_window'])/len(image_trials)


def mean_beh_mat_across_sessions(stim_table, session_list):
    beh_mats = []
    count_mats = []
    image_sets = []
    for session_id in session_list:
        image_set, count_mat, beh_mat = beh_mat_from_stim_table(stim_table, session_id)
        beh_mats.append(beh_mat)
        count_mats.append(count_mat)
        image_sets.append(image_set)
    return image_sets, count_mats, beh_mats


def skip_diag_masking(array):
    return array.T[~np.eye(array.shape[0], dtype=bool)].reshape(array.shape[0], -1).T


def calcHitRate(hits, misses, adjusted=False):
    n = hits + misses
    if n == 0:
        return np.nan
    hitRate = hits/n
    if adjusted:
        if hitRate == 0:
            hitRate = 0.5/n
        elif hitRate == 1:
            hitRate = 1 - 0.5/n
    return hitRate


def calcDprime(hits, misses, falseAlarms, correctRejects):
    hitRate = calcHitRate(hits, misses, adjusted=True)
    falseAlarmRate = calcHitRate(falseAlarms, correctRejects, adjusted=True)
    z = [scipy.stats.norm.ppf(r) for r in (hitRate, falseAlarmRate)]
    return z[0] - z[1]


def get_session_engaged_dprime(stim_table, session_id):
    trials = get_contingent_engaged_trials(stim_table[stim_table['session_id'] == session_id].groupby('behavior_trial_id').head(1))
    num_hits = np.sum(trials['hit'])
    num_misses = np.sum(trials['miss'])
    num_fas = np.sum(trials['false_alarm'])
    num_crs = np.sum(trials['correct_reject'])
    return calcDprime(num_hits, num_misses, num_fas, num_crs)


def get_session_engaged_hit_count(stim_table, session_id):
    trials = get_contingent_engaged_trials(stim_table[stim_table['session_id'] == session_id].groupby('behavior_trial_id').head(1))
    return np.sum(trials['hit'])


def get_experience_session_id_for_mouse(sessions_table, mouse_id, experience_level):
    session = sessions_table[(sessions_table['mouse_id'] == mouse_id) & (sessions_table['experience_level'] == experience_level)]
    if len(session) > 0:
        return session['ecephys_session_id'].values[0]


def get_omission_mean(trials, column):
    omissions = trials[trials['omitted'] & trials['change_eligible_window'] & (trials['reward_rate'] >= 2) & (~trials['grace_period_after_hit'])]
    return np.mean(omissions[column])


def get_post_omission_mean(trials, column):
    omissions = trials[trials['previous_omitted'] & trials['change_eligible_window'] & (trials['reward_rate'] >= 2) & (~trials['grace_period_after_hit'])]
    return np.mean(omissions[column])


def get_shared_change_mean(trials, column):
    """Mean of column for shared image (im083_r, im111_r) change trials."""
    good_trials = get_contingent_engaged_trials(trials)
    image_trials = good_trials[(good_trials['image_name'].isin(['im083_r', 'im111_r'])) & (good_trials['is_change'])]
    return np.mean(image_trials[column])


def get_private_change_mean(trials, column):
    """Mean of column for private (non-shared) image change trials."""
    good_trials = get_contingent_engaged_trials(trials)
    image_trials = good_trials[(~good_trials['image_name'].isin(['im083_r', 'im111_r'])) & (good_trials['is_change'])]
    return np.mean(image_trials[column])


def get_shared_catch_mean(trials, column):
    """Mean of column for shared image (im083_r, im111_r) sham change trials."""
    good_trials = get_contingent_engaged_trials(trials)
    image_trials = good_trials[(good_trials['image_name'].isin(['im083_r', 'im111_r'])) & (good_trials['is_sham_change'])]
    return np.mean(image_trials[column])


def get_private_catch_mean(trials, column):
    """Mean of column for private (non-shared) image sham change trials."""
    good_trials = get_contingent_engaged_trials(trials)
    image_trials = good_trials[(~good_trials['image_name'].isin(['im083_r', 'im111_r'])) & (good_trials['is_sham_change'])]
    return np.mean(image_trials[column])


def get_shared_nonchange_mean(stim_trials, column):
    """Mean of column for shared image non-change eligible flashes."""
    image_trials = stim_trials[
        stim_trials['image_name'].isin(['im083_r', 'im111_r']) &
        stim_trials['change_eligible_window'] &
        ~stim_trials['is_change'].astype(bool) &
        (stim_trials['reward_rate'] >= 2) &
        (~stim_trials['grace_period_after_hit'])
    ]
    return np.mean(image_trials[column])


def get_private_nonchange_mean(stim_trials, column):
    """Mean of column for private image non-change eligible flashes."""
    image_trials = stim_trials[
        ~stim_trials['image_name'].isin(['im083_r', 'im111_r', 'omitted']) &
        stim_trials['change_eligible_window'] &
        ~stim_trials['is_change'].astype(bool) &
        (stim_trials['reward_rate'] >= 2) &
        (~stim_trials['grace_period_after_hit'])
    ]
    return np.mean(image_trials[column])


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def make_mono_colormap(start_color, end_color):
    return LinearSegmentedColormap.from_list('custom', [start_color, end_color], N=1028)


def plot_image(image, gamma=1.0):
    from matplotlib import pyplot as plt
    from matplotlib.colors import PowerNorm
    fig, ax = plt.subplots()
    norm = PowerNorm(gamma)
    ax.imshow(image, cmap='Greys_r', norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    return fig


# ---------------------------------------------------------------------------
# GLM / variance explained
# ---------------------------------------------------------------------------

def variance_explained(row):
    psth = row['psth']
    prediction = row['predicted_psth']
    nan_inds = np.isnan(psth) | np.isnan(prediction)
    if np.sum(nan_inds) > 5:
        return np.nan
    return 1 - (np.sum((psth[~nan_inds] - prediction[~nan_inds])**2) / np.sum((psth[~nan_inds] - np.nanmean(psth))**2))


def resample_df_to_times(df, time_column, val_column, new_times):
    timestamps = df[time_column].values
    vals = df[val_column].values
    interpolator = interp1d(timestamps, vals, kind='linear', bounds_error=False)
    new_values = interpolator(new_times)
    return new_values, new_times


# ---------------------------------------------------------------------------
# Session stim table stats
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure 6 helpers
# ---------------------------------------------------------------------------


def calc_dprime(vals1, vals2, signed=False):
    """Cohen's d-prime between two arrays."""
    vals1 = vals1[~np.isnan(vals1)]
    vals2 = vals2[~np.isnan(vals2)]
    if signed:
        diff_means = vals1.mean() - vals2.mean()
    else:
        diff_means = np.abs(vals1.mean() - vals2.mean())
    mean_sigma = (vals1.std() + vals2.std()) / 2
    return diff_means / mean_sigma


def upsample_interp(data, factor=100):
    """Upsample 1-D array using linear interpolation (np.interp).
    Distinct from upsample() which uses cubic interp1d."""
    x = np.arange(len(data))
    x_upsampled = np.arange(0, len(data), 1 / factor)
    return np.interp(x_upsampled, x, data)


def time_to_threshold_from_baseline_back_from_peak(data, threshold=2, upsample_factor=100):
    """Find time (in original-sample units) where upsampled data crosses threshold,
    searching backwards from the response peak."""
    data_up = upsample_interp(data, upsample_factor)
    max_time = np.argmax(data_up)
    backwards_from_max = data_up[:max_time][::-1]
    time_to_threshold = np.where(backwards_from_max <= threshold)[0]
    if len(time_to_threshold) == 0:
        return np.nan, np.nan
    t = max_time - time_to_threshold[0]
    return t / upsample_factor, data_up[t]


def conf_interval(vals, ci_range=95):
    """Return (lower, upper) percentile confidence interval, ignoring NaNs."""
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    tail = (100 - ci_range) / 2
    return np.percentile(vals, tail), np.percentile(vals, 100 - tail)


# ---------------------------------------------------------------------------
# FigureS2 helpers
# ---------------------------------------------------------------------------

def plot_raster(ax, spikes, start_times, time_before=0.1, time_after=0.2):
    """Plot a spike raster aligned to a set of start times.

    Parameters
    ----------
    ax : matplotlib Axes
    spikes : array-like
        Spike times in seconds.
    start_times : array-like
        Event onset times to align spikes to.
    time_before : float
        Seconds before each onset to include.
    time_after : float
        Seconds after each onset to include.
    """
    raster = []
    for start in start_times:
        r = spikes[(spikes >= start - time_before) & (spikes <= start + time_after)]
        if len(r) > 0:
            raster.append(r - start)
        else:
            raster.append([time_after * 2])
    ax.eventplot(raster, color='k')
    ax.set_xlim([-time_before, time_after])


def get_opto_responses_for_units(unit_ids, opto_tensor, waveform='cosine'):
    """Extract opto response tensor for a set of unit IDs."""
    opto_tensor_unit_ids = opto_tensor['opto']['unitIds'][()]
    unit_tensor_indices = np.array([np.where(opto_tensor_unit_ids == uid)[0][0] for uid in unit_ids])
    opto_tensor_data = opto_tensor['opto'][f'spikes_{waveform}']
    opto_responses = np.full((len(unit_ids), opto_tensor_data.shape[1], opto_tensor_data.shape[2]), np.nan)
    for i, unit_index in enumerate(unit_tensor_indices):
        opto_responses[i] = opto_tensor_data[unit_index]
    return opto_responses


def scatter_ccf(unit_data, plane='coronal', c='(absolute_change_from_full, licks)',
                cmap='viridis_r', alpha=1, size=5, clim=[None, None], ax=None):
    """Scatter plot of units in CCF space colored by a metric."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import pyplot as plt
    plot_data = unit_data.sort_values(by=c, ascending=False)
    plot_data = plot_data.copy()
    plot_data[c] = -plot_data[c]
    if ax is None:
        fig, ax = plt.subplots()
    if plane == 'coronal':
        sc = ax.scatter(456 * 25 - plot_data['left_right_ccf_coordinate'],
                        plot_data['dorsal_ventral_ccf_coordinate'],
                        c=plot_data[c], cmap=cmap, alpha=alpha, s=size, vmin=clim[0], vmax=clim[1])
        ax.invert_yaxis()
        ax.set_xlabel('left->right')
        ax.set_ylabel('ventral->dorsal')
    elif plane == 'sagittal':
        sc = ax.scatter(plot_data['anterior_posterior_ccf_coordinate'],
                        plot_data['dorsal_ventral_ccf_coordinate'],
                        c=plot_data[c], cmap=cmap, alpha=alpha, s=size, vmin=clim[0], vmax=clim[1])
        ax.invert_yaxis()
        ax.set_xlabel('anterior->posterior')
        ax.set_ylabel('ventral->dorsal')
    elif plane == 'horizontal':
        sc = ax.scatter(456 * 25 - plot_data['left_right_ccf_coordinate'],
                        plot_data['anterior_posterior_ccf_coordinate'],
                        c=plot_data[c], cmap=cmap, alpha=alpha, s=size, vmin=clim[0], vmax=clim[1])
        ax.invert_yaxis()
        ax.set_xlabel('left->right')
        ax.set_ylabel('posterior->anterior')
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(sc, cax=cbar_ax)
    return ax


def binned_stat_ccf(unit_data, binsize, plane='coronal',
                    c='(absolute_change_from_full, licks)', cmap='viridis',
                    alpha=1, statistic='mean', ax=None, zscore=False):
    """Binned 2-D statistic of a metric in CCF space."""
    from scipy.stats import binned_statistic_2d
    from matplotlib import pyplot as plt
    if zscore:
        metric = (unit_data[c] - np.nanmean(unit_data[c])) / np.nanstd(unit_data[c])
    else:
        metric = unit_data[c]
    if ax is None:
        fig, ax = plt.subplots()
    if plane == 'coronal':
        x = 456 * 25 - unit_data['left_right_ccf_coordinate']
        y = unit_data['dorsal_ventral_ccf_coordinate']
        xlabel, ylabel = 'left->right', 'ventral->dorsal'
    elif plane == 'sagittal':
        x = unit_data['anterior_posterior_ccf_coordinate']
        y = unit_data['dorsal_ventral_ccf_coordinate']
        xlabel, ylabel = 'anterior->posterior', 'ventral->dorsal'
    elif plane == 'horizontal':
        x = 456 * 25 - unit_data['left_right_ccf_coordinate']
        y = unit_data['anterior_posterior_ccf_coordinate']
        xlabel, ylabel = 'left->right', 'posterior->anterior'
    x_bins = np.arange(np.min(x), np.max(x), binsize)
    y_bins = np.arange(np.min(y), np.max(y), binsize)
    stat, x_edge, y_edge, _ = binned_statistic_2d(x, y, metric, statistic=statistic, bins=[x_bins, y_bins])
    im = ax.imshow(stat.T, extent=[x_edge[0], x_edge[-1], y_edge[-1], y_edge[0]], cmap=cmap, aspect='equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)
    return stat, x_edge, y_edge


def weighted_gaussian_filter(data, weights, sigma):
    """Apply weighted Gaussian smoothing to 3D data."""
    from scipy.ndimage import gaussian_filter
    nanvals = np.isnan(data)
    data = data.copy()
    weights = weights.copy()
    data[nanvals] = np.nanmean(data)
    weights[nanvals] = 1
    smoothed_data = gaussian_filter(data * weights, sigma=sigma)
    smoothed_weights = gaussian_filter(weights, sigma=sigma)
    smoothed_data /= smoothed_weights
    smoothed_data[nanvals] = np.nan
    return smoothed_data


# ---------------------------------------------------------------------------
# FigureS13 helpers
# ---------------------------------------------------------------------------


def calculate_metric_for_selection(stims, col, metric_func, *query_strings):
    """Apply a metric function to a column after chaining query strings."""
    chained_query = ' & '.join(query_strings)
    stims_subset = stims.query(chained_query)
    return metric_func(stims_subset[col])


def paired_image_mat_from_stim_table(stims, col, image_set, metric_func, *query_strings):
    """Build an 8x8 matrix of a metric for each (pre_image, change_image) pair."""
    chained_query = ' & '.join(query_strings)
    chained_query = chained_query + f' & image_set=="{image_set}"'
    stims_subset = stims.query(chained_query)
    images = image_name_dict[image_set]
    mean_mat = np.full((8, 8), np.nan)
    count_mat = np.full((8, 8), np.nan)
    for pre_ind, pre_image in enumerate(images):
        for ch_ind, change_image in enumerate(images):
            condition_trials = stims_subset[
                (stims_subset['initial_image_name'] == pre_image) &
                (stims_subset['change_image_name'] == change_image)
            ]
            mean_mat[pre_ind, ch_ind] = metric_func(condition_trials[col])
            count_mat[pre_ind, ch_ind] = len(condition_trials)
    return images, count_mat, mean_mat


# ---------------------------------------------------------------------------
# Figure 4/5 novelty modulation helpers
# ---------------------------------------------------------------------------


def get_nov_mod_index_norm_bootstrap(fresps, nresps, iterations=10000, aggfunc=np.nanmean):
    """Bootstrap normalised novelty modulation index."""
    inds = []
    for iteration in range(iterations):
        fs = np.random.choice(fresps, len(fresps), replace=True)
        ns = np.random.choice(nresps, len(nresps), replace=True)
        fmed = aggfunc(fs)
        nmed = aggfunc(ns)
        max_abs_dev = np.max([np.abs(v) for v in [fmed, nmed]])
        inds.append((nmed - fmed) / max_abs_dev)
    return inds


def get_nov_mod_index_norm(fresps, nresps, aggfunc=np.nanmean):
    """Point estimate of normalised novelty modulation index."""
    fmed = aggfunc(fresps)
    nmed = aggfunc(nresps)
    max_abs_dev = np.max([np.abs(v) for v in [fmed, nmed]])
    return (nmed - fmed) / max_abs_dev


def get_mod_index(vals1, vals2, *args):
    """Modulation index (v1-v2)/(v1+v2), returns NaN if either is non-positive."""
    if isinstance(vals1, (float, int)):
        if vals1 > 0 and vals2 > 0:
            return (vals1 - vals2) / (vals1 + vals2)
        return np.nan
    inds = []
    for val1, val2 in zip(vals1, vals2):
        if val1 > 0 and val2 > 0:
            inds.append((val1 - val2) / (val1 + val2))
        else:
            inds.append(np.nan)
    return inds


def get_mod_index_norm(vals1, vals2, *args):
    """Normalised modulation index (v1-v2)/max(|v1|,|v2|)."""
    if isinstance(vals1, (float, int)):
        return (vals1 - vals2) / np.max(np.abs([vals1, vals2]))
    max_abs_dev = np.array([np.max([np.abs(v1), np.abs(v2)]) for v1, v2 in zip(vals1, vals2)])
    return (np.array(vals1) - np.array(vals2)) / max_abs_dev


def normalize_trace_pair(trace, trace2, norm_slice=None):
    """Normalize trace by the max abs deviation of either trace within norm_slice.
    Distinct from normalize() which normalises a matrix relative to chance."""
    if norm_slice is None:
        norm_slice = slice(0, len(trace))
    abs_max_dev1 = np.max(np.abs(trace[norm_slice]))
    abs_max_dev2 = np.max(np.abs(trace2[norm_slice]))
    return trace / np.nanmax([abs_max_dev1, abs_max_dev2])


def get_session_stim_table_stats(group):
    first_trial = group.iloc[0]['stimulus_presentations_id']
    last_trial = group.iloc[-1]['stimulus_presentations_id']
    max_diff = np.diff(group['stimulus_presentations_id'].values).max()
    return first_trial, last_trial, max_diff, len(group), group['behavior_trial_id'].min()

# --- FigureS13 response property helpers ---

def fwhm(binwidth, trace):
    half_max = np.max(trace) / 2
    max_index = np.argmax(trace)
    try:
        left_index = np.where(trace[:max_index] <= half_max)[0][-1]
        right_index = np.where(trace[max_index:] <= half_max)[0][0] + max_index
        result = right_index - left_index
    except:
        result = np.nan
    return result * binwidth

def fraction_above_half_max(binwidth, trace):
    half_max = np.max(trace) / 2
    return np.sum(trace > half_max) / len(trace)

def get_peak_time(trace_orig, resp_slice, bin_size):
    trace = trace_orig[resp_slice]
    return (np.argmax(trace) + resp_slice.start) * bin_size
