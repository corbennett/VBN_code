"""
build_opto_metrics.py

Iterates over all optotagging sessions in the VBN cache, computes the four
columns required by decoding_utils.py for unit classification:

    pulse_high_mean_evoked_rate_zscored
    pulse_high_first_spike_latency
    pulse_high_first_spike_jitter
    raised_cosine_high_fraction_time_responsive

Saves the result to unit_opto_metrics.csv.

Usage:
    python build_opto_metrics.py --cache_dir <path> --save_path <path>
"""

import argparse
import os
import numpy as np
import pandas as pd

from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

from opto_tagging_utils import (
    mean_trial_spike_rate,
    first_spike_latency,
    first_spike_jitter,
    fraction_time_responsive,
    get_baseline_bin_rates,
    rename_levels_in_metrics_df,
    get_evoked_rates,
)

DEFAULT_SAVE_PATH = (
    "/Volumes/programs/mindscope/workgroups/np-behavior/"
    "vbn_data_release/supplemental_tables/unit_opto_metrics_claude.csv"
)

CENSOR_PERIOD = 0.0015
DURATIONS = {'pulse': 0.010 - 2 * CENSOR_PERIOD,
             'raised_cosine': 1 - 2 * CENSOR_PERIOD}
BINSIZES = {'pulse': 0.001, 'raised_cosine': 0.01}


def compute_session_opto_metrics(session):
    """Return a DataFrame of per-unit opto metrics for one session.

    Computes only the metrics needed for SST/VIP classification in
    decoding_utils.py:
        {stim}_{level}_mean_trial_spike_rate  (pulse, all levels – for evoked rate)
        {stim}_{level}_first_spike_latency    (pulse, all levels)
        {stim}_{level}_first_spike_jitter     (pulse, all levels)
        raised_cosine_{level}_fraction_time_responsive
        pulse_baseline_mean / pulse_baseline_std
    """
    spike_times = session.spike_times
    units = session.get_units()

    opto_table = session.optotagging_table
    start_times = opto_table.groupby(['stimulus_name', 'level'])['start_time'].apply(list)
    conditions = start_times.index.get_level_values(0)
    levels = start_times.index.get_level_values(1)

    # Baseline windows: gaps between consecutive opto stimuli
    censor = 0.005
    bl_starts = opto_table['stop_time'].values[:-1] + censor
    bl_ends = opto_table['start_time'].values[1:] - censor
    min_gap = np.min(bl_ends - bl_starts)
    bl_starts = bl_starts + min_gap / 2

    rows = []
    for unit_id, _ in units.iterrows():
        spikes = spike_times[unit_id]
        row = {'uid': unit_id}

        # Baseline stats (pulse window size) for evoked-rate z-scoring
        pulse_baseline = get_baseline_bin_rates(spikes, bl_starts, bl_ends,
                                                binsize=DURATIONS['pulse'])
        row['pulse_baseline_mean'] = np.mean(pulse_baseline)
        row['pulse_baseline_std'] = np.std(pulse_baseline)

        # Per-condition, per-level metrics
        rc_baseline = get_baseline_bin_rates(spikes, bl_starts, bl_ends,
                                             binsize=BINSIZES['raised_cosine'])

        for starts, condition, level in zip(start_times, conditions, levels):
            starts = np.array(starts)
            duration = DURATIONS[condition]
            col_prefix = f'{condition}_{level}'

            if condition == 'pulse':
                row[f'{col_prefix}_mean_trial_spike_rate'] = mean_trial_spike_rate(
                    spikes, starts + CENSOR_PERIOD, duration)
                row[f'{col_prefix}_first_spike_latency'] = first_spike_latency(
                    spikes, starts + CENSOR_PERIOD, duration)
                row[f'{col_prefix}_first_spike_jitter'] = first_spike_jitter(
                    spikes, starts + CENSOR_PERIOD, duration)

            elif condition == 'raised_cosine':
                row[f'{col_prefix}_fraction_time_responsive'] = fraction_time_responsive(
                    spikes, starts, 0, duration, rc_baseline, BINSIZES['raised_cosine'])

        rows.append(row)

    return pd.DataFrame(rows)


def build_unit_opto_metrics(cache, save_path):
    session_table = cache.get_ecephys_session_table()

    # Only sessions from Cre lines with opsin expression
    opto_sessions = session_table[
        session_table['genotype'].str.contains('Sst|Vip', na=False)
    ]
    print(f"Found {len(opto_sessions)} optotagging sessions")

    all_metrics = []
    for i, session_id in enumerate(opto_sessions.index):
        print(f"  [{i+1}/{len(opto_sessions)}] session {session_id}")
        try:
            session = cache.get_ecephys_session(ecephys_session_id=session_id)
            metrics = compute_session_opto_metrics(session)
            metrics = get_evoked_rates(metrics)
            metrics = rename_levels_in_metrics_df(metrics)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"    SKIPPED: {e}")

    unit_opto_metrics = pd.concat(all_metrics, ignore_index=True)
    unit_opto_metrics.to_csv(save_path, index=False)
    print(f"Saved {len(unit_opto_metrics)} rows to {save_path}")
    return unit_opto_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()

    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir=args.cache_dir)

    build_unit_opto_metrics(cache, args.save_path)
