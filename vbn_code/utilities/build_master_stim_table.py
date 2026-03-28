"""
build_master_stim_table.py

Consolidated script to build master_stim_table_no_filter.csv from scratch.
Processes each session once, adding all columns in a single pass before
concatenating into the final table.

Previously split across:
  hpc_code/make_stim_trial_tables.py
  hpc_code/stimwise_running_speed.py
  support_files/VBN_add_stimwise_running_to_stimtable.ipynb
  support_files/VBN_familiar_novel_behavior.ipynb

Usage:
  python build_master_stim_table.py \
    --cache_dir /Volumes/programs/.../vbn_s3_cache \
    --output_file /Volumes/programs/.../supplemental_tables/master_stim_table_no_filter.csv
"""
import argparse
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache


def build_session_stim_table(session):
    """
    Build per-session stim trials table with running speed columns.

    Returns a DataFrame with one row per active-block stimulus presentation.
    Passive-block running speed is included as a paired column for comparison.
    """
    sess_id = session.metadata['ecephys_session_id']
    stim = session.stimulus_presentations.copy()
    trials = session.trials.copy()
    licks = session.licks
    running = session.running_speed

    # ---- Lick processing ------------------------------------------------
    lick_times = licks.timestamps.values
    lick_diffs = np.diff(lick_times)
    lick_bout_starts = lick_times[np.insert(lick_diffs, 0, 1) > 0.5]
    lick_bout_starts = lick_bout_starts[lick_bout_starts > stim['start_time'].iloc[0]]
    lick_bout_ends = lick_times[np.append(lick_diffs, 1) > 0.5]
    lick_bout_ends = lick_bout_ends[lick_bout_ends >= lick_bout_starts[0]]
    time_from_last_bout = lick_bout_starts[1:] - lick_bout_ends[:-1]

    lick_flashes = np.searchsorted(stim['start_time'].values, lick_bout_starts) - 1
    stim['lick_time'] = np.nan
    stim['time_from_last_lick_bout'] = np.nan
    stim.loc[stim.index[lick_flashes], 'lick_time'] = lick_bout_starts
    stim.loc[stim.index[lick_flashes], 'time_from_last_lick_bout'] = np.insert(time_from_last_bout, 0, -1)
    stim['previous_omitted'] = np.insert(stim['omitted'].values[:-1], 0, False)
    stim['lick_for_flash'] = stim['lick_time'].notna()

    # ---- Assign stims to behavior trials --------------------------------
    col_names = ['hit', 'false_alarm', 'miss', 'correct_reject', 'aborted']
    trials['trial_type'] = trials[col_names].idxmax(axis=1)
    trials['previous_trial_type'] = np.insert(trials['trial_type'].values[:-1], 0, 'none')

    stim_trial_assignments = np.searchsorted(trials['start_time'].values, stim['start_time'].values) - 1
    stim['behavior_trial_id'] = stim_trial_assignments.astype(float)
    stim.loc[stim['start_time'] > trials.iloc[-1]['stop_time'], 'behavior_trial_id'] = np.nan
    # Copy active-block trial assignments to the passive replay block
    stim.loc[stim['stimulus_block'] == 5, 'behavior_trial_id'] = (
        stim[stim['active']]['behavior_trial_id'].values
    )

    stim_trials = stim.merge(trials, left_on='behavior_trial_id', right_index=True,
                             how='left', suffixes=('_x', '_y'))
    
    # In SDK v0.5.0, stimulus_presentations uses 'end_time' (not 'stop_time'),
    # so only start_time and is_change collide between stim and trials.
    stim_trials = stim_trials.rename(columns={
        'start_time_x': 'start_time',
        'start_time_y': 'trial_start_time',
        'stop_time': 'trial_stop_time',    # only trials has stop_time
        'end_time': 'stop_time',           # stim end_time becomes stop_time
        'is_change_x': 'is_change',        # keep stim version; drop trial duplicate below
    })
    stim_trials = stim_trials.drop(columns=['is_change_y'], errors='ignore')

    # SDK encodes some missing values as -99; convert to NaN for consistency
    for col in ('change_frame', 'stimulus_index'):
        if col in stim_trials.columns:
            stim_trials[col] = stim_trials[col].replace(-99, np.nan)

    # ---- Flash count columns --------------------------------------------
    flashes_since_trial_began = []
    counter = -1
    last_trial_id = stim['behavior_trial_id'].iloc[0]
    for tid in stim['behavior_trial_id'].values:
        if last_trial_id == tid:
            counter += 1
        else:
            counter = 0
        flashes_since_trial_began.append(counter)
        last_trial_id = tid
    stim_trials['trial_flash'] = flashes_since_trial_began

    # Flashes since last lick bout start
    flashes_since_last_lick_bout = []
    counter = 0
    for lt in stim_trials['lick_time'].values:
        counter += 1
        flashes_since_last_lick_bout.append(counter)
        if not np.isnan(lt):
            counter = 0
    stim_trials['flashes_since_last_lick_bout_start'] = flashes_since_last_lick_bout

    # Flashes since last individual lick (not just bout start)
    all_lick_flash_inds = np.append(
        np.searchsorted(stim['start_time'].values, lick_times) - 1, -1
    )
    flashes_since_last_lick = [
        np.min(flash - all_lick_flash_inds[all_lick_flash_inds < flash])
        for flash in stim_trials.index.values
    ]
    stim_trials['flashes_since_last_lick'] = flashes_since_last_lick
    first_lick_stim = stim_trials['lick_time'].idxmin()
    stim_trials.loc[:first_lick_stim, 'flashes_since_last_lick'] = np.nan

    first_lick_in_trial = (
        stim_trials[['lick_time', 'behavior_trial_id']]
        .groupby('behavior_trial_id')
        .transform('min')
    )
    stim_trials['first_lick_in_trial'] = np.isin(stim_trials['lick_time'], first_lick_in_trial)

    # ---- Running speed (paired active / passive columns) ----------------
    def compute_running_for_block(block_stim):
        """Vectorized running speed computation using searchsorted."""
        ts = running['timestamps'].values
        sp = running['speed'].values
        start_times = block_stim['start_time'].values
        stop_times = block_stim['stop_time'].values

        speeds = np.full(len(block_stim), np.nan)
        baselines = np.full(len(block_stim), np.nan)

        # Indices bracketing each stimulus window and 200ms baseline window
        stim_start_idx = np.searchsorted(ts, start_times, side='left')
        stim_end_idx = np.searchsorted(ts, stop_times, side='left')
        base_start_idx = np.searchsorted(ts, start_times - 0.2, side='left')

        for i in range(len(block_stim)):
            s, e = stim_start_idx[i], stim_end_idx[i]
            if e > s:
                speeds[i] = np.nanmean(sp[s:e])
            b = base_start_idx[i]
            if stim_start_idx[i] > b:
                baselines[i] = np.nanmean(sp[b:stim_start_idx[i]])

        return speeds, baselines

    active_stim = stim_trials[stim_trials['active']]
    passive_stim = stim_trials[stim_trials['stimulus_block'] == 5]

    active_speed, active_baseline = compute_running_for_block(active_stim)
    passive_speed, passive_baseline = compute_running_for_block(passive_stim)

    # Each active row gets both its own running speed and the corresponding
    # passive-replay running speed for direct comparison
    stim_trials.loc[active_stim.index, 'active_running_speed'] = active_speed
    stim_trials.loc[active_stim.index, 'active_baseline_running_speed'] = active_baseline
    stim_trials.loc[active_stim.index, 'passive_running_speed'] = passive_speed
    stim_trials.loc[active_stim.index, 'passive_baseline_running_speed'] = passive_baseline

    stim_trials['session_id'] = sess_id

    active = stim_trials[stim_trials['active']].copy()
    active['stimulus_presentations_id'] = np.arange(len(active))
    return active


def add_derived_columns(stim_table, sessions_table):
    """
    Add derived behavioral columns that require the full concatenated table.
    Combines logic from VBN_add_stimwise_running_to_stimtable.ipynb and
    VBN_familiar_novel_behavior.ipynb.

    sessions_table should be the ecephys session table from the cache
    (cache.get_ecephys_session_table()), reset so ecephys_session_id is a column.
    """
    # Shared images (present in both G and H image sets)
    stim_table['is_shared'] = stim_table['image_name'].isin(['im083_r', 'im111_r'])

    # Binary reward column (rewarded is already in stimulus_presentations)
    stim_table['reward'] = stim_table['rewarded'].astype(int)

    # reward_rate: rolling 80-trial sum of rewards per session (center=True)
    # NaN at edges where fewer than 80 observations fill the window (matches original)
    stim_table['reward_rate'] = (
        stim_table.groupby('session_id')['reward']
        .transform(lambda x: x.rolling(80, center=True).sum())
    )

    # reaction_time: time from flash onset to first lick
    stim_table['reaction_time'] = stim_table['lick_time'] - stim_table['start_time']

    # Engagement: reward_rate >= 2 rewards/min
    stim_table['engaged'] = stim_table['reward_rate'] >= 2

    # Experience level (Familiar / Novel) and abnormality flags from cache sessions table
    sess_lookup = sessions_table.set_index('ecephys_session_id')

    # image_set: G or H, derived from stimulus_name
    def get_image_set(stimulus_name):
        return 'G' if '_G_' in str(stimulus_name) else 'H'

    stim_table['image_set'] = stim_table['stimulus_name'].apply(get_image_set)

    # novel_image: already provided by SDK as is_image_novel; rename for consistency
    stim_table['novel_image'] = stim_table['is_image_novel']

    stim_table = stim_table.merge(
        sessions_table[['ecephys_session_id', 'experience_level']],
        left_on='session_id', right_on='ecephys_session_id', how='left'
    ).drop(columns='ecephys_session_id')

    # Lick bout start: True on flashes where a lick bout begins after >2s gap
    lick_bouts = [False] * len(stim_table)
    last_lick_time = -2.0
    last_session = None
    for i, (_, flash) in enumerate(stim_table.iterrows()):
        if not np.isnan(flash['lick_time']):
            if flash['lick_time'] - last_lick_time > 2 or flash['session_id'] != last_session:
                lick_bouts[i] = True
                last_lick_time = flash['lick_time']
                last_session = flash['session_id']
    stim_table['lick_bout_start'] = lick_bouts

    # Response-window lick flags
    lick_offset = stim_table['lick_time'] - stim_table['start_time']
    in_window = (lick_offset > 0.1) & (lick_offset < 0.75)
    stim_table['lick_for_flash_during_response_window'] = stim_table['lick_for_flash'] & in_window
    stim_table['lickbout_for_flash_during_response_window'] = stim_table['lick_bout_start'] & in_window

    # Grace period: flashes after a hit, before the next trial begins
    stim_table['grace_period_after_hit'] = (
        stim_table['hit'].astype(bool) &
        (stim_table['flashes_since_change'] > 0) &
        stim_table['change_frame'].notna() &
        (stim_table['start_frame'] > stim_table['change_frame'])
    )

    # Flag sessions without abnormal activity or histology
    no_abnorm_ids = sessions_table[
        sessions_table['abnormal_activity'].isnull() &
        sessions_table['abnormal_histology'].isnull()
    ]['ecephys_session_id']
    stim_table['no_abnorm'] = stim_table['session_id'].isin(no_abnorm_ids)

    # is_sham_change: already provided by SDK in stimulus_presentations (no-op if present)
    if 'is_sham_change' not in stim_table.columns:
        is_sham = np.zeros(len(stim_table), dtype=bool)
        for _, group in stim_table.groupby('session_id'):
            catch_frames = group[group['catch'] == True]['change_frame'].unique()
            sham_mask = group['start_frame'].isin(catch_frames)
            is_sham[sham_mask[sham_mask].index] = True
        stim_table['is_sham_change'] = is_sham

    # change_eligible_window: flash >= 4 into a trial and not in grace period
    stim_table['change_eligible_window'] = (
        (stim_table['trial_flash'] >= 4) & ~stim_table['grace_period_after_hit']
    )

    # is_prechange: the flash immediately before each change flash
    stim_table['is_prechange'] = False
    change_inds = stim_table[stim_table['is_change']].index.values
    stim_table.loc[change_inds[change_inds > stim_table.index[0]] - 1, 'is_prechange'] = True

    return stim_table


def main():
    parser = argparse.ArgumentParser(description='Build master_stim_table_no_filter.csv from scratch.')
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Path to VBN S3 cache directory')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output path for master_stim_table_no_filter.csv')
    parser.add_argument('--session_ids', type=int, nargs='+', default=None,
                        help='Optional list of session IDs to process (default: all sessions)')
    args = parser.parse_args()

    cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(cache_dir=args.cache_dir)
    # Ensure we use the latest available manifest
    available = [m for m in cache.list_all_downloaded_manifests()
                 if m.startswith('visual-behavior-neuropixels_project_manifest')]
    latest = sorted(available)[-1]
    cache.load_manifest(latest)
    ecephys_session_table = cache.get_ecephys_session_table(filter_abnormalities=False)
    # The cache returns ecephys_session_id as the index; reset so it's a plain column
    sessions_table = ecephys_session_table.reset_index()

    session_ids = args.session_ids if args.session_ids else ecephys_session_table.index.values
    print(f'Building stim table for {len(session_ids)} sessions.')

    all_stim_tables = []
    for session_id in session_ids:
        print(f'  Processing session {session_id}...', flush=True)
        try:
            session = cache.get_ecephys_session(ecephys_session_id=session_id)
            session_stim = build_session_stim_table(session)
            all_stim_tables.append(session_stim)
        except Exception as e:
            print(f'  ERROR on session {session_id}: {e}')

    print('Concatenating sessions...')
    stim_table = pd.concat(all_stim_tables, ignore_index=True)

    print('Adding derived columns...')
    stim_table = add_derived_columns(stim_table, sessions_table)

    print(f'Saving to {args.output_file}...')
    stim_table.to_csv(args.output_file, index=False)
    print(f'Done. {len(stim_table):,} rows, {len(stim_table.columns)} columns.')


if __name__ == '__main__':
    main()
