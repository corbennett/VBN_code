import numpy as np
import pandas as pd
import os, glob
import argparse
import _pickle as cPickle
import bz2
from scipy.interpolate import interp1d

outputDir ="/Volumes/programs/mindscope/workgroups/np-behavior/VBN_revision_glm_prediction_psths"
stim_table_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
stim_table = pd.read_csv(stim_table_file)

def resample_df_to_times(df, time_column, val_column, new_times):

    timestamps = df[time_column].values
    vals = df[val_column].values
    interpolator = interp1d(timestamps, vals, kind='linear', bounds_error=False)#, fill_value=np.nan)
    new_values = interpolator(new_times)

    return new_values, new_times

def run_glm_prediction_psths(session_id, save_location=outputDir):
    
    filename = "/Volumes/programs/braintv/workgroups/nc-ophys/alex.piet/NP/ephys/v_108_active/experiment_model_files/with_predictions/" + str(session_id) + ".pbz2"
    
    with bz2.BZ2File(filename, 'rb') as f:
        fit = cPickle.load(f)

    unit_ids = fit['spike_count_arr'].unit_id.values

    session_stims = stim_table[stim_table['session_id']==int(session_id)]

    hits = (
        session_stims['hit'] &
        session_stims['is_change']
    )

    misses = (
        session_stims['miss'] &
        session_stims['is_change']
    )

    nonchange_licks = (
        (~session_stims['is_change']) &
        (~session_stims['omitted']) &
        (~session_stims['previous_omitted']) &
        (session_stims['flashes_since_change']>5) &
        (session_stims['flashes_since_last_lick']>1) &
        (session_stims['lickbout_for_flash_during_response_window'])
    )

    nonchange_nolicks = (
        (~session_stims['is_change']) &
        (~session_stims['omitted']) &
        (~session_stims['previous_omitted']) &
        (session_stims['flashes_since_change']>5) &
        (session_stims['flashes_since_last_lick']>1) &
        (~session_stims['lickbout_for_flash_during_response_window'])
    )

    if any([f.sum()<5 for f in [hits, misses, nonchange_licks, nonchange_nolicks]]):
        print(f'Not enough trials in one of the conditions, skipping session {session_id}...')
        return

    for unit_index, unit_id in enumerate(unit_ids):
        fit_df = pd.DataFrame({'prediction': fit['full_model_prediction'][:, unit_index], 'activity': fit['spike_count_arr'][:, unit_index].values, 'time': fit['bin_centers']})
        unit_predictions = []
        unit_predictions_sem = []
        unit_psths = []
        unit_psths_sem = []
        for filter in [hits, misses, nonchange_licks, nonchange_nolicks]:

            condition_starttimes = session_stims.loc[filter]['start_time'].values
            condition_psth = []
            condition_predicted_psth = []
            for start_time in condition_starttimes:
                psth, _ = resample_df_to_times(fit_df, 'time', 'activity', np.arange(start_time-0.25, start_time+1, 0.025))
                predicted_psth, _  = resample_df_to_times(fit_df, 'time', 'prediction', np.arange(start_time-0.25, start_time+1, 0.025))

                condition_psth.append(psth[:40])
                condition_predicted_psth.append(predicted_psth[:40])
            
            unit_predictions.append(np.mean(condition_predicted_psth, axis=0))
            unit_predictions_sem.append(np.std(condition_predicted_psth, axis=0)/np.sqrt(len(condition_predicted_psth)))
            unit_psths.append(np.mean(condition_psth, axis=0))
            unit_psths_sem.append(np.std(condition_psth, axis=0)/np.sqrt(len(condition_psth)))
        
        unit_predictions = np.concatenate(unit_predictions)
        unit_psths = np.concatenate(unit_psths)
        unit_predictions_sem = np.concatenate(unit_predictions_sem)
        unit_psths_sem = np.concatenate(unit_psths_sem)
        np.savez(os.path.join(save_location, f'{unit_id}_{session_id}.npz'), **{'prediction': unit_predictions, 'psth': unit_psths, 'prediction_sem':unit_predictions_sem, 'psth_sem': unit_psths_sem, 'trial_counts': [hits.sum(), misses.sum(), nonchange_licks.sum(), nonchange_nolicks.sum()]})
        


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=str)
    args = parser.parse_args()
    
    print('Running GLM prediction psths:')
    print(args)

    run_glm_prediction_psths(
        session_id = args.session_id,
    )