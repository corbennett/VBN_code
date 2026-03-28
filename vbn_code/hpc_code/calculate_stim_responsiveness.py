import h5py
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
from tensor_utils import get_tensor_unit_table, get_tensor_for_unit_selection
from utilities import get_change_flashes, get_nonchange_flashes, findResponsiveUnits_nopeak

save_dir = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/vis_responsiveness"

#Paths to all of the useful supplemental tables and tensors
active_tensor_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/vbnAllUnitSpikeTensor.hdf5"
passive_tensor_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/vbnAllUnitSpikeTensor_passive.hdf5"
opto_tensor_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/vbn_opto_tensor_unit_grouped.hdf5"

stim_table_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
unit_table_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_unit_table.csv"
unit_table_with_rf_stats = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/units_with_rf_stats.csv"
unit_table_opto_metrics = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/unit_opto_metrics.csv"

units = pd.read_csv(unit_table_file)
stim_table = pd.read_csv(stim_table_file)
stim_table = stim_table.drop(columns='Unnamed: 0') #drop redundant column

active_tensor = h5py.File(active_tensor_file)

good_unit_filter = [(units['isi_violations']<1) &
                    #(units['presence_ratio']>0.9) &
                    #(units['amplitude_cutoff']<0.1) &
                    (units['quality']=='good') &
                    (units['snr']>1)&
                    #(units['no_anomalies'])&
                    ([True]*len(units))][0]


good_units = units[good_unit_filter]
good_unit_ids = good_units['unit_id'].values

g_images = ['omitted'] + list(np.sort(stim_table[(stim_table['stimulus_name'].str.contains('_G_'))&
                    (~stim_table['omitted'])&
                    (~stim_table['image_name'].isin(['im083_r','im111_r']))]['image_name'].unique())) + ['im083_r','im111_r']

h_images = ['omitted'] + list(np.sort(stim_table[(stim_table['stimulus_name'].str.contains('_H_'))&
                    (~stim_table['omitted'])&
                    (~stim_table['image_name'].isin(['im083_r','im111_r']))]['image_name'].unique())) + ['im083_r','im111_r']

for session_ind, session_id in enumerate(list(active_tensor.keys())):
    
    session_stim_table = stim_table[stim_table['session_id']==int(session_id)].reset_index()
    session_tensor = active_tensor[str(session_id)]
    session_units = get_tensor_unit_table(units, session_tensor['unitIds'][()])
    good_unit_tensor = get_tensor_for_unit_selection(session_units.index.values, session_tensor['spikes'])

    image_set = g_images if '_G_' in session_stim_table['stimulus_name'].iloc[0] else h_images

    for image in image_set:
        filter_stims = get_nonchange_flashes(session_stim_table, image_id=image)
        stim_sp = good_unit_tensor[:, filter_stims, :]
        pre_stim_sp = good_unit_tensor[:, filter_stims-1, :]
        responsive, pos_mod, mean_evoked, peak_evoked = findResponsiveUnits_nopeak(pre_stim_sp, stim_sp, baseWin = slice(670,750), respWin = slice(20,100))
        session_units[image + '_nonchange_response_pval'] = responsive
        session_units[image + '_nonchange_positive_modulation'] = pos_mod
        session_units[image + '_nonchange_mean_evoked'] = mean_evoked
        session_units[image + '_nonchange_peak_evoked'] = peak_evoked


    filter_stims = get_change_flashes(session_stim_table)
    stim_sp = good_unit_tensor[:, filter_stims, :]
    pre_stim_sp = good_unit_tensor[:, filter_stims-1, :]
    responsive, pos_mod, mean_evoked, peak_evoked = findResponsiveUnits_nopeak(pre_stim_sp, stim_sp, baseWin = slice(670,750), respWin = slice(20,100))
    session_units['change_response_pval'] = responsive
    session_units['change_positive_modulation'] = pos_mod
    session_units['change_mean_evoked'] = mean_evoked
    session_units['change_peak_evoked'] = peak_evoked
    
    pval_cols = [c for c in session_units.columns if 'pval' in c]
    pval_cols_corrected = [c+'_corrected' for c in pval_cols]
    is_sig_cols = [c + '_sig' for c in pval_cols]

    session_units[pval_cols_corrected] = 1
    session_units[is_sig_cols] = False

    for ind, row in session_units.iterrows():
        row_pvals = row[pval_cols].values
        sig, corrected = fdrcorrection(row_pvals)
        session_units.loc[ind, pval_cols_corrected] = corrected
        session_units.loc[ind, is_sig_cols] = sig

    session_units.to_csv(os.path.join(save_dir, f'{session_id}_20_to_100_decision_window.csv'))

active_tensor.close()