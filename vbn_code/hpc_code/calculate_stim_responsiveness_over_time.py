import h5py
import pandas as pd
import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
from tensor_utils import get_tensor_unit_table, get_tensor_for_unit_selection
from utilities import get_change_flashes, get_nonchange_flashes, findResponsiveUnits_nopeak, findResponsiveUnits_overtime

save_dir = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/vis_responsiveness_over_time"


def calculate_responsiveness_over_time(session_id):
    #Paths to all of the useful supplemental tables and tensors
    active_tensor_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/vbnAllUnitSpikeTensor.hdf5"
    
    stim_table_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv"
    unit_table_file = "/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_unit_table.csv"
    
    units = pd.read_csv(unit_table_file)
    stim_table = pd.read_csv(stim_table_file)
    stim_table = stim_table.drop(columns='Unnamed: 0') #drop redundant column

    active_tensor = h5py.File(active_tensor_file)

    g_images = ['omitted'] + list(np.sort(stim_table[(stim_table['stimulus_name'].str.contains('_G_'))&
                        (~stim_table['omitted'])&
                        (~stim_table['image_name'].isin(['im083_r','im111_r']))]['image_name'].unique())) + ['im083_r','im111_r']

    h_images = ['omitted'] + list(np.sort(stim_table[(stim_table['stimulus_name'].str.contains('_H_'))&
                        (~stim_table['omitted'])&
                        (~stim_table['image_name'].isin(['im083_r','im111_r']))]['image_name'].unique())) + ['im083_r','im111_r']


    session_stim_table = stim_table[stim_table['session_id']==int(session_id)].reset_index()
    session_tensor = active_tensor[str(session_id)]
    session_units = get_tensor_unit_table(units, session_tensor['unitIds'][()])
    good_unit_tensor = get_tensor_for_unit_selection(session_units.index.values, session_tensor['spikes'])

    image_set = g_images if '_G_' in session_stim_table['stimulus_name'].iloc[0] else h_images
    window_size = 40
    data_dict = {'unit_ids':session_units['unit_id'].values}
    for image in image_set:
        filter_stims = get_nonchange_flashes(session_stim_table, image_id=image)
        stim_sp = good_unit_tensor[:, filter_stims, :]
        pre_stim_sp = good_unit_tensor[:, filter_stims-1, :]
        pos_resp_pval = findResponsiveUnits_overtime(pre_stim_sp, stim_sp, window_duration=window_size)

        data_dict[image] = pos_resp_pval

    np.save(os.path.join(save_dir, str(session_id) + '_winsize' + str(window_size)), data_dict)

    active_tensor.close()

if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=int)
    parser.add_argument('--cache_dir', type=str)
    args = parser.parse_args()
    

    calculate_responsiveness_over_time(
        args.session_id,
    )