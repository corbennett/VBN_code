import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import h5py
import argparse
import decoding_utils as du

save_dir = '/Volumes/programs/mindscope/workgroups/np-behavior/VBN_decoding_From_sensory_action_clusters'

def session_decoding(session_id, to_decode, save_location=save_dir):
    
    regions = ('LP', 'VISp')#('VISall','VISp','VISl','VISrl','VISal','VISpm','VISam','LP', 'LGd',
            #'SCMRN','MB','Hipp','Sub')

    active_tensor_file = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/vbnAllUnitSpikeTensor.hdf5'
    unitData = h5py.File(active_tensor_file, 'r')

    stim_table_file = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv'
    stimTable = pd.read_csv(stim_table_file)

    unit_table_file = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_units_with_responsiveness.csv'
    unitTable = pd.read_csv(unit_table_file)

    unitSampleSize = [5,]
    
    decodeWindowEnd = 750

    # d = du.decodeImage(session_id, unitTable, unitData, stimTable, regions, unitSampleSize, decodeWindowEnd, use_nonchange=True, class_weight='balanced')
    # np.save(os.path.join(save_dir, str(session_id)+'_nonchange.npy'), d)
    for cluster in ['sensory',]:# 'action', 'all']:
        du.sessionDecoding(session_id, to_decode, cluster, unitTable, unitData, stimTable, regions, unitSampleSize, decodeWindowEnd, 
                    class_weight='balanced', rs=False)
        



if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=int)
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--to_decode', type=str)
    args = parser.parse_args()
    

    session_decoding(
        args.session_id, args.to_decode
    )