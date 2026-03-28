import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import h5py
import argparse
import decoding_utils as du

save_dir = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/image_decoding_sliding_window'

def image_decoding(session_id, save_location=save_dir):
    
    regions = ('VISall',)#'VISp','VISl','VISrl','VISal','VISpm','VISam','LP', 'LGd', 'SCMRN', 'Hipp')

    active_tensor_file = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/vbnAllUnitSpikeTensor.hdf5'
    unitData = h5py.File(active_tensor_file, 'r')

    stim_table_file = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_stim_table_no_filter.csv'
    stimTable = pd.read_csv(stim_table_file)

    unit_table_file = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_units_with_responsiveness.csv'
    unitTable = pd.read_csv(unit_table_file)

    unitSampleSize = [80,]#[10,20,40,80] #run with 80 for sup fig 11 in paper (cluster = 'all', rs=True)
    
    unitSampleSize = [40,] # reduce sample size for sustained cluster

    decodeWindowSize = 50
    decodeWindowBinSize = 10
    decodeWindowSlidingStep = 20
    decodeWindowEnd = 1500#750

    # d = du.decodeImage(session_id, unitTable, unitData, stimTable, regions, unitSampleSize, decodeWindowEnd, use_nonchange=True, class_weight='balanced')
    # np.save(os.path.join(save_dir, str(session_id)+'_nonchange.npy'), d)

    # d = du.decodeImage(session_id, unitTable, unitData, stimTable, regions, unitSampleSize, decodeWindowEnd, 
    #                    decode_full_timecourse_index=4, use_nonchange=False, class_weight='balanced', rs=True)
    
    clusters = [['all',]] #[[1,], [2,], [3,], [4,], [5,]]
    clusters = [[2,4],]
    for cluster in clusters:
        d = du.decodeImageSlidingWindow(session_id, unitTable, unitData, stimTable, 
                                regions, unitSampleSize, decodeWindowEnd, decodeWindowSize, decodeWindowBinSize, decodeWindowSlidingStep,
                                use_nonchange=False, through_omission=True, class_weight=None, rs=True, cluster=cluster,
                                )
        if cluster == [2,4]:
            name = 'sustained'
        else:
            name = cluster[0]
            
        np.save(os.path.join(save_dir, f'{session_id}_throughomissionRS_{name}.npy'), d)



if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=int)
    parser.add_argument('--cache_dir', type=str)
    args = parser.parse_args()
    

    image_decoding(
        args.session_id,
    )