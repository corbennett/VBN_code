import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import argparse
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
from brain_observatory_utilities.datasets.electrophysiology.\
    receptive_field_mapping import ReceptiveFieldMapping_VBN

rf_metrics_dir = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/rfs'
save_dir = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/rfs/arrays'

def run_rfs(session, save_location=save_dir):
    '''
    a function that gets RF metrics for every unit
    and saves results out to csv file
    '''
    rf = ReceptiveFieldMapping_VBN(session)
    # read the metrics we already generated
    sess_id = session.metadata['ecephys_session_id']
    rf_metrics = pd.read_csv(os.path.join(rf_metrics_dir, str(sess_id) +'.csv'))

    good_rfs = rf_metrics[(rf_metrics.p_value_rf<0.001)&rf_metrics.on_screen_rf]
    good_rfs = good_rfs.set_index('unit_id')

    for iu, unit in good_rfs.iterrows():
        urf = rf.get_receptive_field(iu)
        np.save(os.path.join(save_dir, str(iu)+'.npy'), urf)
    
    # save_path = os.path.join(save_location, str(sess_id) + '.csv')
    # rf_metrics.to_csv(save_path)


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id', type=int)
    parser.add_argument('--cache_dir', type=str)
    args = parser.parse_args()
    
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=args.cache_dir)

    session = cache.get_ecephys_session(
           ecephys_session_id=args.session_id)
    # call the plotting function
    run_rfs(
        session,
    )