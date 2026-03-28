import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import argparse
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
from brain_observatory_utilities.datasets.electrophysiology.\
    receptive_field_mapping import ReceptiveFieldMapping_VBN

save_dir = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/rfs'

def run_rfs(session, save_location=save_dir):
    '''
    a function that gets RF metrics for every unit
    and saves results out to csv file
    '''
    rf = ReceptiveFieldMapping_VBN(session)
    rf_metrics = rf.metrics

    sess_id = session.metadata['ecephys_session_id']
    save_path = os.path.join(save_location, str(sess_id) + '.csv')
    rf_metrics.to_csv(save_path)


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
        session
    )