import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
import argparse
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache

save_dir_metrics = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/unit_quality'

def get_session_metrics(session, save_location=save_dir_metrics):
    
    sess_id = session.metadata['ecephys_session_id']

    units = session.get_units()
    quality_df = units.loc[:, ['quality']]

    quality_df.to_csv(os.path.join(save_dir_metrics, str(sess_id) + '.csv'))


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
    get_session_metrics(
        session,
    )