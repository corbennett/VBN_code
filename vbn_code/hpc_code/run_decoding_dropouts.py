import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os, glob
import h5py
import argparse
import decoding_utils as du

outputDir = '/Volumes/programs/mindscope/workgroups/np-behavior/VBN_revision_decoding_dropouts'

def unit_subset_dropout_decoding(unit_set_ids, unit_subset_ids, unit_subset_name, unitSampleSize = 100, nPseudoFlashes=100, nUnitSamples=100, experience='all'):
    '''
    unit_set_ids: list of unit ids to sample from
    unit_subset: list of unit ids to exclude from sampling (for dropout) AND to exclusively use for sufficiency test

    Loop above this function should define different unit subsets to test AND run full models to compare to dropouts
    '''

    # run dropout decoding (exclude unit_subset from sampling)
    unit_ids_dropout = np.setdiff1d(unit_set_ids, unit_subset_ids)
    du.pooledDecoding_unit_subsets('change', unit_ids_dropout, unit_subset_name + '_dropout', unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience=experience)
    du.pooledDecoding_unit_subsets('image', unit_ids_dropout, unit_subset_name + '_dropout', unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience=experience)


    # run sufficiency decoding (only use unit_subset for sampling)
    du.pooledDecoding_unit_subsets('change', unit_subset_ids, unit_subset_name + '_sufficiency', unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience=experience, use_max_sample_size_available=True)
    du.pooledDecoding_unit_subsets('image', unit_subset_ids, unit_subset_name + '_sufficiency', unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience=experience, use_max_sample_size_available=True)


def run_subset_decoding(unit_set_region='all', unit_set_layer='all', unit_set_cell_type='all', unit_set_cluster=('all',),
                        unit_subset_region='all', unit_subset_layer='all', unit_subset_cell_type='all', unit_subset_cluster=('all',),
                        unitSampleSize = 100, nPseudoFlashes=100, nUnitSamples=100, experience='all'):
    '''
    Define unit set and unit subset based on region/layer/cell type filters
    '''

    # Load unit table
    unit_table_file = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/master_units_with_responsiveness.csv'
    units = pd.read_csv(unit_table_file)
    units = units.set_index('unit_id')
    units['cortical_layer'].replace('3-Feb', '2/3', inplace=True)

    def get_clusters_from_cluster_string(cluster_string):
        if cluster_string == 'all':
            return np.arange(13)
        elif cluster_string == 'sensory':
            return np.arange(6)
        elif cluster_string == 'action':
            return [6,7,9,10,11,12]
        elif cluster_string=='change':
            return [6,]
        elif cluster_string =='transient':
            return [1,]
        elif cluster_string in [str(i) for i in range(13)]:
            return [int(cluster_string),]
        else:
            raise ValueError(f'Unknown cluster string: {cluster_string}')

    # Define unit set
    inRegion = du.getUnitsInRegion(units, unit_set_region, cell_type=unit_set_cell_type, layer=unit_set_layer)
    highQuality = du.apply_unit_quality_filter(units, no_abnorm=True)
    inCluster = du.get_units_in_cluster(units, *get_clusters_from_cluster_string(unit_set_cluster), clustering='new')
    unit_set_ids = units.index[inRegion & highQuality & inCluster].values
    print(f'num units in set: {len(unit_set_ids)}')

    # Define unit subset
    inRegion = du.getUnitsInRegion(units, unit_subset_region, cell_type=unit_subset_cell_type, layer=unit_subset_layer)
    highQuality = du.apply_unit_quality_filter(units, no_abnorm=True)
    inCluster = du.get_units_in_cluster(units, *get_clusters_from_cluster_string(unit_subset_cluster), clustering='new')
    unit_subset_ids = units.index[inRegion & highQuality & inCluster].values
    print(f'num units in subset: {len(unit_subset_ids)}')

    if len(unit_subset_ids) < 1:
        print('No units in subset, skipping...')
        return

    # Before running dropouts, run full model for comparison
    unit_set_layer_str = unit_set_layer.replace('/', '')
    unit_subset_layer_str = unit_subset_layer.replace('/', '')

    # Check to see if full model has already been run
    full_model_run = False
    full_model_substring = f"set_{unit_set_region}_{unit_set_layer_str}_{unit_set_cell_type}_{unit_set_cluster}_{experience}_full"
    for root, dirs, files in os.walk(outputDir):
        for fname in files:
            if full_model_substring in fname and fname.split('_')[-4]==str(unitSampleSize):
                full_model_run = True
                break
    
    if not full_model_run:
        print('Running full model for comparison...')
        du.pooledDecoding_unit_subsets('change', unit_set_ids, f"set_{unit_set_region}_{unit_set_layer_str}_{unit_set_cell_type}_{unit_set_cluster}_{experience}_full", unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience=experience)
        du.pooledDecoding_unit_subsets('image', unit_set_ids, f"set_{unit_set_region}_{unit_set_layer_str}_{unit_set_cell_type}_{unit_set_cluster}_{experience}_full", unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience=experience)
    else:
        print('Full model already run, skipping...')

    # Run dropout and sufficiency decoding
    unit_subset_dropout_decoding(unit_set_ids, unit_subset_ids, 
                                 f"set_{unit_set_region}_{unit_set_layer_str}_{unit_set_cell_type}_{unit_set_cluster}_{experience}_subset_{unit_subset_region}_{unit_subset_layer_str}_{unit_subset_cell_type}_{unit_subset_cluster}_{experience}",
                                 unitSampleSize, nPseudoFlashes, nUnitSamples, experience=experience)

if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit_set_region', type=str)
    parser.add_argument('--unit_set_layer', type=str)
    parser.add_argument('--unit_set_cell_type', type=str)
    parser.add_argument('--unit_set_cluster', type=str)
    parser.add_argument('--unit_subset_region', type=str)
    parser.add_argument('--unit_subset_layer', type=str)
    parser.add_argument('--unit_subset_cell_type', type=str)
    parser.add_argument('--unit_subset_cluster', type=str)
    parser.add_argument('--experience', type=str)
    parser.add_argument('--unit_sample_size', type=int, default=100)
    args = parser.parse_args()
    
    print('Running subset decoding with args:')
    print(args)

    run_subset_decoding(
        unit_set_region=args.unit_set_region,
        unit_set_layer=args.unit_set_layer,
        unit_set_cell_type=args.unit_set_cell_type,
        unit_set_cluster=args.unit_set_cluster,
        unit_subset_region=args.unit_subset_region,
        unit_subset_layer=args.unit_subset_layer,
        unit_subset_cell_type=args.unit_subset_cell_type,
        unit_subset_cluster=args.unit_subset_cluster,
        experience=args.experience,
        unitSampleSize=args.unit_sample_size,
    )