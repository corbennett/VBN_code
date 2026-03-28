import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os, glob
import h5py
import argparse
import decoding_utils as du

outputDir = '/Volumes/programs/mindscope/workgroups/np-behavior/VBN_revision_decoder_area_comparison_nulls'

def run_monte_carlo_null_estimation(unit_set1_region='all', unit_set1_layer='all', unit_set1_cell_type='all', unit_set1_cluster=('all',),
                        unit_set2_region='all', unit_set2_layer='all', unit_set2_cell_type='all', unit_set2_cluster=('all',),
                        unitSampleSize = 100, nPseudoFlashes=100, nUnitSamples=1000, experience='all', iteration=0, decoding_label='change'):
    '''
    Define unit set and unit set2 based on region/layer/cell type filters

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

    # Define unit set1
    inRegion = du.getUnitsInRegion(units, unit_set1_region, cell_type=unit_set1_cell_type, layer=unit_set1_layer)
    highQuality = du.apply_unit_quality_filter(units, no_abnorm=True)
    inCluster = du.get_units_in_cluster(units, *get_clusters_from_cluster_string(unit_set1_cluster), clustering='new')
    unit_set1_ids = units.index[inRegion & highQuality & inCluster].values
    print(f'num units in set: {len(unit_set1_ids)}')

    # Define unit set2
    inRegion = du.getUnitsInRegion(units, unit_set2_region, cell_type=unit_set2_cell_type, layer=unit_set2_layer)
    highQuality = du.apply_unit_quality_filter(units, no_abnorm=True)
    inCluster = du.get_units_in_cluster(units, *get_clusters_from_cluster_string(unit_set2_cluster), clustering='new')
    unit_set2_ids = units.index[inRegion & highQuality & inCluster].values
    print(f'num units in set2: {len(unit_set2_ids)}')

    if (len(unit_set2_ids) < 1) or (len(unit_set1_ids) < 1):
        print('No units in set1 or set2, skipping...')
        return

    unit_set1_layer_str = unit_set1_layer.replace('/', '')
    unit_set2_layer_str = unit_set2_layer.replace('/', '')

    concatenated_unit_set_ids = np.concatenate([unit_set1_ids, unit_set2_ids])
    np.random.shuffle(concatenated_unit_set_ids)
    #Decode from permuted list of units from both sets (this will be the null distribution to compare to the actual decoding results where we decode from each set separately)
    #Preserve difference in number of units between the two sets
    du.pooledDecoding_unit_subsets(decoding_label, concatenated_unit_set_ids[:len(unit_set1_ids)], f"set1_{unit_set1_region}_{unit_set1_layer_str}_{unit_set1_cell_type}_{unit_set1_cluster}_set2_{unit_set2_region}_{unit_set2_layer_str}_{unit_set2_cell_type}_{unit_set2_cluster}_{experience}_null_pop1_{iteration}",
                                   unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience=experience, use_max_sample_size_available=False, outputDir=outputDir)
    du.pooledDecoding_unit_subsets(decoding_label, concatenated_unit_set_ids[len(unit_set1_ids):], f"set1_{unit_set1_region}_{unit_set1_layer_str}_{unit_set1_cell_type}_{unit_set1_cluster}_set2_{unit_set2_region}_{unit_set2_layer_str}_{unit_set2_cell_type}_{unit_set2_cluster}_{experience}_null_pop2_{iteration}",
                                   unitSampleSize, nPseudoFlashes, nUnitSamples, condition='active', experience=experience, use_max_sample_size_available=False, outputDir=outputDir)


if __name__ == "__main__":
    # define args
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit_set1_region', type=str)
    parser.add_argument('--unit_set1_layer', type=str)
    parser.add_argument('--unit_set1_cell_type', type=str)
    parser.add_argument('--unit_set1_cluster', type=str)
    parser.add_argument('--unit_set2_region', type=str)
    parser.add_argument('--unit_set2_layer', type=str)
    parser.add_argument('--unit_set2_cell_type', type=str)
    parser.add_argument('--unit_set2_cluster', type=str)
    parser.add_argument('--experience', type=str)
    parser.add_argument('--unit_sample_size', type=int, default=100)
    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--decoding_label', type=str, default='change')
    args = parser.parse_args()
    
    print('Running subset decoding with args:')
    print(args)

    run_monte_carlo_null_estimation(
        unit_set1_region=args.unit_set1_region,
        unit_set1_layer=args.unit_set1_layer,
        unit_set1_cell_type=args.unit_set1_cell_type,
        unit_set1_cluster=args.unit_set1_cluster,
        unit_set2_region=args.unit_set2_region,
        unit_set2_layer=args.unit_set2_layer,
        unit_set2_cell_type=args.unit_set2_cell_type,
        unit_set2_cluster=args.unit_set2_cluster,
        experience=args.experience,
        unitSampleSize=args.unit_sample_size,
        iteration = args.iteration,
        decoding_label=args.decoding_label
    )