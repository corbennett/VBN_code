# imports
import pandas as pd
from simple_slurm import Slurm
import os
from pathlib import Path
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache
import numpy as np

# define the conda environment
conda_environment = 'allensdk_38'

# script to run
script_path = '~/python_scripts/run_session_decoding.py'
print(f'running {script_path}')

# define the job record output folder
stdout_location = os.path.join(os.path.expanduser("~"), 'job_records')
# make the job record location if it doesn't already exist
os.mkdir(stdout_location) if not os.path.exists(stdout_location) else None

# define the plot save location
# this is where the python script will save plots
plot_save_location = os.path.join(os.path.expanduser("~"), 'sample_plots')

# build the python path
# this assumes that the environments are saved in the user's home directory in a folder called 'anaconda3'
# this will be user setup dependent.
python_path = os.path.join(
    os.path.expanduser("~"), 
    'miniconda3', 
    'envs', 
    conda_environment,
    'bin',
    'python'
)

#vbn_cache = '/allen/aibs/informatics/chris.morrison/ticket-27/allensdk_caches/vbn_cache_2022_Jul29/'
#vbn_cache = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_cache'
#vbn_cache = '/Volumes/programs/mindscope/workgroups/np-behavior/vbn_data_release/vbn_s3_cache'
vbn_cache = '/Volumes/programs/mindscope/workgroups/np-exp/vbn_data_release'
cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=vbn_cache)
# #cache.load_manifest('visual-behavior-neuropixels_project_manifest_v0.4.0.json')
ecephys_sessions_table = cache.get_ecephys_session_table(filter_abnormalities=True)

# instantiate a Slurm object
# slurm = Slurm(
#     cpus_per_task=1,
#     partition='braintv',
#     job_name='vbn_test',
#     output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
# )

slurm = Slurm(
    cpus_per_task=1,
    partition='braintv',
    job_name='run_session_metrics',
    output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    # output=f'{stdout_location}/log.out',
    #mail_type=['FAIL'],          # Mail events (NONE, BEGIN, END, FAIL, ALL)
    #mail_user='corbettb@alleninstitute.org',     # Where to send mail  
    time='24:00:00',
    mem_per_cpu='16gb',
)

run_table = ecephys_sessions_table
#run_table = ecephys_sessions_table[(~ecephys_sessions_table['abnormal_histology'].isnull())|(~ecephys_sessions_table['abnormal_activity'].isnull())]
# failed = [1065449881,
#             1067588044,
#             1081079981,
#             1089296550,
#             1093638203,
#             1093867806,
#             1095138995,
#             1118324999,
#             1119946360]
# run_table = ecephys_sessions_table.loc[failed]
# run_table = ecephys_sessions_table.loc[[1091039902]]

# call the `sbatch` command to run the jobs. We will get one job per desired frequency
# for session, row in run_table.iterrows():
#     slurm.sbatch('{} {} --session_id {} --cache_dir {}'.format(
#             python_path,
#             script_path,
#             session,
#             vbn_cache,
#         )
#     )

# call the `sbatch` command to run the jobs. We will get one job per desired frequency
# for session, row in run_table.iterrows():
#     slurm.sbatch('{} {} --session_id {}'.format(
#             python_path,
#             script_path,
#             session,
#         )
#     )

# sessions_to_run = [1044408432, 1044624428, 1048005547, 1048009327, 1048221709,
#        1048222325, 1049299003, 1049542142, 1052162536, 1052374521,
#        1052572359, 1053759573, 1053759575, 1053960984, 1053960987,
#        1055253879, 1055260435, 1055431030, 1055434752, 1062781531,
#        1063068136, 1064442478, 1064445631, 1064666428, 1064668541,
#        1065491938, 1065499125, 1065927708, 1065929713, 1067611872,
#        1067817036, 1069241166, 1069518782, 1071409007, 1081118392,
#        1081125370, 1081469485, 1081474898, 1086220230, 1086446431,
#        1087742174, 1088053452, 1089343256, 1090829963, 1090830527,
#        1091068323, 1091070296, 1092312238, 1092494311, 1093668878,
#        1093670602, 1093935326, 1093938328, 1095177109, 1095376453,
#        1096694612, 1096959372, 1098148095, 1099625106, 1099904879,
#        1104103368, 1104106044, 1104327081, 1104327579, 1105594809,
#        1105829335, 1108365127, 1108369066, 1108565586, 1108567373,
#        1109709062, 1109924339, 1111047108, 1111250074, 1112334683,
#        1112548552, 1115115784, 1115115786, 1115423017, 1115431348,
#        1116979796, 1117166453, 1118353478, 1118353987, 1118600316,
#        1118614693, 1119976479, 1120272974, 1121438973, 1121648067,
#        1122933470, 1123126929, 1124315985, 1124536942, 1125752903,
#        1125965838, 1128557546, 1128774624, 1130191879, 1130378962,
#        1139969640, 1152646832, 1152845816, 1153972744]
    

# call the `sbatch` command to run the jobs. We will get one job per desired frequency
# for session, row in run_table.iterrows():
# for session in sessions_to_run:
#     slurm.sbatch('{} {} --session_id {} --cache_dir {}'.format(
#             python_path,
#             script_path,
#             session,
#             vbn_cache,
#         )
#     )

#call the `sbatch` command to run the jobs. We will get one job per desired frequency
# for session, row in run_table.iterrows():
#     for to_decode in ['change',]:#['lick', 'change', 'image']:
#         slurm.sbatch('{} {} --session_id {} --cache_dir {} --to_decode {}'.format(
#                 python_path,
#                 script_path,
#                 session,
#                 vbn_cache,
#                 to_decode,
#             )
#         )


# areas_to_run = ['VISp','VISl', 'VISal', 'VISrl', 'VISam', 'VISpm', 'VISall',
#                 'LGd', 'LP', 'Hipp', 'SCMRN', 'midbrain', 'Sub']
# clusters_to_run = ['sensory', 'action', 'change', 'all']
# labels_to_run = ['lick', 'change', 'image', 'reaction_time']
# unitSampleSize_nPseudoFlashes_nUnitSamples_combos = ((20, 100, 100), 
#                                                     (50, 100, 100),
#                                                     (100, 100, 100),
#                                                    )
# areas_to_run = ['VISp','VISl', 'VISal', 'VISrl', 'VISam', 'VISpm', 'VISall',
#                 'LGd', 'LP', 'Hipp', 'SCMRN', 'midbrain', 'Sub']
# #areas_to_run = ['VISall',]
# clusters_to_run = ['transient', 'sustained']# 'action']#, 'change', 'all']
# labels_to_run = ['lick', 'change', 'image', 'reaction_time']
# unitSampleSize_nPseudoFlashes_nUnitSamples_combos = (
#                                                     (100, 100, 100),)
# for area in areas_to_run:
#     for clusters in clusters_to_run:
#         for label in labels_to_run:
#             for unitSampleSize_nPseudoFlashes_nUnitSamples in unitSampleSize_nPseudoFlashes_nUnitSamples_combos:
#                 slurm.sbatch('{} {} --label {} --region {} --cluster {} --unitSampleSize {} --nPseudoFlashes {} --nUnitSamples {}'.format(
#                     python_path,
#                     '/home/corbettb/python_scripts/decoding_utils.py',
#                     label,
#                     area,
#                     clusters,
#                     unitSampleSize_nPseudoFlashes_nUnitSamples[0],
#                     unitSampleSize_nPseudoFlashes_nUnitSamples[1],
#                     unitSampleSize_nPseudoFlashes_nUnitSamples[2],
#                     )
#                 )

# areas_to_run = ['VISp','VISl', 'VISal', 'VISrl', 'VISam', 'VISpm', 'VISall',
#                 'LGd', 'LP', 'Hipp', 'SCMRN', 'midbrain', 'Sub']
# clusters_to_run = ['sensory', 'action', 'change', 'all']
# labels_to_run = ['lick', 'change', 'image', 'reaction_time']
# unitSampleSize_nPseudoFlashes_nUnitSamples_combos = ((20, 100, 100), 
#                                                     (50, 100, 100),
#                                                     (100, 100, 100),
#                                                    )
# areas_to_run = ['all', 'VISall', 'SCMRN', 'midbrain', 'Hipp']
# areas_to_run = ['VISall', 'SCMRN', 'LGd', 'LP', 'Hipp']
# areas_to_run = ['VISall', 'SCMRN',]# 'LGd', 'LP', 'VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam']
# clusters_to_run = ['sensory', 'action']# 6] #[c for c in np.arange(1, 13)]# 'action']#, 'change', 'all']
# # clusters_to_run = ['action', 6] #[c for c in np.arange(1, 13)]# 'action']#, 'change', 'all']

# labels_to_run = ['lick', 'change', 'image',] #['visual_response', ]#['lick', 'change', 'image', 'reaction_time'] #['lick_imagematched',] #['change_eligible', 'lick', 'image', 'reaction_time']
# # labels_to_run = ['lick', 'reaction_time'] #['lick_imagematched',] #['change_eligible', 'lick', 'image', 'reaction_time']

# unitSampleSize_nPseudoFlashes_nUnitSamples_combos = (
#                                                     (100, 100, 1000),
#                                                    )

# for condition in ['active', 'passive']:
#     for area in areas_to_run:
#         for clusters in clusters_to_run:
#             for label in labels_to_run:
#                 for unitSampleSize_nPseudoFlashes_nUnitSamples in unitSampleSize_nPseudoFlashes_nUnitSamples_combos:
#                     if clusters=='sensory' and label not in ['lick', 'reaction_time']:
#                         continue
#                     if clusters=='action' and label not in ['image', 'change']:
#                         continue

#                     slurm.sbatch('{} {} --label {} --region {} --cluster {} --unitSampleSize {} --nPseudoFlashes {} --nUnitSamples {} --condition {}'.format(
#                         python_path,
#                         '/home/corbettb/python_scripts/decoding_utils.py',
#                         label,
#                         area,
#                         clusters,
#                         unitSampleSize_nPseudoFlashes_nUnitSamples[0],
#                         unitSampleSize_nPseudoFlashes_nUnitSamples[1],
#                         unitSampleSize_nPseudoFlashes_nUnitSamples[2],
#                         condition
#                         )
#                     )


# slurm.sbatch('{} {} --label {} --region1 {} --region2 {} --cluster {} --unitSampleSize {} --nPseudoFlashes {} --nUnitSamples {}'.format(
#                     python_path,
#                     '/home/corbettb/python_scripts/decoding_utils.py',
#                     'change',
#                     'VISp',
#                     'LP',
#                     'sensory',
#                     100,
#                     100,
#                     100,
#                     )
# )

# Run decoder dropouts
# # unit_set_regions = ['VISall','VISall', 'VISall', 'VISall'] + ['VISall','VISall', 'VISall', 'VISall']
# # unit_set_layers = ['all', 'all', 'all', 'all'] + ['all', 'all', 'all', 'all']
# # unit_set_cell_types = ['all', 'all', 'all', 'all'] + ['all', 'all', 'all', 'all']
# # unit_set_clusters = ['sensory', 'sensory', 'sensory', 'sensory'] + ['sensory', 'sensory', 'sensory', 'sensory']
# # unit_subset_regions = ['VISall','VISall', 'VISall', 'VISall'] + ['VISall','VISall', 'VISall', 'VISall']
# # unit_subset_layers = ['2/3', '4', '5', '6',] + ['all', 'all', 'all', 'all']
# # unit_subset_cell_types = ['all', 'all', 'all', 'all'] + ['RS', 'FS', 'SST', 'VIP']
# # unit_subset_clusters = ['sensory', 'sensory', 'sensory', 'sensory'] + ['sensory', 'sensory', 'sensory', 'sensory']

# # unit_set_regions = ['VISall',] + ['VISall','VISall', 'VISall', 'VISall']
# # unit_set_layers = ['all',] + ['all', 'all', 'all', 'all']
# # unit_set_cell_types = ['all',] + ['all', 'all', 'all', 'all']
# # unit_set_clusters = ['sensory',] + ['sensory', 'sensory', 'sensory', 'sensory']
# # unit_subset_regions = ['VISall',] + ['VISall','VISall', 'VISall', 'VISall']
# # unit_subset_layers = ['6',] + ['2/3', '4', '5', '6',]
# # unit_subset_cell_types = ['all',] + ['RS', 'RS', 'RS', 'RS']
# # unit_subset_clusters = ['sensory',] + ['sensory', 'sensory', 'sensory', 'sensory']

# unit_set1_regions = ['VISall',] * 2
# unit_set1_layers = ['all',] * 2
# unit_set1_cell_types = ['all',]* 2
# unit_set1_clusters = ['sensory',]* 2
# unit_set2_regions = ['VISall',]* 2
# unit_set2_layers = ['all',]* 2
# unit_set2_cell_types = ['VIP',]* 2
# unit_set2_clusters = ['sensory',]* 2
# experiences = ['Familiar', 'Novel']
# unit_sample_sizes = [40, 40]

# # unit_set1_regions = ['VISall',] * 8
# # unit_set1_layers = ['all',] * 8
# # unit_set1_cell_types = ['all',] * 8
# # unit_set1_clusters = ['sensory',] * 8
# # unit_set2_regions = ['VISall',] * 6 + ['VISp_VISl_VISal', 'VISrl_VISpm_VISam',]
# # unit_set2_layers = ['2/3', '4', '5', '6', 'all', 'all'] + ['all', 'all']
# # unit_set2_cell_types = ['RS', 'RS', 'RS', 'RS', 'FS', 'SST'] + ['SST', 'SST']
# # unit_set2_clusters = ['sensory',] * 8
# # experiences = ['Novel',] * 8
# # unit_sample_sizes = [40,] * 8

# for unit_set1_region, unit_set1_layer, unit_set1_cell_type, unit_set1_cluster, unit_set2_region, unit_set2_layer, unit_set2_cell_type, unit_set2_cluster, experience, unit_sample_size in zip(
#     unit_set1_regions,
#     unit_set1_layers,
#     unit_set1_cell_types,
#     unit_set1_clusters,
#     unit_set2_regions,
#     unit_set2_layers,
#     unit_set2_cell_types,
#     unit_set2_clusters,
#     experiences,
#     unit_sample_sizes
# ):

#     slurm.sbatch('{} {} --unit_set1_region {} --unit_set1_layer {} --unit_set1_cell_type {} --unit_set1_cluster {} --unit_set2_region {} --unit_set2_layer {} --unit_set2_cell_type {} --unit_set2_cluster {} --experience {} --unit_sample_size {}'.format(
#                             python_path,
#                             '/home/corbettb/python_scripts/run_decoding_dropouts.py',
#                             unit_set1_region,
#                             unit_set1_layer,
#                             unit_set1_cell_type,
#                             unit_set1_cluster,
#                             unit_set2_region,
#                             unit_set2_layer,
#                             unit_set2_cell_type,
#                             unit_set2_cluster,
#                             experience,
#                             unit_sample_size
#                             )
#                 )
## Run decoder area comparison monte carlo null estimation

unit_set1_regions = ['VISall',]
unit_set1_layers = ['all',]
unit_set1_cell_types = ['all',]
unit_set1_clusters = ['action',]
unit_set2_regions = ['SCMRN',]
unit_set2_layers = ['all',]
unit_set2_cell_types = ['all',]
unit_set2_clusters = ['action',]
experiences = ['all',]
unit_sample_sizes = [100,]
decoding_labels = ['lick',]

# num_iterations = 100

for iteration in range(1000):

    for unit_set1_region, unit_set1_layer, unit_set1_cell_type, unit_set1_cluster, unit_set2_region, unit_set2_layer, unit_set2_cell_type, unit_set2_cluster, experience, unit_sample_size, decoding_label in zip(
        unit_set1_regions,
        unit_set1_layers,
        unit_set1_cell_types,
        unit_set1_clusters,
        unit_set2_regions,
        unit_set2_layers,
        unit_set2_cell_types,
        unit_set2_clusters,
        experiences,
        unit_sample_sizes,
        decoding_labels
    ):

        slurm.sbatch('{} {} --unit_set1_region {} --unit_set1_layer {} --unit_set1_cell_type {} --unit_set1_cluster {} --unit_set2_region {} --unit_set2_layer {} --unit_set2_cell_type {} --unit_set2_cluster {} --experience {} --unit_sample_size {} --iteration {} --decoding_label {}'.format(
                                python_path,
                                '/home/corbettb/python_scripts/run_decoder_area_comparison_monte_carlo.py',
                                unit_set1_region,
                                unit_set1_layer,
                                unit_set1_cell_type,
                                unit_set1_cluster,
                                unit_set2_region,
                                unit_set2_layer,
                                unit_set2_cell_type,
                                unit_set2_cluster,
                                experience,
                                unit_sample_size,
                                iteration,
                                decoding_label
                                )
                    )

## Run GLM prediction psths
# conda_environment = 'allensdk_glm'
# python_path = os.path.join(
#     os.path.expanduser("~"), 
#     'miniconda3', 
#     'envs', 
#     conda_environment,
#     'bin',
#     'python'
# )
# script_path = '~/python_scripts/run_GLM_prediction_psths.py'

# # call the `sbatch` command to run the jobs. We will get one job per desired frequency

# for session, row in run_table.iterrows():
#     slurm.sbatch('{} {} --session_id {}'.format(
#             python_path,
#             script_path,
#             session,
#         )
#     )