
# imports
import pandas as pd
from simple_slurm import Slurm
import os, glob
from pathlib import Path
import numpy as np

# define the conda environment
conda_environment = 'allensdk_38'

# script to run
script_path = '~/python_scripts/calculate_stim_responsiveness.py'
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

slurm = Slurm(
    cpus_per_task=1,
    partition='braintv',
    job_name='run_session_metrics',
    output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    # output=f'{stdout_location}/log.out',
    #mail_type=['FAIL'],          # Mail events (NONE, BEGIN, END, FAIL, ALL)
    #mail_user='corbettb@alleninstitute.org',     # Where to send mail  
    time='24:00:00',
    mem_per_cpu='64gb',
)

print(f'Running script: {script_path}')
slurm.sbatch('{} {}'.format(
        python_path,
        script_path,
    )
)
