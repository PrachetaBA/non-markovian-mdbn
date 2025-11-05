"""General script to submit multiple parallel jobs to the cluster for DBN inference."""
import os

INDICATOR = False
NREPS = 30
# CONFIG_FILE = f'configs/indicator-{INDICATOR}_nreps-{NREPS}.json'

# With pooling, we also specify the experiment number (one has extrapolation, the other doesn't)
CONFIG_FILE = f'configs/indicator-{INDICATOR}_nreps-{NREPS}_pooled-False.json'
EXP_NUM = 1

print(f'Submitting job for {CONFIG_FILE}')
os.system(f'sbatch construct_dbn.sh {CONFIG_FILE}'
          f' {EXP_NUM}'  # Used for pooling and extrapolation experiments
         )
