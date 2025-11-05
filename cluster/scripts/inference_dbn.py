"""General script to submit multiple parallel jobs to the cluster for DBN inference."""
import os

INDICATOR = False
INFERENCE = 'exact-lazyprop'
NREPS = 30
CONFIG_FILE = f'configs/indicator-{INDICATOR}_inference-{INFERENCE}_nreps-{NREPS}_pooled-True.json'
DBN_NAME = f'dbn_indicator-{INDICATOR}_nreps-{NREPS}_pooled-True_extrapolation-True'

start = 1
end = 200

for experiment_number in range(start, end + 1):
    print(f'Submitting job for experiment {experiment_number}...')
    os.system(
        f'sbatch inference_dbn.sh {CONFIG_FILE} {experiment_number} {DBN_NAME}')
