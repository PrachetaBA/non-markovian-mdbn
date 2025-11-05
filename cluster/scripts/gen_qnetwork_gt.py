"""General script to submit multiple parallel jobs to the cluster for Monte Carlo simulations."""
import os

INDICATOR = False
INFERENCE_METHOD = 'exact-lazyprop'
NREPS = 30
CONFIG_FILE = f'configs/indicator-{INDICATOR}_inference-{INFERENCE_METHOD}_nreps-{NREPS}_pooled-False.json'
GT_FOLDER = 'queries_nreps-30_pooled'

start = 1
end = 200

for experiment_number in range(start, end + 1):
    print(f'Submitting job for experiment {experiment_number}...')
    os.system(
        f'sbatch gen_qnetwork_gt.sh {CONFIG_FILE} {experiment_number} {GT_FOLDER}'
    )
