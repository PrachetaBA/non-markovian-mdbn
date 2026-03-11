"""General script to submit multiple parallel jobs to the cluster for DBN inference."""
import os

CONFIG_FILE = 'config/query_workload_exp-1.json'
DBN_NAME = 'dbn_hypoexpm1_exp8_crosstab'

start = 1
end = 500

for experiment_number in range(start, end + 1):
    print(f'Submitting job for experiment {experiment_number}...')
    os.system(
        f'sbatch inference_dbn.sh {CONFIG_FILE} {experiment_number} {DBN_NAME}')
