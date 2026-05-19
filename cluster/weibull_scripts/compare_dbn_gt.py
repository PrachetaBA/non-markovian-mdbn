"""General script to submit multiple parallel jobs to the cluster for comparing DBN and gt."""
import os

CONFIG_FILE = 'config/weibull_query_workload_exp-5.json'

start = 1
end = 500

for experiment_number in range(start, end + 1):
    print(f'Submitting job for experiment {experiment_number}...')
    os.system(f'sbatch compare_dbn_gt.sh {CONFIG_FILE} {experiment_number}')