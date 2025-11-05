"""General script to submit multiple parallel jobs to the cluster for comparing DBN and gt."""
import os

CONFIG_FILE = 'configs/tandemq_queries_markov_1.json'

start = 65
end = 68

for experiment_number in range(start, end + 1):
    print(f'Submitting job for experiment {experiment_number}...')
    os.system(f'sbatch compare_dbn_gt.sh {CONFIG_FILE} {experiment_number}')