"""General script to submit multiple parallel jobs to the cluster for Monte Carlo simulations."""
import os



start = 1
end = 500
CONFIG_FILE = 'config/weibull_query_workload_exp-5.json'
GT_FOLDER = 'dbn_weibull_hypoexpm1_exp5'

for experiment_number in range(start, end + 1):
    print(f'Submitting job for experiment {experiment_number}...')
    os.system(
        f'sbatch gen_qnetwork_gt.sh {CONFIG_FILE} {experiment_number} {GT_FOLDER}'
    )
