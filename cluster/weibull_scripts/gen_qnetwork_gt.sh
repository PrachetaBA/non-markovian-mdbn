#!/bin/bash
#SBATCH --job-name=gen_qnetwork_gt  # Job name
#SBATCH --mem=4196  # Requested Memory
#SBATCH -p cpu      # Partition
#SBATCH -t 3:30:00  # Job time limit (30 minutes)
#SBATCH -o ../logs/weibull_gen_qnetwork_gt_get_runtime/job-%j.out
#SBATCH -e ../logs/weibull_gen_qnetwork_gt_get_runtime/job-%j.err

module load conda/latest
conda activate erm1-mdbn

cd "$SLURM_SUBMIT_DIR"
python -u src_weibull/simulator_hypoexp_interventions.py --config_file $1 --experiment_number $2 -g $3
python -u src_weibull/simulator_weibull_interventions.py --config_file $1 --experiment_number $2 -g $3
python -u src_weibull/compute_montecarlo_gt.py --config_file $1 --experiment_number $2