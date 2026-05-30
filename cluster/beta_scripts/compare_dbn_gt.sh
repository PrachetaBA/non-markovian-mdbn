#!/bin/bash
#SBATCH --mem=1000  # Requested Memory
#SBATCH -p cpu      # Partition
#SBATCH -t 0:30:00  # Job time limit
#SBATCH -o ../logs/beta_compare_dbn_gt_different_time_discretization/job-%j.out
#SBATCH -e ../logs/beta_compare_dbn_gt_different_time_discretization/job-%j.err

module load conda/latest
conda activate erm1-mdbn

cd "$SLURM_SUBMIT_DIR"
python -u utils/beta_compare_queries.py --config_file $1 --experiment_number $2