#!/bin/bash
#SBATCH --job-name=construct_2tbn  # Job name
#SBATCH --mem=10000  # Requested Memory
#SBATCH --account=pi_phaas_umass_edu  # Account
#SBATCH -p cpu      # Partition
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o ../logs/gen_weibull_data/job-%j.log
#SBATCH -e ../logs/gen_weibull_data/job-%j.err

module load conda/latest
conda activate erm1-mdbn

cd "$SLURM_SUBMIT_DIR"
python -u src_weibull/subsampling_2tbn.py --config_file $1 --experiment_number $2