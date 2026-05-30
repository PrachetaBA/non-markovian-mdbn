#!/bin/bash
#SBATCH --job-name=construct_dbn  # Job name
#SBATCH --mem=24000  # Requested Memory
#SBATCH --account=pi_phaas_umass_edu  # Account
#SBATCH -p cpu      # Partition
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o ../logs/gen_beta_data/job-%j.log
#SBATCH -e ../logs/gen_beta_data/job-%j.err

module load conda/latest
conda activate erm1-mdbn

cd "$SLURM_SUBMIT_DIR"
python -u src_beta/construct_dbn.py --config_file $1 --experiment_number $2 --sim_config $3 --time_disc_config $4
