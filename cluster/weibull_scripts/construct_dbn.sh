#!/bin/bash
#SBATCH --job-name=construct_dbn  # Job name
#SBATCH --mem=10000  # Requested Memory
#SBATCH --account=pi_phaas_umass_edu  # Account
#SBATCH -p cpu      # Partition
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o ../logs/gen_hypoexp_data/job-%j.log
#SBATCH -e ../logs/gen_hypoexp_data/job-%j.err

module load conda/latest
conda activate erm1-mdbn

cd /scratch4/workspace/pboddavarama_umass_edu-erlang-mdbn/anant/erlang-queue-mdbn
python -u src_weibull/construct_dbn.py --config_file $1 --experiment_number $2 --sim_config $3 --time_disc_config $4
