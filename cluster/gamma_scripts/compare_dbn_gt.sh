#!/bin/bash
#SBATCH --mem=1000  # Requested Memory
#SBATCH -p cpu      # Partition
#SBATCH -t 0:30:00  # Job time limit
#SBATCH -o ../logs/compare_dbn_gt/job-%j.out
#SBATCH -e ../logs/compare_dbn_gt/job-%j.err

module load conda/latest
conda activate erm1-mdbn

cd /scratch4/workspace/pboddavarama_umass_edu-erlang-mdbn/anant/erlang-queue-mdbn
python -u utils/compare_queries.py --config_file $1 --experiment_number $2