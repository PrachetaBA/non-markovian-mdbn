#!/bin/bash
#SBATCH --mem=1000  # Requested Memory
#SBATCH -p cpu      # Partition
#SBATCH -t 0:30:00  # Job time limit
#SBATCH -o ../logs/compare_dbn_gt/job-%j.out
#SBATCH -e ../logs/compare_dbn_gt/job-%j.err

module load miniconda/22.11.1-1
conda activate mdbn-pyagrum

cd /work/pi_jensen_umass_edu/pboddavarama_umass_edu/synthesis/gitworktree/markovian-qnetwork/
python -u utils/compare_queries.py --config_file $1 --experiment_number $2