#!/bin/bash
#SBATCH --mem=10000  # Requested Memory
#SBATCH -p cpu-long      # Partition
#SBATCH -t 3-00:00:00  # Job time limit
#SBATCH -o ../logs/gen_discrete_time_data/job-%j.out
#SBATCH -e ../logs/gen_discrete_time_data/job-%j.err

module load miniconda/22.11.1-1
conda activate mdbn-pyagrum

cd /work/pi_jensen_umass_edu/pboddavarama_umass_edu/synthesis/gitworktree/jackson-3qnetwork/
python -u src/subsampling_dbn.py --config_file $1 --experiment_number $2 -v 