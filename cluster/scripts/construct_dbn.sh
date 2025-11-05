#!/bin/bash
#SBATCH --job-name=construct_dbn  # Job name
#SBATCH --mem=4196  # Requested Memory
#SBATCH -p cpu-long  # Partition
#SBATCH -t 3-10:00:00  # Job time limit
#SBATCH -o ../logs/construct_dbn/job-%j.out
#SBATCH -e ../logs/construct_dbn/job-%j.err

module load miniconda/22.11.1-1
conda activate mdbn-pyagrum

cd /work/pi_jensen_umass_edu/pboddavarama_umass_edu/synthesis/gitworktree/jackson-3qnetwork/
python -u src/construct_dbn.py --config_file $1 --experiment_number $2 -v
# python -u src/dbn_extrapolation_pooling.py --config_file $1 --experiment_number $2 -v
