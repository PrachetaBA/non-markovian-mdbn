#!/bin/bash
#SBATCH --job-name=inference_dbn  # Job name
#SBATCH --mem=32000  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -C amd7763 # nodes with AMD EPYC 7763 processors constraint
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o ../logs/inference_dbn/job-%j.out
#SBATCH -e ../logs/inference_dbn/job-%j.err

module load miniconda/22.11.1-1
conda activate mdbn-pyagrum

cd /work/pi_jensen_umass_edu/pboddavarama_umass_edu/synthesis/gitworktree/jackson-3qnetwork/
python -u src/inference.py --config_file $1 --experiment_number $2 --dbn_name $3
