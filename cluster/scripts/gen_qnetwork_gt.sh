#!/bin/bash
#SBATCH --job-name=gen_qnetwork_gt  # Job name
#SBATCH --mem=4196  # Requested Memory
#SBATCH -p cpu      # Partition
#SBATCH -t 3:30:00  # Job time limit (30 minutes)
#SBATCH -o ../logs/gen_qnetwork_gt/job-%j.out
#SBATCH -e ../logs/gen_qnetwork_gt/job-%j.err

module load miniconda/22.11.1-1
conda activate mdbn-pyagrum

cd /work/pi_jensen_umass_edu/pboddavarama_umass_edu/synthesis/gitworktree/jackson-3qnetwork/
python -u src/simulator_interventions.py --config_file $1 --experiment_number $2 --gt_folder $3
python -u src/compute_montecarlo_gt.py --config_file $1 --experiment_number $2 --gt_folder $3
