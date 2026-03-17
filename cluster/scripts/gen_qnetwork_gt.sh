#!/bin/bash
#SBATCH --job-name=gen_qnetwork_gt  # Job name
#SBATCH --mem=4196  # Requested Memory
#SBATCH -p cpu      # Partition
#SBATCH -t 3:30:00  # Job time limit (30 minutes)
#SBATCH -o ../logs/gen_qnetwork_gt/job-%j.out
#SBATCH -e ../logs/gen_qnetwork_gt/job-%j.err

module load conda/latest
conda activate /scratch4/workspace/pboddavarama_umass_edu-mdbn-qnetwork/conda_envs/pyagrum-gpu

cd /scratch4/workspace/pboddavarama_umass_edu-erlang-mdbn/anant/erlang-queue-mdbn
python -u src_hypoexp/simulator_gamma_interventions.py --config_file $1 --experiment_number $2 -g $3
python -u src_hypoexp/compute_montecarlo_gt.py --config_file $1 --experiment_number $2