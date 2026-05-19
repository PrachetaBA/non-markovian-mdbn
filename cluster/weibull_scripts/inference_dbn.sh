#!/bin/bash
#SBATCH --job-name=inference_dbn  # Job name
#SBATCH --mem=5000  # Requested Memory
#SBATCH --account=pi_phaas_umass_edu
#SBATCH -p cpu  # Partition
#SBATCH -t 5:00:00  # Job time limit
#SBATCH -o ../logs/weibull_inference_dbn/job-%j.out
#SBATCH -e ../logs/weibull_inference_dbn/job-%j.err

module load conda/latest
conda activate erm1-mdbn

cd /scratch4/workspace/pboddavarama_umass_edu-erlang-mdbn/anant/erlang-queue-mdbn
python -u src_weibull/inference.py --config_file $1 --experiment_number $2 --dbn_name $3
