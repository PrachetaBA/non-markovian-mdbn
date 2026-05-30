#!/bin/bash
#SBATCH --job-name=inference_dbn  # Job name
#SBATCH --mem=32000  # Requested Memory
#SBATCH --account=pi_phaas_umass_edu
#SBATCH -p cpu  # Partition
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o ../logs/inference_dbn/job-%j.out
#SBATCH -e ../logs/inference_dbn/job-%j.err

module load conda/latest
conda activate erm1-mdbn

cd "$SLURM_SUBMIT_DIR"
python -u src_hypoexp/inference.py --config_file $1 --experiment_number $2 --dbn_name $3
