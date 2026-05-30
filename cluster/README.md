# Cluster scripts

These SLURM batch scripts wrap the pipeline steps for execution on HPC clusters. There is one subdirectory per distribution branch (`beta_scripts/`, `gamma_scripts/`, `weibull_scripts/`), each containing the same set of scripts.

## Before running

Three things must be set to match your cluster environment:

1. **Working directory** — All scripts use `$SLURM_SUBMIT_DIR`, so submit jobs from the **repository root**:
   ```bash
   sbatch cluster/gamma_scripts/construct_dbn.sh <args>
   ```

2. **Conda environment** — Scripts activate the environment named `erm1-mdbn`. Create it from the repo root first:
   ```bash
   conda env create -f environment.yml -n erm1-mdbn
   ```

3. **SLURM account and partition** — Each script contains:
   ```
   #SBATCH --account=pi_phaas_umass_edu
   #SBATCH -p cpu
   ```
   Replace these with your own account name and partition before submitting.

## Script reference

| Script | Pipeline step | Key arguments |
|--------|--------------|---------------|
| `gen_hypoexp_simulation.sh` | Run simulator | `<sim_config> <experiment_number>` |
| `gen_discrete_dbn.sh` | Subsample for DBN | `<time_disc_config> <experiment_number> <sim_config>` |
| `gen_discrete_2tbn.sh` | Subsample for 2-TBN | `<time_disc_config> <experiment_number> <sim_config>` |
| `construct_dbn.sh` | Build the DBN | `<dbn_config> <experiment_number> <sim_config> <time_disc_config>` |
| `inference_dbn.sh` | Run inference | `<query_config> <experiment_number> <dbn_name>` |
| `gen_qnetwork_gt.sh` | Generate ground truth | `<sim_config> <experiment_number>` |
| `compare_dbn_gt.sh` | Compare DBN vs ground truth | `<experiment_number>` |

Config files for each branch are in `config/`. See `README_detailed.md` for the full pipeline walkthrough.
