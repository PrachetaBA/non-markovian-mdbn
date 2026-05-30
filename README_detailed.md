## Detailed repository and workflow reference

This repository implements a queueing simulation and DBN metamodel pipeline for non-Markovian queueing systems. It is aligned with the paper on extending MDBNs to non-Markovian queues by approximating general arrival distributions with phase-type / hypoexponential models. This repo implements that approach for Gamma, Beta and Weibull arrival processes.


### Root folders

- `config/` — YAML and JSON configuration files for simulation, time discretization, DBN construction, and query workloads.
- `src_gamma/` — G/M/1 experiments with Gamma arrivals and hypoexponential approximation.
- `src_beta/` — G/M/1 experiments with Beta arrivals and hypoexponential approximation.
- `src_weibull/` — G/M/1 experiments with Weibull arrivals and hypoexponential approximation.
- `cluster/` — SLURM wrapper scripts for cluster execution.
- `data/`, `output/`, `results/`, `figures/` — stored generated data, inference outputs, evaluation results, and plots.

## High-level experiment pipeline

The standard workflow for each distribution branch is:

1. Generate simulation data from `src_*/simulator.py` using a branch-specific config file.
2. Convert or approximate non-Markovian arrivals using phase-type / hypoexponential representation if needed.
3. Discretize the simulation output into a time-series or 2-TBN dataset using `src_*/subsampling_2tbn.py` and/or `src_*/subsampling_dbn.py`.
4. Construct the DBN from discretized data using `src_*/construct_dbn.py`.
5. Run queries on the DBN with `src_*/inference.py`.
6. Optionally compare DBN inference to ground truth via Monte Carlo or intervention-enabled simulation.

## `config/` files

Key configuration files in `config/` include:

- `gamma_simulator.yaml`, `weibull_simulator.yaml`, `beta_simulator.yaml` — Original arrival distribution simulation settings.
- `hypoexp_gamma_simulator.yaml` — Gamma-to-hypoexp simulation settings.
- `hypoexp_weibull_simulator.yaml` — Weibull-to-hypoexp simulation settings.
- `hypoexp_beta_simulator.yaml` — Beta-to-hypoexp simulation settings.
- `gamma_hypoexp_time_discretization.yaml`, `beta_hypoexp_time_discretization.yaml`, `weibull_hypoexp_time_discretization.yaml` — time discretization settings.
- `gamma_hypoexp_construct_dbn.yaml`, `beta_hypoexp_construct_dbn.yaml`, `weibull_hypoexp_construct_dbn.yaml` — DBN construction settings.
- `gamma_query_workload_exp-1.json` , `beta_query_workload_exp-1.json`, `weibull_query_workload_exp-1.json`, etc. — query workloads for inference.

The configs use these keys:
- `runs` — number of simulation runs
- `simulation_end` — simulation end time
- `varying_iql` — should starting queuelength vary during simulations
- `max_iql` — maximum queuelength
- distribution-specific keys such as `arrival_distributions`, `ALPHAS`, `B_BETAS`, `phase_rates`, etc.
- `output_folder` — where to store the results
- `service_rates` — service rate follows expenontial distribution


## Distribution-specific source branches

### `src_gamma/`

This is the main branch for Gamma arrival experiments. It includes the full pipeline.

Scripts:
- `simulator.py` — generate simulation data using `config/hypoexp_gamma_simulator.yaml`.
- `gamma_to_hypoexp.py` — convert Gamma parameters into hypoexponential phase-rates.
- `subsampling_2tbn.py` — produce 2-TBN training data.
- `subsampling_dbn.py` — produce DBN training data.
- `construct_dbn.py` — construct the DBN.
- `inference.py` — run inference on the constructed DBN.
- `compute_montecarlo_gt.py` — compute Monte Carlo ground truth.
- `simulator_gamma_interventions.py`, `simulator_hypoexp_interventions.py` — support intervention-based ground-truth data.

### `src_beta/`

This branch mirrors the Gamma pipeline for Beta arrival distributions.

Scripts:
- `simulator.py` — generate Beta simulation data.
- `beta_to_hypoexp.py` — convert Beta parameters into hypoexponential phase-rates.
- `subsampling_2tbn.py`, `subsampling_dbn.py` — discretize simulation output.
- `construct_dbn.py` — build the DBN.
- `inference.py` — run inference.
- `compute_montecarlo_gt.py` — compute ground truth.
- `simulator_beta_interventions.py`, `simulator_hypoexp_interventions.py` — intervention-enabled simulation.

### `src_weibull/`

This branch mirrors the Gamma/Beta pipeline for Weibull arrival distributions.

Scripts:
- `simulator.py` — generate Weibull simulation data.
- `weibull_to_hypoexp.py` — convert Weibull parameters into hypoexponential phase-rates.
- `subsampling_2tbn.py`, `subsampling_dbn.py` — discretize simulation output.
- `construct_dbn.py` — build the DBN.
- `inference.py` — run inference.
- `compute_montecarlo_gt.py` — compute ground truth.
- `simulator_weibull_interventions.py`, `simulator_hypoexp_interventions.py` — intervention-enabled simulation.


## Key script usage patterns

The main scripts in each branch generally use the same CLI pattern:

- `python src_*/simulator.py -c <config_file> -e <experiment_number> [-v]`
- `python src_*/subsampling_2tbn.py -c <time_disc_config> -e <experiment_number> [-v] -s <sim_config>`
- `python src_*/construct_dbn.py -c <dbn_config> -e <experiment_number> [-v] -s <sim_config> -t <time_disc_config>`
- `python src_*/inference.py -c <query_config> -e <experiment_number> [-v] -t <time_disc_config> -d <dbn_name>`

The branch-specific config files are stored in `config/`.

## Output folders

Typical output locations are:
- simulation CSV files in `data/simulation`
- discretized DBN input in `data/discrete_time`
- posterior inference outputs in `output/queries`
- ground-truth distributions in `data/queries_gt`


## Cluster scripts

The `cluster/` directory contains wrappers for batch execution of the pipeline on HPC clusters using Slurm. Example directories include:
- `cluster/beta_scripts/`
- `cluster/gamma_scripts/`
- `cluster/weibull_scripts/`

These scripts typically load a conda environment and run the Python branch scripts with the provided config file and experiment number.