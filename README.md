# erlang-queue-mdbn

This repository contains research code for causal metamodeling of queueing systems using modular dynamic Bayesian networks (MDBNs). It implements a pipeline for simulating queues, approximating non-Markovian arrival distributions with phase-type / hypoexponential representations, constructing discrete-time DBNs, and answering probabilistic and causal queries.

The code supports experiments for multiple queueing branches:
- `src_gamma/`, `src_beta/`, `src_weibull/` — non-Markovian experiments using hypoexponential approximations of Gamma, Beta, and Weibull arrivals

This repo is aligned with the paper "Extending Causal Metamodeling to a Non-Markovian Queue" (Amaranath, Bhide, Jensen, and Haas).

## Quick start

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate mdbn-pyagrum
   ```

2. Generate hypoexponential yaml, for example Gamma
   ```bash
   python src_gamma/gamma_to_hypoexp.py -c config/gamma_simulator.yaml -e 1
   ```

3. Run simulation, for example Gamma/M/1 queue:
   ```bash
   python src_gamma/simulator.py -c config/hypoexp_gamma_simulator.yaml -e 1
   ```

4. Generate discretized data:
   ```bash
   python src_gamma/subsampling_dbn.py -c config/gamma_hypoexp_time_discretization.yaml -e 1 -s config/hypoexp_gamma_simulator.yaml
   python src_gamma/subsampling_2tbn.py -c config/gamma_hypoexp_time_discretization.yaml -e 1 -s config/hypoexp_gamma_simulator.yaml
   ```

5. Construct the DBN:
   ```bash
   python src_gamma/construct_dbn.py -c config/gamma_hypoexp_construct_dbn.yaml -e 1 -s config/hypoexp_gamma_simulator.yaml -t config/gamma_hypoexp_time_discretization.yaml
   ```

6. Run inference on the constructed DBN:
   ```bash
   python src_gamma/inference.py -c config/gamma_query_workload_exp-1.json -e 1 -d <dbn_name> -t config/gamma_hypoexp_time_discretization.yaml
   ```

7. For exact script details and configuration references, see [`README_detailed.md`](README_detailed.md).

## Repository structure

- `config/` — experiment configs for simulation, time discretization, DBN construction, and query workloads.
- `src_gamma/`, `src_beta/`, `src_weibull/` — distribution-specific pipelines.
- `cluster/` — SLURM wrapper scripts for running the pipeline on cluster nodes.
- `data/` — generated simulation and ground-truth data.
- `output/` — query results from inference.
- `figures/` — generated plots and comparison figures.