# pylint: disable=logging-fstring-interpolation
"""
Sample data generated using the Hypoexp/M/1 simulator and create tables that sample
the data at different sampling intervals. 
We generate 2TBN data that consists of the following columns (Hypoexp/M/1):
1. Lambda_tprev: Phase rate at time t (previous slice)
2. Mu_tprev: Service rate at time t (previous slice)
3. K_tprev: Number of Hypoexp phases at time t (previous slice)
4. CurrentPhase_tprev: Current Hypoexp phase at time t (previous slice)
5. QueueLength_tprev: Queue length at time t (previous slice)
6. Lambda: Phase rate at time t+Δ (current slice)
7. Mu: Service rate at time t+Δ (current slice)
8. K: Number of Hypoexp phases at time t+Δ (current slice)
9. CurrentPhase: Current Hypoexp phase at time t+Δ (current slice)
10. QueueLength: Queue length at time t+Δ (current slice)
"""

# Import libraries
import ast
import os
from bisect import bisect  # Bisect is used for subsampling
import argparse
from pathlib import Path
import time
import logging
from tqdm import tqdm
import yaml
import numpy as np
import polars as pl

logger = logging.getLogger('subsampling_discrete_2tbn')


def hypoexp_construct_2tbn_data(input_filename, output_filename, sampling_rate):
    """
    Function to construct discrete-time sub-sampled data from Hypoexp/M/1 simulation data
    ----------
    Arguments:
    1. Input Filename: CSV produced by the simulator with columns
       Run,Alpha,Theta,Phase_Rates,Mu,Current_Phase,IQL,End,Time,Event,Queue_Length
    2. Output Filename: the name of the file to save the time series data 
    3. simulation_end: the end time for the simulator 
    4. sampling_rate: sampling interval (0.1, 0.01, 1.0 etc)
    """
    start = time.time()  # Start time

    # Read the simulator data (continuous time) from the file
    df = pl.read_csv(f"{input_filename}")
    #simulation_end = float(df['Time'].max().ceil())
    simulation_end = float(np.ceil(df['Time'].max()))
    global_df = pl.DataFrame(schema=[
        ("Alpha", pl.Float64),
        ("Theta", pl.Float64),
        ("Lambda_tprev", pl.Float64),
        ("Mu_tprev", pl.Float64),
        ("K_tprev", pl.Int64),
        ("CurrentPhase_tprev", pl.Int64),
        ("QueueLength_tprev", pl.Int64),
        ("Lambda", pl.Float64),
        ("Mu", pl.Float64),
        ("K", pl.Int64),
        ("CurrentPhase", pl.Int64),
        ("QueueLength", pl.Int64),
    ])  # Initialize the global dataframe

    runs = int(df["Run"].max())  # Extract the number of runs in the file
    list_global_df = []  # List to store the time series data for each run

    for r in tqdm(range(1, runs + 1)):  # Construct sampling data for each run
        d = df.filter(pl.col("Run") == r).select(df.columns)
        if d.is_empty():
            logger.debug(f"Run {r} is empty")
            continue

        # Extract per-run fixed parameters
        phase_rates = ast.literal_eval(d[0, "Phase_Rates"])
        alpha_run = float(d[0, "Alpha"])
        theta_run = float(d[0, "Theta"])
        mu_run = float(d[0, "Mu"])  # Service rate
        k_run = len(phase_rates)

        # Initialize a new polars dataframe with the columns for the previous slice
        d1 = pl.DataFrame(schema=[
            ("Alpha", pl.Float64),
            ("Theta", pl.Float64),
            ("Lambda_tprev", pl.Float64),
            ("Mu_tprev", pl.Float64),
            ("K_tprev", pl.Int64),
            ("CurrentPhase_tprev", pl.Int64),
            ("QueueLength_tprev", pl.Int64),
        ])

        # For every sampling instant, find the last event <= t and capture state
        # Do not use np.arange or np.linspace, it produces floating point errors
        t = 0.0  # Start time = 0.0
        while t <= simulation_end:
            lower_idx = bisect(d.get_column("Time").to_numpy(), t) - 1
            if lower_idx < 0:
                event = d[0, "Event"]
                curr_phase = int(d[0, "Current_Phase"])
                curr_ql = int(d[0, "Queue_Length"])
                assert event == "Initialization"
            else:
                curr_phase = int(d[lower_idx, "Current_Phase"])
                curr_ql = int(d[lower_idx, "Queue_Length"])

            lambda_active = phase_rates[curr_phase - 1]

            time_series = pl.DataFrame({
                "Alpha": [alpha_run],
                "Theta": [theta_run],
                "Lambda_tprev": [lambda_active],
                "Mu_tprev": [mu_run],
                "K_tprev": [k_run],
                "CurrentPhase_tprev": [curr_phase],
                "QueueLength_tprev": [curr_ql],
            })
            d1 = d1.vstack(time_series)
            t = t + sampling_rate
            t = round(t, 4)

        # If not enough samples to construct transitions, skip this run
        if len(d1) < 2:
            logger.debug(f"Run {r} has insufficient samples ({len(d1)}). Skipping.")
            continue

        # Prepare the second (current) slice by shifting d1
        d2 = d1.clone()
        d2 = d2[1:, :]  # Remove first row
        last_row_values = d2.row(len(d2) - 1)  # Get the last row values
        last_row = pl.DataFrame({
            'Alpha': [last_row_values[0]],
            'Theta': [last_row_values[1]],
            'Lambda_tprev': [last_row_values[2]],
            'Mu_tprev': [last_row_values[3]],
            'K_tprev': [last_row_values[4]],
            'CurrentPhase_tprev': [last_row_values[5]],
            'QueueLength_tprev': [last_row_values[6]],
        })
        d2 = d2.vstack(last_row)
        # REMOVE static params from the *current* slice so we don't duplicate Alpha/Theta
        d2 = d2.drop(["Alpha", "Theta"])
        # Rename d2 columns to represent current slice variables
        d2.columns = [
            'Lambda', 'Mu', 'K', 'CurrentPhase', 'QueueLength'
        ]
        d2 = d2[:len(d2) - 1, :]  # Remove the last row to make lengths equal

        # Concatenate previous and current slice horizontally
        df_2tbn = pl.concat([d1, d2], how='horizontal')
        df_2tbn = df_2tbn.drop_nulls()

        list_global_df.append(df_2tbn)  # Append the time series data for each run

    if len(list_global_df) == 0:
        # Nothing to concatenate; write empty schema CSV
        global_df.write_csv(output_filename)
        logger.info("No data constructed for 2TBN; wrote empty file.")
        return

    global_df = pl.concat(list_global_df, how="vertical")  # Concatenate runs

    end = time.time()  # End time
    # Save the time series data to a file
    global_df.write_csv(output_filename)
    logger.info(f"Time taken to construct time series data: {end - start} seconds")


if __name__ == "__main__":
    # Input arguments
    # Example function call - python src/subsampling_discrete_2tbn_erm1.py
    # --config_file configs/time_discretization.yaml --experiment_number 1 -v
    parser = argparse.ArgumentParser(
        description=
        "Construct discrete time 2TBN data for the Hypoexp/M/1 queueing system.")
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help=
        "Path to the configuration file (e.g. configs/time_discretization.yaml)",
        default="configs/time_discretization.yaml")
    parser.add_argument("--experiment_number",
                        "-e",
                        type=int,
                        help="Experiment number (e.g. 1)",
                        default=1)
    parser.add_argument('--verbose',
                        '-v',
                        help='Increase output verbosity',
                        action='store_true')
    parser.add_argument('--sim_config',
                        '-s',
                        help='Path to the simulator configuration file',
                        default='configs/simulator.yaml',
                        required=False,
                        type=str)
    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    experiment_number = args.experiment_number

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read the configuration file and extract the parameters
    with open(config_file, 'r', encoding='utf-8') as file:
        all_configs = yaml.safe_load(file)
    config = all_configs[f'experiment_{experiment_number}']

    # Extract the parameters
    time_series_experiment = config['hypoexp_m1_time_series_experiment']
    output_folder = config['time_discretization_folder']
    sampling_interval = config['sampling_interval']

    # Compute the simulation CSV path
    project_root = Path(__file__).resolve().parents[1]
    simulation_file = project_root / "data" / "simulation" / f"hypoexp-m1-{time_series_experiment}.csv"
    output_path = project_root / "data" / output_folder
    output_path.mkdir(parents=True, exist_ok=True)
    discrete_time_file = output_path / f"hypoexp-discrete-time-2tbn-exp-{experiment_number}.csv"

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct time series data
    hypoexp_construct_2tbn_data(simulation_file, discrete_time_file, sampling_interval)
