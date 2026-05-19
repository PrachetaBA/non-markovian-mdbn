# pylint: disable=logging-fstring-interpolation
"""
Sample data generated using the Hypoexp/M/1 simulator and create tables that sample
the data at different sampling intervals. 
We generate DBN data that creates row-wise implementations of each run according
to the sampling interval. It consists of the following variables (for each time slice t): 
1. Lambda_{t}: Phase rate for the time slice t
2. Mu_{t}: Service rate for the time slice t
3. K_{t}: Number of Erlang phases
4. CurrentPhase_{t}: Current Erlang phase at time slice t
5. QueueLength_{t}: Queue length at time slice t
"""

# import necessary libraries
import ast
import argparse
import logging
import time
from bisect import bisect  # Bisect is used for subsampling
import pandas as pd
from tqdm import tqdm
import yaml
from pathlib import Path
import numpy as np

logger = logging.getLogger('subsampling_discrete_dbn_weibull_hypoexp_m1')


def construct_dbn_weibull_hypoexp_m1_data(input_filename, output_filename, sampling_rate):
    """
    Function to construct discrete-time sub-sampled DBN data from Weibull/Hypoexp/M/1 simulation data
    ----------
    Arguments:
    1. input_filename: CSV produced by the simulator with columns
       Run,WB_SHAPE,WB_SCALE,Phase_Rates,Mu,Current_Phase,IQL,End,Time,Event,Queue_Length
    2. output_filename: the name of the file to save the time series data 
    3. simulation_end: the end time for the simulator 
    4. sampling_rate: sampling interval (0.1, 0.01, 1.0 etc)
    """
    start = time.time()  # Start time

    # Read the simulator data (continuous time) from the file
    df = pd.read_csv(f"{input_filename}")
    simulation_end = float(np.ceil(df['Time'].max()))
    dbn_data = pd.DataFrame()

    runs = max(df['Run']) # Number of simulation runs

    for r in tqdm(range(1, runs + 1)):
        # Filter the data for each run
        d = df[df['Run'] == r]
        # Print warning if run is empty (no data)
        if d.empty:
            logger.debug(f"Warning: Run {r} is empty")
            continue

        # Get fixed values of the input parameters for this run
        phase_rates = ast.literal_eval(d['Phase_Rates'].values[0])
        wb_shape_run = d['WB_SHAPE'].values[0]
        wb_scale_run = d['WB_SCALE'].values[0]
        mu_run = d['Mu'].values[0]
        k_run = len(phase_rates)

        #time_series = []
        time_series = [wb_shape_run, wb_scale_run]

        # Start at time t = 0.0
        t = 0.0
        time_step_counter = 0
        while t <= simulation_end:
            # last event time <= t
            lower_idx = bisect(d['Time'].values, t) - 1
            if lower_idx < 0:
                # if events before time t use initialization
                event = d.iloc[0]['Event']
                curr_phase = int(d.iloc[0]['Current_Phase'])
                curr_ql = int(d.iloc[0]['Queue_Length'])
                # Assert that the event is "Initialization"
                assert event == "Initialization"
                lambda_active = phase_rates[curr_phase - 1]
                time_series.extend([lambda_active, mu_run, k_run, curr_phase, curr_ql])
            else:
                curr_phase = int(d.iloc[lower_idx]['Current_Phase'])
                curr_ql = int(d.iloc[lower_idx]['Queue_Length'])
                lambda_active = phase_rates[curr_phase - 1]
                time_series.extend([lambda_active, mu_run, k_run, curr_phase, curr_ql])
            t += sampling_rate
            t = round(t, 4)
            time_step_counter += 1

        # If r == 1, initialize the dataframe columns
        if r == 1:
            colnames = ['WBshape0', 'WBscale0']
            counter = 0
            while counter < time_step_counter:
                colnames.extend([
                    f'Lambda{counter}', f'Mu{counter}', f'K{counter}',
                    f'CurrentPhase{counter}', f'QueueLength{counter}'
                ])
                counter += 1
            dbn_data = pd.DataFrame(columns=colnames)
            logger.info(f"Colnames: {colnames}")
            logger.info(f"Time series length: {len(time_series)}")
            logger.info(f"Number of columns: {len(colnames)}")
            # Sanity check that they are the same
            assert len(time_series) == len(colnames)
            dbn_data.loc[0] = time_series
        else:
            dbn_data.loc[len(dbn_data)] = time_series

    # Save the data to a file
    dbn_data.to_csv(output_filename, index=False)

    end = time.time()  # End time
    logger.info(f"Time taken to construct Hypoexp/M/1 DBN data: {end - start} seconds")


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(
        description="Construct discrete time DBN data for an Hypoexp/M/1 queueing system.")
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="Path to the configuration file (e.g. configs/weibull_hypoexp_time_discretization.yaml)",
        default="configs/weibull_hypoexp_time_discretization.yaml")
    parser.add_argument("--experiment_number",
                        "-e",
                        type=int,
                        help="Experiment number (e.g. 1)",
                        default=1)
    parser.add_argument('--verbose',
                        '-v',
                        help='Increase output verbosity',
                        action='store_true')
    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    experiment_number = args.experiment_number

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # read the configuration file and extract the parameters
    with open(config_file, 'r', encoding='utf-8') as file:
        all_configs = yaml.safe_load(file)
    config = all_configs[f'experiment_{experiment_number}']

    # extract the parameters
    time_series_experiment = config['weibull_hypoexp_m1_time_series_experiment']
    input_file = f"simulation/weibull_hypoexp-m1-{time_series_experiment}.csv"
    output_folder = config['time_discretization_folder']
    sampling_interval = config['sampling_interval']

    # expected file paths
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / input_file
    output_path = project_root / "data" / output_folder
    output_path.mkdir(parents=True, exist_ok=True)
    discrete_time_file = output_path / f"weibull_hypoexp-discrete-time-dbn-exp-{experiment_number}.csv"

    # call the subsampling function
    construct_dbn_weibull_hypoexp_m1_data(input_path, discrete_time_file, sampling_interval)
