# pylint: disable=logging-fstring-interpolation
"""
Sample data generated using the Er/M/1 simulator and create tables that sample
the data at different sampling intervals. 
We generate DBN data that creates row-wise implementations of each run according
to the sampling interval. It consists of the following variables (for each time slice t): 
1. Lambda_{t}: Arrival rate for the time slice t
2. Mu_{t}: Service rate for the time slice t
3. K_{t}: Number of Erlang phases
4. CurrentPhase_{t}: Current Erlang phase at time slice t
5. QueueLength_{t}: Queue length at time slice t
"""

# import necessary libraries
import argparse
import os
import logging
import time
from bisect import bisect  # Bisect is used for subsampling
import pandas as pd
from tqdm import tqdm
import yaml

logger = logging.getLogger('subsampling_discrete_dbn_erm1')


def construct_dbn_erm1_data(input_filename, output_filename, simulation_end,
                       sampling_rate):
    """
    Function to construct discrete-time sub-sampled DBN data from Er/M/1 simulation data
    ----------
    Arguments:
    1. input_filename: CSV produced by the simulator with columns
       Run,Lambda,Mu,Current_Phase,K,End,Time,Event,Queue_Length
    2. output_filename: the name of the file to save the time series data 
    3. simulation_end: the end time for the simulator 
    4. sampling_rate: sampling interval (0.1, 0.01, 1.0 etc)
    """
    start = time.time()  # Start time

    # Read the simulator data (continuous time) from the file
    df = pd.read_csv(f"{input_filename}")
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
        lambda_run = d['Lambda'].values[0]
        mu_run = d['Mu'].values[0]
        k_run = int(d['K'].values[0])

        time_series = []

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
                time_series.extend([
                    lambda_run, mu_run, k_run, curr_phase, curr_ql
                ])
            else:
                curr_phase = int(d.iloc[lower_idx]['Current_Phase'])
                curr_ql = int(d.iloc[lower_idx]['Queue_Length'])
                time_series.extend([
                    lambda_run, mu_run, k_run, curr_phase, curr_ql
                ])
            t += sampling_rate
            t = round(t, 4)
            time_step_counter += 1

        # If r == 1, initialize the dataframe columns
        if r == 1:
            colnames = []
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
    logger.info(f"Time taken to construct Er/M/1 DBN data: {end - start} seconds")


if __name__ == "__main__":
    # # Input arguments
    # # Example function call - python subsampling_dbn_erm1.py
    # # --config_file configs/time_discretization.yaml --experiment_number 1 -v
    # parser = argparse.ArgumentParser(
    #     description="Construct discrete time DBN data for an Er/M/1 queueing system.")
    # parser.add_argument(
    #     "--config_file",
    #     "-c",
    #     type=str,
    #     help="Path to the configuration file (e.g. configs/time_discretization.yaml)",
    #     default="configs/time_discretization.yaml")
    # parser.add_argument("--experiment_number",
    #                     "-e",
    #                     type=int,
    #                     help="Experiment number (e.g. 1)",
    #                     default=1)
    # parser.add_argument('--verbose',
    #                     '-v',
    #                     help='Increase output verbosity',
    #                     action='store_true')
    # parser.add_argument('--sim_config',
    #                     '-s',
    #                     type=str,
    #                     help='Path to the simulator configuration file',
    #                     default='configs/simulator.yaml',
    #                     required=False)
    # # Parse the arguments
    # args = parser.parse_args()
    # config_file = args.config_file
    # experiment_number = args.experiment_number

    # if args.verbose:
    #     logging.basicConfig(level=logging.DEBUG)
    # else:
    #     logging.basicConfig(level=logging.INFO)

    # # Read the configuration file and extract the parameters
    # with open(config_file, 'r', encoding='utf-8') as file:
    #     all_configs = yaml.safe_load(file)
    # config = all_configs[f'experiment_{experiment_number}']

    # # Extract the parameters
    # time_series_experiment = config['time_series_experiment']
    # output_folder = config['time_discretization_folder']
    # sampling_interval = config['sampling_interval']

    # with open(args.sim_config, 'r', encoding='utf-8') as sim_file:
    #     sim_expts = yaml.safe_load(sim_file)
    # sim_config = sim_expts[f'experiment_{time_series_experiment}']

    # # Extract simulator parameters (expected keys in your simulator config)
    # experimental_design = sim_config.get('experimental_design', None)
    # mean_interarrival_rates = sim_config.get('arrival_rates', None)
    # mean_service_rates = sim_config.get('service_rates', None)
    # simulation_reps = sim_config.get('replications', None)
    # simulation_end_time = sim_config.get('simulation_end', None)
    # num_configs = sim_config.get('configurations', None)
    # time_series_folder = sim_config.get('output_folder', None)

    # # Print the parameters
    # logger.info(f"Experimental Design: {experimental_design}")
    # logger.info(f"Mean Interarrival Rates: {mean_interarrival_rates}")
    # logger.info(f"Mean Service Rates: {mean_service_rates}")
    # logger.info(f"Number of Replications: {simulation_reps}")
    # logger.info(f"Simulation End Time: {simulation_end_time}")
    # logger.info(f"Number of Configurations: {num_configs}")
    # logger.info(f"Time Series Folder: {time_series_folder}")
    # logger.info(f"Output Folder: {output_folder}")
    # logger.info(f"Sampling Interval: {sampling_interval}")

    # # Construct expected file paths
    # simulation_file = f"{time_series_folder}/time-series-exp-{time_series_experiment}.csv"
    # discrete_time_file = f"{output_folder}/discrete-time-dbn-erm1-exp-{experiment_number}.csv"

    # # Create the output folder if it does not exist
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # # Construct time series data for Er/M/1
    # construct_dbn_erm1_data(simulation_file, discrete_time_file, simulation_end_time,
    #                    sampling_interval)


    # Input CSV from your Er/M/1 simulation
    #simulation_file = "src/test_data/results.csv"
    #discrete_time_file = "src/test_data/discrete-time-dbn-erm1.csv"
    simulation_file = r"D:\UMass Amherst\Research App\MDBN\Er-M-1\erlang-queue-mdbn\src\test_data\results.csv"
    discrete_time_file = r"D:\UMass Amherst\Research App\MDBN\Er-M-1\erlang-queue-mdbn\src\test_data\discrete-time-dbn-erm1.csv"

    simulation_end_time = 3.0
    sampling_interval = 1.0
    
    # Call the subsampling function
    construct_dbn_erm1_data(simulation_file, discrete_time_file,
                       simulation_end_time, sampling_interval)