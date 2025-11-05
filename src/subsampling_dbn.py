# pylint: disable=logging-fstring-interpolation
"""
Sample data generated using the simulator and create tables that sample
the data at different sampling intervals. 
We generate DBN data that creates row wise implementations of each run according
to the sampling interval. It consists of the following variables: 
1. Lambda_qa_{t}: Arrival rate at Queue A for the time slice t
2. Mu_qa_{t}: Service rate at Queue A for the time slice t
3. Lambda_qb_{t}: Arrival rate at Queue B for the time slice t
4. Mu_qb_{t}: Service rate at Queue B for the time slice t
5. Lambda_qc_{t}: Arrival rate at Queue C for the time slice t
6. Mu_qc_{t}: Service rate at Queue C for the time slice t
7. R_ab_{t}: Routing probability from Queue A to Queue B for the time slice t
8. R_bc_{t}: Routing probability from Queue B to Queue C for the time slice t
9. R_ca_{t}: Routing probability from Queue C to Queue A for the time slice t
10. L_qa_{t}: Queue length at Queue A for the time slice t
11. L_qb_{t}: Queue length at Queue B for the time slice t
12. L_qc_{t}: Queue length at Queue C for the time slice t
"""

# Import libraries

import argparse
import os
import logging
import time

from bisect import bisect  # Bisect is used for subsampling
import pandas as pd
from tqdm import tqdm
import yaml
logger = logging.getLogger('subsampling_discrete_dbn')


def construct_dbn_data(input_filename, output_filename, simulation_end,
                       sampling_rate):
    """
    Function to construct discrete-time sub-sampled DBN data from simulation data
    ----------
    Arguments:
    1. Input Filename: maintains the number of simulation runs, the arrival and service rates
    2. Output Filename: the name of the file to save the time series data 
    3. simulation_end: the end time for the simulator 
    4. sampling_rate: sampling interval (0.1, 0.01, 1.0 etc)
    """
    start = time.time()  # Start time

    # Read the simulator data (continuous time) from the file
    df = pd.read_csv(f"{input_filename}")
    dbn_data = pd.DataFrame()

    runs = max(df['Run'])  # Number of simulation runs

    for r in tqdm(range(1, runs + 1)):
        # Filter the data for each run
        d = df[df['Run'] == r]
        # Print warning if run is empty (no data)
        if d.empty:
            logger.debug(f"Warning: Run {r} is empty")
            continue

        # Get fixed values of the input parameters for this run
        lambda_qa = d['Lambda_qA'].values[0]
        mu_qa = d['Mu_qA'].values[0]
        lambda_qb = d['Lambda_qB'].values[0]
        mu_qb = d['Mu_qB'].values[0]
        lambda_qc = d['Lambda_qC'].values[0]
        mu_qc = d['Mu_qC'].values[0]
        r_ab = d['Rp_ab'].values[0]
        r_bc = d['Rp_bc'].values[0]
        r_ca = d['Rp_ca'].values[0]

        time_series = []

        # Start at time t = 0.0
        t = 0.0
        time_step_counter = 0
        while t <= simulation_end:
            lower_idx = bisect(d['Time'].values, t) - 1
            if lower_idx < 0:
                event = d.iloc[0]['Event']
                qa_ql = int(d.iloc[0]['QueueAL'])
                qb_ql = int(d.iloc[0]['QueueBL'])
                qc_ql = int(d.iloc[0]['QueueCL'])
                # Assert that the event is "Initialization"
                assert event == "Initialization"
                time_series.extend([
                    lambda_qa, mu_qa, lambda_qb, mu_qb, lambda_qc,
                    mu_qc, r_ab, r_bc, r_ca,
                    qa_ql, qb_ql, qc_ql
                ])
            else:
                curr_qa_ql = int(d.iloc[lower_idx]['QueueAL'])
                curr_qb_ql = int(d.iloc[lower_idx]['QueueBL'])
                curr_qc_ql = int(d.iloc[lower_idx]['QueueCL'])
                time_series.extend([
                    lambda_qa, mu_qa, lambda_qb, mu_qb, lambda_qc, mu_qc, r_ab, r_bc, r_ca,
                    curr_qa_ql, curr_qb_ql, curr_qc_ql
                ])
            t += sampling_rate
            t = round(t, 4)
            time_step_counter += 1

        # If r == 1, initialize the dataframe
        if r == 1:
            colnames = []
            counter = 0
            while counter < time_step_counter:
                colnames.extend([
                    f'Lambdaqa{counter}', f'Muqa{counter}', f'Lambdaqb{counter}',
                    f'Muqb{counter}', f'Lambdaqc{counter}', f'Muqc{counter}', 
                    f'Rab{counter}', f'Rbc{counter}', f'Rca{counter}',
                    f'Lqa{counter}', f'Lqb{counter}', f'Lqc{counter}'
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
    dbn_data.to_csv(output_filename)

    end = time.time()  # End time
    logger.info(f"Time taken to construct DBN data: {end - start} seconds")


if __name__ == "__main__":
    # Input arguments
    # Example function call - python src/subsampling_discrete_dbn.py
    # --config_file configs/time_discretization.yaml --experiment_number 1 -v
    parser = argparse.ArgumentParser(
        description=
        "Construct discrete time DBN data for the Markovian queueing system.")
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
                        type=str,
                        help='Path to the simulator configuration file',
                        default='configs/simulator.yaml',
                        required=False)
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
    time_series_experiment = config['time_series_experiment']
    output_folder = config['time_discretization_folder']
    sampling_interval = config['sampling_interval']

    with open(args.sim_config, 'r', encoding='utf-8') as sim_file:
        sim_expts = yaml.safe_load(sim_file)
    sim_config = sim_expts[f'experiment_{time_series_experiment}']

    # Extract the parameters
    experimental_design = sim_config['experimental_design']
    mean_interarrival_rates_queue_a = sim_config['queue_a_arrival_rates']
    mean_service_rates_queue_a = sim_config['queue_a_service_rates']
    mean_interarrival_rates_queue_b = sim_config['queue_b_arrival_rates']
    mean_service_rates_queue_b = sim_config['queue_b_service_rates']
    mean_interarrival_rates_queue_c = sim_config['queue_c_arrival_rates']
    mean_service_rates_queue_c = sim_config['queue_c_service_rates']
    rp_a_to_b = sim_config['routing_probabilities']['a_to_b']
    rp_b_to_c = sim_config['routing_probabilities']['b_to_c']
    rp_c_to_a = sim_config['routing_probabilities']['c_to_a']
    simulation_reps = sim_config['replications']
    simulation_end_time = sim_config['simulation_end']
    num_configs = sim_config['configurations']
    varying_iql = sim_config['varying_iql']
    max_iql = sim_config['max_iql']
    time_series_folder = sim_config['output_folder']

    # Print the parameters
    logger.info(f"Experimental Design: {experimental_design}")
    logger.info(
        f"Mean Interarrival Rates Queue A: {mean_interarrival_rates_queue_a}")
    logger.info(f"Mean Service Rate Queue A: {mean_service_rates_queue_a}")
    logger.info(
        f"Mean Interarrival Rates Queue B: {mean_interarrival_rates_queue_b}")
    logger.info(f"Mean Service Rate Queue B: {mean_service_rates_queue_b}")
    logger.info(f"Mean Interarrival Rates Queue C: {mean_interarrival_rates_queue_c}")
    logger.info(f"Mean Service Rate Queue C: {mean_service_rates_queue_c}")
    logger.info(f"Routing Probabilities A to B: {rp_a_to_b}")
    logger.info(f"Routing Probabilities B to C: {rp_b_to_c}")
    logger.info(f"Routing Probabilities C to A: {rp_c_to_a}")
    logger.info(f"Number of Replications: {simulation_reps}")
    logger.info(f"Simulation End Time: {simulation_end_time}")
    logger.info(f"Number of Configurations: {num_configs}")
    logger.info(f"Varying Initial Queue Length: {varying_iql}")
    logger.info(f"Max Initial Queue Length: {max_iql}")
    logger.info(f"Time Series Folder: {time_series_folder}")
    logger.info(f"Output Folder: {output_folder}")
    logger.info(f"Sampling Interval: {sampling_interval}")

    simulation_file = f"{time_series_folder}/time-series-exp-{time_series_experiment}.csv"

    discrete_time_file = f"{output_folder}/discrete-time-dbn-exp-{experiment_number}.csv"

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct time series data
    construct_dbn_data(simulation_file, discrete_time_file, simulation_end_time,
                       sampling_interval)
