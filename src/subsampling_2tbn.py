# pylint: disable=logging-fstring-interpolation
"""
Sample data generated using the simulator and create tables that sample
the data at different sampling intervals. 
We generate 2TBN data that consists of the following columns. 
1. Lambda_qa: Arrival rate at Queue A
2. Mu_qa: Service rate at Queue A
3. Lambda_qb: Arrival rate at Queue B
4. Mu_qb: Service rate at Queue B
5. Lambda_qc: Arrival rate at Queue C
6. Mu_qc: Service rate at Queue C
7. R_ab: Routing probability from Queue A to Queue B
8. R_bc: Routing probability from Queue B to Queue C
9. R_ca: Routing probability from Queue C to Queue A
10. L_qa: Queue length at Queue A
11. L_qb: Queue length at Queue B
12. L_qc: Queue length at Queue C
"""

# Import libraries
import os
from bisect import bisect  # Bisect is used for subsampling
import argparse
import time
import logging
from tqdm import tqdm
import yaml

import polars as pl
logger = logging.getLogger('subsampling_discrete_2tbn')


def construct_2tbn_data(input_filename, output_filename, simulation_end,
                        sampling_rate):
    """
    Function to construct discrete-time sub-sampled data from simulation data
    ----------
    Arguments:
    1. Input Filename: maintains the number of simulation runs, the arrival and service rates
    2. Output Filename: the name of the file to save the time series data 
    3. simulation_end: the end time for the simulator 
    4. sampling_rate: sampling interval (0.1, 0.01, 1.0 etc)
    """
    start = time.time()  # Start time

    # Read the simulator data (continuous time) from the file
    df = pl.read_csv(f"{input_filename}")
    global_df = pl.DataFrame(schema=[("Lambda_qa_tprev", pl.Float64),
                                     ("Mu_qa_tprev", pl.Float64),
                                     ("Lambda_qb_tprev", pl.Float64),
                                     ("Mu_qb_tprev", pl.Float64),
                                     ("Lambda_qc_tprev", pl.Float64),
                                     ("Mu_qc_tprev", pl.Float64),
                                     ("R_ab_tprev", pl.Float64),
                                     ("R_bc_tprev", pl.Float64),
                                     ("R_ca_tprev", pl.Float64),
                                     ("L_qa_tprev", pl.Int64),
                                     ("L_qb_tprev", pl.Int64),
                                     ("L_qc_tprev", pl.Int64),
                                     ("Lambda_qa", pl.Float64),
                                     ("Mu_qa", pl.Float64),
                                     ("Lambda_qb", pl.Float64),
                                     ("Mu_qb", pl.Float64),
                                     ("Lambda_qc", pl.Float64),
                                     ("Mu_qc", pl.Float64), 
                                     ("R_ab", pl.Float64),
                                     ("R_bc", pl.Float64),
                                     ("R_ca", pl.Float64),
                                     ("L_qa", pl.Int64),
                                     ("L_qb", pl.Int64),
                                     ("L_qc", pl.Int64)])  # Initialize the global dataframe          # pylint: disable=unexpected-keyword-arg,

    runs = df["Run"].max()  # Extract the number of runs in the file
    list_global_df = []  # List to store the time series data for each run

    for r in tqdm(range(1, runs +
                        1)):  # Construct sampling data for each run in the file
        d = df.filter(pl.col("Run") == r).select(df.columns)
        if d.is_empty():
            logger.debug(f"Run {r} is empty")
            continue

        arr_qa = float(d[0, "Lambda_qA"])  # Arrival rate at queue A
        ser_qa = float(d[0, "Mu_qA"])  # Service rate at queue A
        arr_qb = float(d[0, "Lambda_qB"])  # Arrival rate at queue B
        ser_qb = float(d[0, "Mu_qB"])  # Service rate at queue B
        arr_qc = float(d[0, "Lambda_qC"])  # Arrival rate at queue C
        ser_qc = float(d[0, "Mu_qC"])  # Service rate at queue C
        r_ab = float(d[0, "Rp_ab"])  # Routing probability from A to B
        r_bc = float(d[0, "Rp_bc"])  # Routing probability from B to C
        r_ca = float(d[0, "Rp_ca"])  # Routing probability from C to A

        # Initialize a new polars dataframe with the columns Lambda_ and L
        d1 = pl.DataFrame(schema=[
            ("Lambda_qa_tprev", pl.Float64),
            ("Mu_qa_tprev", pl.Float64),
            ("Lambda_qb_tprev", pl.Float64),
            ("Mu_qb_tprev", pl.Float64),
            ("Lambda_qc_tprev", pl.Float64),
            ("Mu_qc_tprev", pl.Float64),
            ("R_ab_tprev", pl.Float64),
            ("R_bc_tprev", pl.Float64),
            ("R_ca_tprev", pl.Float64),
            ("L_qa_tprev", pl.Float64),
            ("L_qb_tprev", pl.Float64),
            ("L_qc_tprev", pl.Float64),
        ])  # pylint: disable=unexpected-keyword-arg

        # For every time instance according to the sampling interval, find the queue length
        # Do not use np.arange or np.linspace, it produces floating point errors
        t = 0.0  # Start time = 0.0
        while t <= simulation_end:
            lower_idx = bisect(d.get_column("Time").to_numpy(), t) - 1
            if lower_idx < 0:
                event = d[0, "Event"]
                qa_ql = float(d[0, "QueueAL"])
                qb_ql = float(d[0, "QueueBL"])
                qc_ql = float(d[0, "QueueCL"])
                assert (
                    event == "Initialization"
                )  # If capturing initial queue length, ensure that initialization is recorded
            else:
                event = d[lower_idx, "Event"]
                qa_ql = float(d[lower_idx, "QueueAL"])
                qb_ql = float(d[lower_idx, "QueueBL"])
                qc_ql = float(d[lower_idx, "QueueCL"])

            time_series = pl.DataFrame({
                "Lambda_qa_tprev": [arr_qa],
                "Mu_qa_tprev": [ser_qa],
                "Lambda_qb_tprev": [arr_qb],
                "Mu_qb_tprev": [ser_qb],
                "Lambda_qc_tprev": [arr_qc],
                "Mu_qc_tprev": [ser_qc],
                "R_ab_tprev": [r_ab],
                "R_bc_tprev": [r_bc],
                "R_ca_tprev": [r_ca],
                "L_qa_tprev": [qa_ql],
                "L_qb_tprev": [qb_ql],
                "L_qc_tprev": [qc_ql],
            })
            d1 = d1.vstack(time_series)
            t = t + sampling_rate
            t = round(t, 4)

        df_2tbn = pl.DataFrame(
            schema=[('Lambda_qa_tprev',pl.Float64),
                    ('Mu_qa_tprev', pl.Float64),
                    ('Lambda_qb_tprev', pl.Float64),
                    ('Mu_qb_tprev', pl.Float64),
                    ('Lambda_qc_tprev', pl.Float64),
                    ('Mu_qc_tprev', pl.Float64),
                    ('R_ab_tprev', pl.Float64),
                    ('R_bc_tprev', pl.Float64),
                    ('R_ca_tprev', pl.Float64),
                    ('L_qa_tprev', pl.Float64),
                    ('L_qb_tprev', pl.Float64),
                    ('L_qc_tprev', pl.Float64),
                    ('Lambda_qa', pl.Float64),
                    ('Mu_qa', pl.Float64),
                    ('Lambda_qb', pl.Float64),
                    ('Mu_qb', pl.Float64),
                    ('Lambda_qc', pl.Float64),
                    ('Mu_qc', pl.Float64),
                    ('R_ab', pl.Float64),
                    ('R_bc', pl.Float64),
                    ('R_ca', pl.Float64),
                    ('L_qa', pl.Float64),
                    ('L_qb', pl.Float64),
                    ('L_qc', pl.Float64)])
        d2 = d1.clone()
        d2 = d2[1:, :]  # Remove the first row
        last_row_values = d2.row(len(d2) - 1)  # Get the last row values
        last_row = pl.DataFrame({
            'Lambda_qa_tprev': [last_row_values[0]],
            'Mu_qa_tprev': [last_row_values[1]],
            'Lambda_qb_tprev': [last_row_values[2]],
            'Mu_qb_tprev': [last_row_values[3]],
            'Lambda_qc_tprev': [last_row_values[4]],
            'Mu_qc_tprev': [last_row_values[5]],
            'R_ab_tprev': [last_row_values[6]],
            'R_bc_tprev': [last_row_values[7]],
            'R_ca_tprev': [last_row_values[8]],
            'L_qa_tprev': [last_row_values[9]],
            'L_qb_tprev': [last_row_values[10]],
            'L_qc_tprev': [last_row_values[11]],
        })
        d2 = d2.vstack(last_row)
        d2.columns = [
            'Lambda_qa', 'Mu_qa', 'Lambda_qb', 'Mu_qb',
            'Lambda_qc', 'Mu_qc', 'R_ab', 'R_bc', 'R_ca',
            'L_qa', 'L_qb', 'L_qc'
        ]
        d2 = d2[:len(d2) - 1, :]  # Remove the last row
        df_2tbn = pl.concat([d1, d2], how='horizontal')
        df_2tbn = df_2tbn.drop_nulls()

        list_global_df.append(
            df_2tbn)  # Append the time series data for each run to the list

    global_df = pl.concat(
        list_global_df,
        how="vertical")  # Concatenate the time series data for each run

    end = time.time()  # End time
    # Save the time series data to a file
    global_df.write_csv(output_filename)
    logger.info(
        f"Time taken to construct time series data: {end - start} seconds")


if __name__ == "__main__":
    # Input arguments
    # Example function call - python src/subsampling_discrete_2tbn.py
    # --config_file configs/time_discretization.yaml --experiment_number 1 -v
    parser = argparse.ArgumentParser(
        description=
        "Construct discrete time 2TBN data for the Markovian queueing system.")
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

    discrete_time_file = f"{output_folder}/discrete-time-2tbn-exp-{experiment_number}.csv"

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct time series data
    construct_2tbn_data(simulation_file, discrete_time_file,
                        simulation_end_time, sampling_interval)
