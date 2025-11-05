# pylint: disable=logging-fstring-interpolation
"""
Simulation model of a Markovian queueing network that is used
in the paper. It consists of three queues A, B and C.

Necessarily we have that A -> B -> C -> A. (There is feedback
in the network, but we also have probabilistic routing between 
B and C, as well as C and A).
All queues have exogenous arrivals, and exogenous departures as 
well as departures into the next queue.

Simulation input parameters: 
Queue A: Interarrival time mean 
Queue A: Service time mean
Queue B: Interarrival time mean
Queue B: Service time mean
Queue C: Interarrival time mean
Queue C: Service time mean 
R_AB: Probability of routing from A to B
R_BC: Probability of routing from B to C
R_CA: Probability of routing from C to A

Note that there are 9 input parameters, we ensure that each can 
take up to 2 values.
"""
# Import libraries
import warnings
import argparse
import time
import dataclasses
import logging
import os
import yaml

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class InputParameters:
    """Dataclass to store the input parameters of the simulation."""
    queue_a_mean_arrival_time: float
    queue_a_mean_service_time: float
    queue_b_mean_arrival_time: float
    queue_b_mean_service_time: float
    queue_c_mean_arrival_time: float
    queue_c_mean_service_time: float
    simulation_end: float

@dataclasses.dataclass
class NextEventTimes:
    """Dataclass to store the next event times."""
    t_arrival_queue_a: float
    t_departure_queue_a: float
    t_arrival_queue_b: float
    t_departure_queue_b: float
    t_arrival_queue_c: float
    t_departure_queue_c: float

@dataclasses.dataclass
class RoutingProbabilities:
    """Dataclass to store the routing probabilities."""
    prob_a_to_b: float
    prob_b_to_c: float
    prob_c_to_a: float

@dataclasses.dataclass
class ServerStates:
    """Dataclass to store the server states."""
    state_server_queue_a: bool
    state_server_queue_b: bool
    state_server_queue_c: bool


class QueueSimulation:
    """Class to simulate a Markovian queue network."""

    def __init__(self,
                 queue_a_mean_arrival_time: float,
                 queue_a_mean_service_time: float,
                 queue_b_mean_arrival_time: float,
                 queue_b_mean_service_time: float,
                 queue_c_mean_arrival_time: float,
                 queue_c_mean_service_time: float,
                 prob_a_to_b: float,
                 prob_b_to_c: float,
                 prob_c_to_a: float,
                 simulation_end: float,
                 initial_queue_length=0,
                 seed=0):
        self.clock = 0.0  # Simulation clock
        self.rng = np.random.default_rng(
            seed=seed)  # random number generator stream
        self.input_params = InputParameters(
            queue_a_mean_arrival_time=queue_a_mean_arrival_time,
            queue_a_mean_service_time=queue_a_mean_service_time,
            queue_b_mean_arrival_time=queue_b_mean_arrival_time,
            queue_b_mean_service_time=queue_b_mean_service_time,
            queue_c_mean_arrival_time=queue_c_mean_arrival_time,
            queue_c_mean_service_time=queue_c_mean_service_time,
            simulation_end=simulation_end)  # Input parameters
        self.routing_probs = RoutingProbabilities(
            prob_a_to_b=prob_a_to_b,
            prob_b_to_c=prob_b_to_c,
            prob_c_to_a=prob_c_to_a)    # Routing probabilities

        # Change the initialization depending on the initial queue length
        if initial_queue_length == 0:  # No varying initial queue length
            self.next_event_times = NextEventTimes(
                t_arrival_queue_a=self.clock +
                self.gen_int_arr(self.input_params.queue_a_mean_arrival_time),
                t_departure_queue_a=float('inf'),
                t_arrival_queue_b=self.clock +
                self.gen_int_arr(self.input_params.queue_b_mean_arrival_time),
                t_departure_queue_b=float('inf'),
                t_arrival_queue_c=self.clock +
                self.gen_int_arr(self.input_params.queue_c_mean_arrival_time),
                t_departure_queue_c=float('inf'))
            self.server_states = ServerStates(
                state_server_queue_a=False,
                state_server_queue_b=False,
                state_server_queue_c=False)  # Server states

            self.curr_num_in_queue_a = 0  # Current number of customers in queue a
            self.curr_num_in_queue_a_system = 0  # Number of people in the system a at any time
            self.curr_num_in_queue_b = 0  # Current number of customers in queue b
            self.curr_num_in_queue_b_system = 0  # Number of people in the system b at any time
            self.curr_num_in_queue_c = 0  # Current number of customers in queue c
            self.curr_num_in_queue_c_system = 0  # Number of people in the system c at any time

        elif initial_queue_length > 0:  # Varying initial queue length
            self.server_states = ServerStates(state_server_queue_a=True,
                                              state_server_queue_b=True,
                                              state_server_queue_c=True)
            self.curr_num_in_queue_a = initial_queue_length - 1
            self.curr_num_in_queue_a_system = initial_queue_length
            self.curr_num_in_queue_b = initial_queue_length - 1
            self.curr_num_in_queue_b_system = initial_queue_length
            self.curr_num_in_queue_c = initial_queue_length - 1
            self.curr_num_in_queue_c_system = initial_queue_length

            self.next_event_times = NextEventTimes(
                t_arrival_queue_a=self.clock +
                self.gen_int_arr(self.input_params.queue_a_mean_arrival_time),
                t_departure_queue_a=self.clock + self.gen_service_time(
                    self.input_params.queue_a_mean_service_time),
                t_arrival_queue_b=self.clock +
                self.gen_int_arr(self.input_params.queue_b_mean_arrival_time),
                t_departure_queue_b=self.clock + self.gen_service_time(
                    self.input_params.queue_b_mean_service_time),
                t_arrival_queue_c=self.clock +
                self.gen_int_arr(self.input_params.queue_c_mean_arrival_time),
                t_departure_queue_c=self.clock + self.gen_service_time(
                    self.input_params.queue_c_mean_service_time))

        # Record the time series of the simulation
        self.time_series = pd.DataFrame(columns=[
            "Time", "Event", "QueueAL", "QueueBL", "QueueCL"
        ])  # dataframe to record the events of the simulation
        self.log_query("Initialization")

    def time_adv(self):
        """Function to advance the time of the simulation."""
        event_times = [
            self.next_event_times.t_arrival_queue_a,
            self.next_event_times.t_departure_queue_a,
            self.next_event_times.t_arrival_queue_b,
            self.next_event_times.t_departure_queue_b,
            self.next_event_times.t_arrival_queue_c,
            self.next_event_times.t_departure_queue_c
        ]
        event_names = [
            "arrival_queue_a", "departure_queue_a", "arrival_queue_b",
            "departure_queue_b", "arrival_queue_c", "departure_queue_c"
        ]
        next_event_idx, next_event_time = min(enumerate(event_times),
                                              key=lambda x: x[1])
        next_event_name = event_names[next_event_idx]

        self.clock = next_event_time  # Reset the clock
        if next_event_time >= self.input_params.simulation_end:
            return True  # Stop the simulation if the end time is reached

        queue_id = next_event_name.split("_")[-1]
        if next_event_name in [
                "arrival_queue_a", "arrival_queue_b", "arrival_queue_c"
        ]:
            self.arrival(queue_id=queue_id)
        elif next_event_name in [
                "departure_queue_a", "departure_queue_b", "departure_queue_c"
        ]:
            self.departure(queue_id=queue_id)

    def arrival(self, queue_id):
        """Function to handle the arrival event into Queue referenced by queue_id.
        
        Note that these are only handling exogenous arrivals.
        
        Args:
            queue_id (str): The queue identifier.
        """
        # Depending on the queue_id, increment the number of people in the system
        setattr(self, f"curr_num_in_queue_{queue_id}_system",
                getattr(self, f"curr_num_in_queue_{queue_id}_system") + 1)

        # If the queue is empty and the server is idle, schedule the departure
        if getattr(self, f"curr_num_in_queue_{queue_id}") == 0 and not getattr(
                self.server_states, f"state_server_queue_{queue_id}"):
            setattr(self.server_states, f"state_server_queue_{queue_id}", True)
            service_time = self.gen_service_time(
                getattr(self.input_params,
                        f"queue_{queue_id}_mean_service_time"))
            setattr(self.next_event_times, f"t_departure_queue_{queue_id}",
                    self.clock + service_time)

        else:  # If the queue is not empty, increment the queue length
            setattr(self, f"curr_num_in_queue_{queue_id}",
                    getattr(self, f"curr_num_in_queue_{queue_id}") + 1)

        # Set the clock for the next exogenous arrival to the queue
        setattr(
            self.next_event_times, f"t_arrival_queue_{queue_id}",
            self.clock + self.gen_int_arr(
                getattr(self.input_params,
                        f"queue_{queue_id}_mean_arrival_time")))

        # Record the arrival event in the time series
        self.log_query(f"Arrival (exogenous) Queue {queue_id}")

    def departure(self, queue_id):
        """Function to handle the departure event from the Queue referenced by queue_id.
        
        Args:
            queue_id (str): The queue identifier.
        """
        # Decrease Queue length from the system
        setattr(self, f"curr_num_in_queue_{queue_id}_system",
                getattr(self, f"curr_num_in_queue_{queue_id}_system") - 1)

        # If the queue is not empty, schedule the next departure
        if getattr(self, f"curr_num_in_queue_{queue_id}") > 0:
            service_time = self.gen_service_time(
                getattr(self.input_params,
                        f"queue_{queue_id}_mean_service_time"))
            setattr(self.next_event_times, f"t_departure_queue_{queue_id}",
                    self.clock + service_time)
            setattr(self, f"curr_num_in_queue_{queue_id}",
                    getattr(self, f"curr_num_in_queue_{queue_id}") - 1)

        else:
            setattr(self.next_event_times, f"t_departure_queue_{queue_id}",
                    float('inf'))
            setattr(self.server_states, f"state_server_queue_{queue_id}", False)

        # Add the same job to the next queue with probability based on the routing probabilities
        # Determine what the next queue is, if it is the last queue, then add to the first queue
        next_queue_id = self.get_next_queue_id(queue_id)

        if self.rng.uniform() < getattr(self.routing_probs,
                                        f"prob_{queue_id}_to_{next_queue_id}"):
            # Do this every time there is a departure event
            setattr(
                self, f"curr_num_in_queue_{next_queue_id}_system",
                getattr(self, f"curr_num_in_queue_{next_queue_id}_system") + 1)

            # If the next queue is empty
            if getattr(
                    self,
                    f"curr_num_in_queue_{next_queue_id}") == 0 and not getattr(
                        self.server_states,
                        f"state_server_queue_{next_queue_id}"):
                setattr(self.server_states,
                        f"state_server_queue_{next_queue_id}", True)
                service_time = self.gen_service_time(
                    getattr(self.input_params,
                            f"queue_{next_queue_id}_mean_service_time"))
                setattr(self.next_event_times,
                        f"t_departure_queue_{next_queue_id}",
                        self.clock + service_time)
            else:  # If the next queue is not empty add to queue
                setattr(self, f"curr_num_in_queue_{next_queue_id}",
                        getattr(self, f"curr_num_in_queue_{next_queue_id}") + 1)

            # Record the arrival event in the time series
            self.log_query(f"Arrival (internal) Queue {next_queue_id}")
        else:
            self.log_query(f"Departure Queue {queue_id}")

    def gen_int_arr(self, mean_arrival_time):
        """Function to generate the interarrival times using a Poisson distribution."""
        return self.rng.exponential(scale=mean_arrival_time, size=1)[0]

    def gen_service_time(self, mean_service_time):
        """Function to generate the service times using a Poisson distribution."""
        return self.rng.exponential(scale=mean_service_time, size=1)[0]

    def get_next_queue_id(self, queue_id):
        """Function to get the next queue id in the network."""
        if queue_id == 'a':
            return 'b'
        elif queue_id == 'b':
            return 'c'
        elif queue_id == 'c':
            return 'a'

    def log_query(self, event_type):
        """Function to log the events of the simulation."""
        row = pd.Series([
            self.clock, event_type, self.curr_num_in_queue_a_system,
            self.curr_num_in_queue_b_system, self.curr_num_in_queue_c_system
        ],
                        index=self.time_series.columns)
        self.time_series.loc[len(self.time_series)] = row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate full factorial data from the specified parameters"
    )
    # Example call - python src/simulator.py
    # --config_file configs/simulator.yaml --experiment_number 1
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help=
        "Path to the configuration file (e.g. configs/simulator.yaml)",
        default="configs/simulator.yaml")
    parser.add_argument("--experiment_number",
                        "-e",
                        type=int,
                        help="Experiment number (e.g. 1)",
                        default=1)
    parser.add_argument('--verbose',
                        '-v',
                        help='Increase output verbosity',
                        action='store_true',
                        default=False,
                        required=False)

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
    experimental_design = config['experimental_design']
    mean_interarrival_rates_queue_a = config['queue_a_arrival_rates']
    mean_service_rates_queue_a = config['queue_a_service_rates']
    mean_interarrival_rates_queue_b = config['queue_b_arrival_rates']
    mean_service_rates_queue_b = config['queue_b_service_rates']
    mean_interarrival_rates_queue_c = config['queue_c_arrival_rates']
    mean_service_rates_queue_c = config['queue_c_service_rates']
    rp_a_to_b = config['routing_probabilities']['a_to_b']
    rp_b_to_c = config['routing_probabilities']['b_to_c']
    rp_c_to_a = config['routing_probabilities']['c_to_a']
    simulation_reps = config['replications']
    simulation_end_time = config['simulation_end']
    num_configs = config['configurations']
    varying_iql = config['varying_iql']
    max_iql = config['max_iql']
    output_folder = config['output_folder']

    # Print the parameters
    logger.info(f"Experimental Design: {experimental_design}")
    logger.info(
        f"Mean Interarrival Rate Queue A: {mean_interarrival_rates_queue_a}")
    logger.info(f"Mean Service Rate Queue A: {mean_service_rates_queue_a}")
    logger.info(
        f"Mean Interarrival Rate Queue B: {mean_interarrival_rates_queue_b}")
    logger.info(f"Mean Service Rate Queue B: {mean_service_rates_queue_b}")
    logger.info(
        f"Mean Interarrival Rate Queue C: {mean_interarrival_rates_queue_c}")
    logger.info(f"Mean Service Rate Queue C: {mean_service_rates_queue_c}")
    logger.info(
        f"Routing Probabilities: A to B: {rp_a_to_b}, B to C: {rp_b_to_c}, C to A: {rp_c_to_a}"
    )
    logger.info(f"Number of Replications: {simulation_reps}")
    logger.info(f"Simulation End Time: {simulation_end_time}")
    logger.info(f"Number of Configurations: {num_configs}")
    logger.info(f"Varying Initial Queue Length: {varying_iql}")
    logger.info(f"Max Initial Queue Length: {max_iql}")
    logger.info(f"Output Folder: {output_folder}")

    df_list = [
    ]  # Global dataframe that is used to log the events of the simulator
    start_time = time.time()
    RUN = 1

    # Different experimental designs can be used here
    if experimental_design == 'full-factorial':
        # Full factorial design
        for lambda_qa in tqdm(mean_interarrival_rates_queue_a,
                              desc=' queue a arrival rates',
                              position=0):
            for mu_qa in tqdm(mean_service_rates_queue_a,
                              desc=' queue a service rates',
                              position=1,
                              leave=False):
                for lambda_qb in tqdm(mean_interarrival_rates_queue_b,
                                      desc=' queue b arrival rates',
                                      position=2,
                                      leave=False):
                    for mu_qb in tqdm(mean_service_rates_queue_b,
                                      desc=' queue b service rates',
                                      position=3,
                                      leave=False):
                        for lambda_qc in tqdm(mean_interarrival_rates_queue_c,
                                            desc=' queue c arrival rates',
                                            position=4,
                                            leave=False):
                            for mu_qc in tqdm(mean_service_rates_queue_c,
                                                desc=' queue c service rates',
                                                position=5,
                                                leave=False):
                                for r_ab in tqdm(rp_a_to_b,
                                                desc=' routing probabilities a to b',
                                                position=6,
                                                leave=False):
                                    for r_bc in tqdm(rp_b_to_c,
                                                    desc =' routing probabilities b to c',
                                                        position=7,
                                                        leave=False):
                                        for r_ca in tqdm(rp_c_to_a,
                                                        desc=' routing probabilities c to a',
                                                        position=8,
                                                        leave=False):
                                            if varying_iql:
                                                for iql in range(0, max_iql + 1):
                                                    logger.debug(
                                                        f"Running simulation for lambda_qa={lambda_qa},"
                                                        f"mu_qa={mu_qa}, lambda_qb={lambda_qb},"
                                                        f"mu_qb={mu_qb}, lambda_qc={lambda_qc},"
                                                        f"mu_qc={mu_qc}, iql={iql}")
                                                    for i in range(simulation_reps):
                                                        q = QueueSimulation(
                                                            (1.0 / lambda_qa), (1.0 / mu_qa),
                                                            (1.0 / lambda_qb), (1.0 / mu_qb),
                                                            (1.0 / lambda_qc), (1.0 / mu_qc),
                                                            r_ab, r_bc, r_ca,
                                                            simulation_end_time,
                                                            initial_queue_length=iql,
                                                            seed=i + 123)

                                                        while True:
                                                            CHECK_SIMEND = q.time_adv()
                                                            if CHECK_SIMEND is True:
                                                                break

                                                        df_list.append(
                                                            q.time_series.assign(
                                                                Run=RUN,
                                                                Lambda_qA=round(lambda_qa, 2),
                                                                Mu_qA=round(mu_qa, 2),
                                                                Lambda_qB=round(lambda_qb, 2),
                                                                Mu_qB=round(mu_qb, 2),
                                                                Lambda_qC=round(lambda_qc, 2),
                                                                Mu_qC=round(mu_qc, 2),
                                                                Rp_ab = round(r_ab, 2),
                                                                Rp_bc = round(r_bc, 2),
                                                                Rp_ca = round(r_ca, 2),
                                                                End=simulation_end_time))
                                                        RUN += 1
                                            else:
                                                logger.debug(
                                                        f"Running simulation for lambda_qa={lambda_qa},"
                                                        f"mu_qa={mu_qa}, lambda_qb={lambda_qb},"
                                                        f"mu_qb={mu_qb}, lambda_qc={lambda_qc},"
                                                        f"mu_qc={mu_qc}, iql={0}")
                                                for i in range(simulation_reps):
                                                    q = QueueSimulation(
                                                        (1.0 / lambda_qa), (1.0 / mu_qa),
                                                        (1.0 / lambda_qb), (1.0 / mu_qb),
                                                        (1.0 / lambda_qc), (1.0 / mu_qc),
                                                        r_ab, r_bc, r_ca,
                                                        simulation_end_time,
                                                        initial_queue_length=0,
                                                        seed=i + 123)

                                                    while True:
                                                        CHECK_SIMEND = q.time_adv()
                                                        if CHECK_SIMEND is True:
                                                            break

                                                    df_list.append(
                                                        q.time_series.assign(
                                                            Run=RUN,
                                                            Lambda_qA=round(lambda_qa, 2),
                                                            Mu_qA=round(mu_qa, 2),
                                                            Lambda_qB=round(lambda_qb, 2),
                                                            Mu_qB=round(mu_qb, 2),
                                                            Lambda_qC=round(lambda_qc, 2),
                                                            Mu_qC=round(mu_qc, 2),
                                                            Rp_ab = round(r_ab, 2),
                                                            Rp_bc = round(r_bc, 2),
                                                            Rp_ca = round(r_ca, 2),
                                                            End=simulation_end_time))
                                                    RUN += 1

    logger.info(
        f"Total number of simulation runs (with replications) = {RUN-1}")

    df = pd.concat(df_list, ignore_index=True)
    df = df[[
        "Run", "Lambda_qA", "Mu_qA", "Lambda_qB", "Mu_qB",
        "Lambda_qC", "Mu_qC", "Rp_ab", "Rp_bc", "Rp_ca", "End", "Time",
        "Event", "QueueAL", "QueueBL", "QueueCL"
    ]]
    end_time = time.time()
    logger.info(
        f"Time to run the simulation in Python3: {end_time - start_time} seconds"
    )

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    TIMESERIES_FILEPATH = f"{config['output_folder']}/time-series-exp-{experiment_number}.csv"

    logger.info(f'Writing the time series to {TIMESERIES_FILEPATH}')
    df.to_csv(TIMESERIES_FILEPATH, index=False)  # Write results to csv
