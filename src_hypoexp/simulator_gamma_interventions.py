# pylint: disable=logging-fstring-interpolation
"""
Simulation model of an G/M/1 queue with a Gamma distribution.
Input parameters: 
    Gamma: 
        alpha: shape
        theta: scale
    Exponential:
        mu: mean service rate
"""
import warnings
import os
import argparse
import time
import dataclasses
import logging
from tqdm import tqdm
import yaml

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger('simulator_interventions_logger')

@dataclasses.dataclass
class InputParameters:
    """Dataclass to store the input parameters of the simulation."""
    alpha: float
    theta: float
    mean_service_rate: float
    simulation_end: float

@dataclasses.dataclass
class ServerStates:
    """Dataclass to store the server states."""
    state_server: bool

class GammaM1Simulation:
    """Class to simulate a Gamma/M/1 queue."""

    def __init__(self,
                 simulation_end,
                 query,
                 seed=0):
        self.clock = 0.0  # Simulation clock
        self.rng = np.random.default_rng(
            seed=seed)  # random number generator stream
        self.input_params = InputParameters(
            alpha = query['start_parameters']['Alpha'],
            theta = query['start_parameters']['Theta'],
            mean_service_rate = query['start_parameters']['Mu'],
            simulation_end=simulation_end)  # Input parameters
        self.server_states = ServerStates(
            state_server=False,
        )  # Server states initialized to default 0

        # Initialize and define the curr_num_in_queue variables
        self.curr_num_in_queue_system = 0
        self.curr_num_in_queue = 0

        # Initialize the arrival and departure times for the queues
        # They will then be overridden by the actual values in the query
        self.t_arrival = self.clock + self.gen_int_arr(
            self.input_params.alpha, self.input_params.theta)
        self.t_departure = float('inf')

        # Set the initial queue lengths, as we want to be able to test
        # queries where it can be non zero.
        if query['start_parameters']['QueueLength'] > 0:
            self.curr_num_in_queue_system = query['start_parameters']['QueueLength']
            self.curr_num_in_queue = self.curr_num_in_queue_system - 1
            self.t_departure = self.clock + self.gen_service_time(
                getattr(self.input_params, 'mean_service_rate'))
            self.server_states.state_server = True

        # Record the time series of the simulation
        self.time_series = pd.DataFrame(columns=[
            "Alpha", "Theta", "Mu",
            "Time", "Event", "QueueLength"
        ])  # dataframe to record the events of the simulation
        row = pd.Series([
            self.input_params.alpha,
            self.input_params.theta,
            self.input_params.mean_service_rate,
            0.0, "Initialization",
            self.curr_num_in_queue_system
        ],
                        index=self.time_series.columns)
        self.time_series.loc[len(self.time_series)] = row

        # Initialize the evidence times and values
        self.evidence_times = []
        self.evidence_values = []
        self.evidence_types = []
        self.evidence_variables = []
        # Add in the evidence times, if not empty
        if query['interventions']:
            all_interventions = sorted(query['interventions'],
                                       key=lambda k: k['intervention_start'])
            for iv in all_interventions:
                self.evidence_times.append(iv['intervention_start'])
                self.evidence_values.append(iv['intervention_value'])
                self.evidence_types.append(iv['intervention_type'])
                self.evidence_variables.append(iv['intervention_variable'])

    def time_adv(self):
        """Function to advance the time of the simulation."""
        event_times = [
            self.t_arrival, self.t_departure,
            *self.evidence_times
        ]  # Order matters: q_arr, q_dep, evidence
        next_event_idx, next_event_time = min(enumerate(event_times),
                                              key=lambda x: x[1])
        logger.debug(f'Event times: {event_times}')
        logger.debug(
            f'Next event index: {next_event_idx}, Next event time: {next_event_time}'
        )
        # logger.debug(f'Evidence times: {self.evidence_times}')
        # logger.debug(f'Evidence values: {self.evidence_values}')
        # logger.debug(f'Evidence types: {self.evidence_variables}')

        # If the next event is an intervention, then we need to remove it for future time steps
        if next_event_idx > 1:
            evidence_val = self.evidence_values.pop(next_event_idx - 2)
            evidence_type = self.evidence_types.pop(next_event_idx - 2)
            evidence_var = self.evidence_variables.pop(next_event_idx - 2)
            self.evidence_times.pop(next_event_idx - 2)
        else: 
            evidence_val = None
            evidence_type = None
            evidence_var = None

        self.clock = next_event_time  # Reset the clock

        if next_event_time >= self.input_params.simulation_end:
            self.log_query("Simulation End")
            logger.debug(f"Simulation ended, not completed: {event_times}")
            return True  # Stop the simulation if the end time is reached

        if next_event_idx == 0: 
            self.arrival()
        elif next_event_idx == 1:
            self.departure()
        elif next_event_idx > 1:
            logger.debug(f'Intervention type: {evidence_type}')
            if evidence_type == 'conditional':
                self.queue_condition(evidence_var, evidence_val)
            elif evidence_type == 'interventional':
                self.queue_intervention(evidence_var, evidence_val)
            elif evidence_type == 'additive' or evidence_type == 'subtractive':
                self.queue_modintervention(evidence_var, evidence_val,
                                           evidence_type)
            elif evidence_type == 'parameter_intervention':
                self.parameter_intervention(evidence_var, evidence_val)

    def arrival(self):
        """Function to handle the arrival event into the queue specified by the queue_id."""
        # Depending on the queue_id, increment the number of people in the system
        self.curr_num_in_queue_system += 1

        # If the queue is empty and the server is idle, schedule the departure
        if self.curr_num_in_queue == 0 and not self.server_states.state_server: 
            self.server_states.state_server = True
            service_time = self.gen_service_time(self.input_params.mean_service_rate)
            self.t_departure = self.clock + service_time
        else:  # If the queue is not empty, increment the queue length
            self.curr_num_in_queue += 1

        # Set the clock for the next exogenous arrival to the queue
        self.t_arrival = self.clock + self.gen_int_arr(self.input_params.alpha, self.input_params.theta)

        # Record the arrival event in the time series
        self.log_query(f"Arrival")

    def departure(self):
        """Function to handle the departure event from the Queue referenced by queue_id.
        
        Args:
            queue_id (str): The queue identifier.
        """
        # Decrease Queue length from the system
        self.curr_num_in_queue_system -= 1

        # If the queue is not empty, schedule the next departure
        if self.curr_num_in_queue > 0: 
            service_time = self.gen_service_time(self.input_params.mean_service_rate)
            self.t_departure = self.clock + service_time
            self.curr_num_in_queue -= 1
        else: 
            self.t_departure = float('inf')
            self.server_states.state_server = False   

        self.log_query(f"Departure")

    def gen_int_arr(self, alpha, theta):
        """Function to generate the interarrival times using a Gamma distribution."""
        return self.rng.gamma(shape=alpha, scale=theta, size=1)[0] # TODO: Check correct rate vs. inverse

    def gen_service_time(self, mean_service_rate):
        """Function to generate the service times using a Poisson distribution."""
        return self.rng.exponential(scale=(1.0 / mean_service_rate), size=1)[0] # TODO: Check rate vs. inverse

    def log_query(self, event_type):
        """Function to log the events of the simulation."""
        row = pd.Series([
            self.input_params.alpha,
            self.input_params.theta,
            self.input_params.mean_service_rate,
            self.clock, event_type,
            self.curr_num_in_queue_system
        ],
                        index=self.time_series.columns)
        self.time_series.loc[len(self.time_series)] = row

    def parameter_intervention(self, evidence_var, evidence_val):
        """Function to change parameters when there is an intervention."""
        logger.debug(f'Parameter Intervention: {evidence_var} = {evidence_val}')
        logger.debug(f'Current parameters: {self.input_params}')
        if evidence_var == f'alpha' or evidence_var == f'theta':
            setattr(self.input_params, evidence_var, evidence_val)
            if self.t_arrival >= self.clock:
                self.t_arrival = self.clock + self.gen_int_arr(
                    self.input_params.alpha, self.input_params.theta)
        elif evidence_var == f'Mu':
            self.input_params.mean_service_rate = evidence_val
            if self.t_departure != float('inf'):
                self.t_departure = self.clock + self.gen_service_time(
                    self.input_params.mean_service_rate)
        self.log_query("Parameter Intervention")

    def queue_condition(self, evidence_var, evidence_val):
        """Function to check whether the queue condition is satisfied."""
        if evidence_var == 'L':
            if self.curr_num_in_queue_system == evidence_val:
                self.log_query("Conditional")

    def queue_intervention(self, evidence_var, evidence_val):
        """Function to intervene on the queue lengths."""
        if evidence_var == f'L':
            if evidence_val == 0:
                # The system has to be cleared immediately, current service is dropped
                self.curr_num_in_queue = 0 
                self.curr_num_in_queue_system = 0
                self.server_states.state_server = False
                self.t_departure = float('inf')
                self.t_arrival = self.clock + self.gen_int_arr(
                    self.input_params.alpha, self.input_params.theta)
            elif evidence_val > 0:
                # Check if there are any active jobs being serviced in the system
                # and intervene accordingly
                if self.server_states.state_server:
                    self.curr_num_in_queue = max(evidence_val - 1, 0)
                    self.curr_num_in_queue_system = evidence_val
                    self.t_arrival = self.clock + self.gen_int_arr(
                        self.input_params.alpha, self.input_params.theta)
                else:
                    # Service the new job if the parent is idle
                    self.server_states.state_server = True
                    self.t_departure = self.clock + self.gen_service_time(
                        self.input_params.mean_service_rate)
                    self.curr_num_in_queue = max(evidence_val - 1, 0)
                    self.curr_num_in_queue_system = evidence_val
                    self.t_arrival = self.clock + self.gen_int_arr(
                        self.input_params.alpha, self.input_params.theta)
        self.log_query("Intervention")

    def queue_modintervention(self, evidence_var, evidence_val, evidence_type):
        """Function to add or subtract from the queue lengths."""
        logger.debug('Values of the system before intervention')
        logger.debug(f'Queue system: {self.curr_num_in_queue_system}')
        if evidence_var == 'L':
            if evidence_type == 'additive':
                if self.curr_num_in_queue_system == 0:
                    self.curr_num_in_queue_system = evidence_val
                    self.curr_num_in_queue = self.curr_num_in_queue_system - 1
                    self.server_states.state_server = True
                    self.t_departure = self.clock + self.gen_service_time(
                        self.input_params.mean_service_rate)
                    self.t_arrival = self.clock + self.gen_int_arr(
                        self.input_params.alpha, self.input_params.theta)
                else: 
                    self.curr_num_in_queue_system += evidence_val
                    self.curr_num_in_queue += evidence_val - 1
                    self.t_arrival = self.clock + self.gen_int_arr(
                        self.input_params.alpha, self.input_params.theta)
            elif evidence_type == 'subtractive':
                if self.curr_num_in_queue_system == 0:
                    pass
                elif self.curr_num_in_queue_system <= evidence_val:
                    # Remove all jobs
                    self.curr_num_in_queue = 0
                    self.curr_num_in_queue_system = 0
                    self.server_states.state_server = False
                    self.t_departure = float('inf')
                    self.t_arrival = self.clock + self.gen_int_arr(
                        self.input_params.alpha, self.input_params.theta)
                elif self.curr_num_in_queue_system - evidence_val == 1:  
                    self.curr_num_in_queue = 0
                    self.curr_num_in_queue_system = 1
                    self.server_states.state_server = True
                    self.t_arrival = self.clock + self.gen_int_arr(
                        self.input_params.alpha, self.input_params.theta)
                else:  # Remove the specified number of jobs from the system
                    self.curr_num_in_queue -= evidence_val - 1
                    self.curr_num_in_queue_system -= evidence_val
                    self.t_arrival = self.clock + self.gen_int_arr(
                        self.input_params.alpha, self.input_params.theta)
        self.log_query("Mod Intervention")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Generate interventional data for the Markovian queue network.")
    # Example call - python src/simulator_gamma_interventions.py
    # --config_file configs/queries.json --experiment_number 1 -v

    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help=
        "Path to the configuration file (e.g. configs/queries.json)",
        default="config/queries.json")
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
    parser.add_argument('--gt_folder',
                        '-g',
                        help='Folder to store the ground truth results',
                        type=str,
                        default=None,
                        required=False)

    args = parser.parse_args()
    config_file = args.config_file
    experiment_number = args.experiment_number

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load the evidence file
    with open(config_file, 'r', encoding='utf-8') as file:
        query_data = yaml.safe_load(file)
    query_details = query_data[f'experiment_{experiment_number}']
    
    alpha = query_details['start_parameters']['Alpha']
    theta = query_details['start_parameters']['Theta']
    mean_service_rate = query_details['start_parameters']['Mu']
    simulation_end_time = query_details['query_time']
    gt_replications = query_details['gt_replications']

    df_list = [
    ]  # Global dataframe that is used to log the events of the simulator
    start_time = time.time()
    RUN = 1

    for i in tqdm(range(gt_replications)):
        q = GammaM1Simulation(simulation_end=simulation_end_time,
                            query=query_details,
                            seed=i + 123)

        while True:
            CHECK_SIMEND = q.time_adv()
            if CHECK_SIMEND is True:
                logger.debug(f"### Simulation run {RUN} completed ###")
                break

        df_list.append(q.time_series.assign(Run=RUN, End=simulation_end_time))
        RUN += 1

    logger.info(
        f"Total number of simulation runs (with replications) = {RUN-1}")

    df = pd.concat(df_list, ignore_index=True)
    df = df[[
        "Run", "Alpha", "Theta", "Mu",
        "Time", "Event", "QueueLength"
    ]]
    end_time = time.time()

    logger.info(
        f"Time to run the simulation in Python3: {end_time - start_time} seconds"
    )

    if args.gt_folder != None:
        gt_folder = f"{query_details['gt_results_folder']}/{args.gt_folder}"
    else:
        gt_folder = f"{query_details['gt_results_folder']}/{query_details['expt_name']}"

    # Create the directory if it does not exist
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)

    TIMESERIES_FILEPATH = f"{gt_folder}/gt-exp-{experiment_number}.csv"
    df.to_csv(TIMESERIES_FILEPATH, index=False)  # Write results to csv
