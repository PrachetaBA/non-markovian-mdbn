# pylint: disable=logging-fstring-interpolation
"""
Simulation model of an Markovian queueing network with 3 queues.
Input parameters: 
Queue A interarrival time mean 
Queue A service time mean
Queue B interarrival time mean
Queue B service time mean
Queue A interarrival time mean
Queue C service time mean
Routing probability from A to B
Routing probability from B to C
Routing probability from C to A

Events: 
1. Exogenous arrivals 
2. Exogenous departures
3. Internal transitions
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
    queue_a_mean_arrival_time: float
    queue_a_mean_service_time: float
    queue_b_mean_arrival_time: float
    queue_b_mean_service_time: float
    queue_c_mean_arrival_time: float
    queue_c_mean_service_time: float
    simulation_end: float

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
    """Class to simulate a tandem queue network."""

    def __init__(self,
                 simulation_end,
                 query,
                 seed=0):
        self.clock = 0.0  # Simulation clock
        self.rng = np.random.default_rng(
            seed=seed)  # random number generator stream
        self.input_params = InputParameters(
            queue_a_mean_arrival_time=query['start_parameters']['Lambdaqa'],
            queue_a_mean_service_time=query['start_parameters']['Muqa'],
            queue_b_mean_arrival_time=query['start_parameters']['Lambdaqb'],
            queue_b_mean_service_time=query['start_parameters']['Muqb'],
            queue_c_mean_arrival_time=query['start_parameters']['Lambdaqc'],
            queue_c_mean_service_time=query['start_parameters']['Muqc'],
            simulation_end=simulation_end)  # Input parameters
        self.routing_probs = RoutingProbabilities(
            prob_a_to_b=query['start_parameters']['Rab'],
            prob_b_to_c=query['start_parameters']['Rbc'],
            prob_c_to_a=query['start_parameters']['Rca']) # Routing probabilities
        self.server_states = ServerStates(
            state_server_queue_a=False,
            state_server_queue_b=False,
            state_server_queue_c=False
        )  # Server states initialized to default 0

        # Initialize and define the curr_num_in_queue variables
        self.curr_num_in_queue_a_system = 0
        self.curr_num_in_queue_b_system = 0
        self.curr_num_in_queue_c_system = 0

        self.curr_num_in_queue_a = 0
        self.curr_num_in_queue_b = 0
        self.curr_num_in_queue_c = 0

        # Initialize the arrival and departure times for the queues
        # They will then be overridden by the actual values in the query
        self.t_arrival_queue_a = self.clock + self.gen_int_arr(
            getattr(self.input_params, 'queue_a_mean_arrival_time'))
        self.t_departure_queue_a = float('inf')
        self.t_arrival_queue_b = self.clock + self.gen_int_arr(
            getattr(self.input_params, 'queue_b_mean_arrival_time'))
        self.t_departure_queue_b = float('inf')
        self.t_arrival_queue_c = self.clock + self.gen_int_arr(
            getattr(self.input_params, 'queue_c_mean_arrival_time'))
        self.t_departure_queue_c = float('inf')

        # Set the initial queue lengths, as we want to be able to test
        # queries where it can be non zero.
        if query['start_parameters']['Lqa'] > 0:
            self.curr_num_in_queue_a_system = query['start_parameters']['Lqa']
            self.curr_num_in_queue_a = self.curr_num_in_queue_a_system - 1
            self.t_departure_queue_a = self.clock + self.gen_service_time(
                getattr(self.input_params, 'queue_a_mean_service_time'))
            self.server_states.state_server_queue_a = True
        if query['start_parameters']['Lqb'] > 0:
            self.curr_num_in_queue_b_system = query['start_parameters']['Lqb']
            self.curr_num_in_queue_b = self.curr_num_in_queue_b_system - 1
            self.t_departure_queue_b = self.clock + self.gen_service_time(
                getattr(self.input_params, 'queue_b_mean_service_time'))
            self.server_states.state_server_queue_b = True
        if query['start_parameters']['Lqc'] > 0:
            self.curr_num_in_queue_c_system = query['start_parameters']['Lqc']
            self.curr_num_in_queue_c = self.curr_num_in_queue_c_system - 1
            self.t_departure_queue_c = self.clock + self.gen_service_time(
                getattr(self.input_params, 'queue_c_mean_service_time'))
            self.server_states.state_server_queue_c = True

        # Record the time series of the simulation
        self.time_series = pd.DataFrame(columns=[
            "Lambda_qA", "Mu_qA", "Lambda_qB", "Mu_qB",
            "Lambda_qC", "Mu_qC",
            "Rp_ab", "Rp_bc", "Rp_ca",
            "Time", "Event", "QueueAL", "QueueBL", "QueueCL"
        ])  # dataframe to record the events of the simulation
        row = pd.Series([
            self.input_params.queue_a_mean_arrival_time,
            self.input_params.queue_a_mean_service_time,
            self.input_params.queue_b_mean_arrival_time,
            self.input_params.queue_b_mean_service_time,
            self.input_params.queue_c_mean_arrival_time,
            self.input_params.queue_c_mean_service_time,
            self.routing_probs.prob_a_to_b,
            self.routing_probs.prob_b_to_c,
            self.routing_probs.prob_c_to_a,
            0.0, "Initialization",
            self.curr_num_in_queue_a_system, self.curr_num_in_queue_b_system,
            self.curr_num_in_queue_c_system
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
        # print("Simulator initialized")

    def time_adv(self):
        """Function to advance the time of the simulation."""
        event_times = [
            self.t_arrival_queue_a, self.t_departure_queue_a,
            self.t_arrival_queue_b, self.t_departure_queue_b,
            self.t_arrival_queue_c, self.t_departure_queue_c,
            *self.evidence_times
        ]  # Order matters: qa_arr, qa_dep, qa_arr, qb_dep, qc_arr, qc_dep, evidence
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
        if next_event_idx > 5:
            evidence_val = self.evidence_values.pop(next_event_idx - 6)
            evidence_type = self.evidence_types.pop(next_event_idx - 6)
            evidence_var = self.evidence_variables.pop(next_event_idx - 6)
            self.evidence_times.pop(next_event_idx - 6)

        self.clock = next_event_time  # Reset the clock

        if next_event_time >= self.input_params.simulation_end:
            self.log_query("Simulation End")
            logger.debug(f"Simulation ended, not completed: {event_times}")
            return True  # Stop the simulation if the end time is reached

        # Extract the queue_id from the event index
        if next_event_idx in [0, 1]:
            queue_id = 'a'
        elif next_event_idx in [2, 3]:
            queue_id = 'b'
        elif next_event_idx in [4, 5]:
            queue_id = 'c'

        if next_event_idx in [0, 2, 4]:
            self.arrival(queue_id=queue_id)
        elif next_event_idx in [1, 3, 5]:
            self.departure(queue_id=queue_id)
        elif next_event_idx > 5:
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

    def arrival(self, queue_id):
        """Function to handle the arrival event into the queue specified by the queue_id."""
        # Depending on the queue_id, increment the number of people in the system
        setattr(self, f"curr_num_in_queue_{queue_id}_system",
                (getattr(self, f"curr_num_in_queue_{queue_id}_system") + 1))

        # If the queue is empty and the server is idle, schedule the departure
        if getattr(self, f"curr_num_in_queue_{queue_id}") == 0 and not getattr(
                self.server_states, f"state_server_queue_{queue_id}"):
            setattr(self.server_states, f"state_server_queue_{queue_id}", True)
            service_time = self.gen_service_time(
                getattr(self.input_params,
                        f"queue_{queue_id}_mean_service_time"))
            setattr(self, f"t_departure_queue_{queue_id}",
                    (self.clock + service_time))

        else:  # If the queue is not empty, increment the queue length
            setattr(self, f"curr_num_in_queue_{queue_id}",
                    (getattr(self, f"curr_num_in_queue_{queue_id}") + 1))

        # Set the clock for the next exogenous arrival to the queue
        setattr(
            self, f"t_arrival_queue_{queue_id}", self.clock + self.gen_int_arr(
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
                (getattr(self, f"curr_num_in_queue_{queue_id}_system") - 1))

        # If the queue is not empty, schedule the next departure
        if getattr(self, f"curr_num_in_queue_{queue_id}") > 0:
            service_time = self.gen_service_time(
                getattr(self.input_params,
                        f"queue_{queue_id}_mean_service_time"))
            setattr(self, f"t_departure_queue_{queue_id}",
                    (self.clock + service_time))
            setattr(self, f"curr_num_in_queue_{queue_id}",
                    (getattr(self, f"curr_num_in_queue_{queue_id}") - 1))

        else:
            setattr(self, f"t_departure_queue_{queue_id}", float('inf'))
            setattr(self.server_states, f"state_server_queue_{queue_id}", False)

        # Add the same job to the next queue with probability based on the routing probabilities
        # Determine what the next queue is, if it is the last queue, then add to the first queue
        next_queue_id = self.get_next_queue_id(queue_id)

        if self.rng.uniform() < getattr(self.routing_probs,
                                        f"prob_{queue_id}_to_{next_queue_id}"):
            # Do this every time there is a departure event
            setattr(
                self, f"curr_num_in_queue_{next_queue_id}_system",
                (getattr(self, f"curr_num_in_queue_{next_queue_id}_system") + 1))

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
                setattr(self, f"t_departure_queue_{next_queue_id}",
                        (self.clock + service_time))
            else:  # If the next queue is not empty add to queue
                setattr(self, f"curr_num_in_queue_{next_queue_id}",
                        (getattr(self, f"curr_num_in_queue_{next_queue_id}") + 1))

            # Record the arrival event in the time series
            self.log_query(f"Arrival (internal) Queue {next_queue_id}")
        else:
            self.log_query(f"Departure Queue {queue_id}")

    def gen_int_arr(self, mean_interarrival_time):
        """Function to generate the interarrival times using a Poisson distribution."""
        return self.rng.exponential(scale=(1.0 / mean_interarrival_time),
                                    size=1)[0]

    def gen_service_time(self, mean_service_time):
        """Function to generate the service times using a Poisson distribution."""
        return self.rng.exponential(scale=(1.0 / mean_service_time), size=1)[0]

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
            self.input_params.queue_a_mean_arrival_time,
            self.input_params.queue_a_mean_service_time,
            self.input_params.queue_b_mean_arrival_time,
            self.input_params.queue_b_mean_service_time,
            self.input_params.queue_c_mean_arrival_time,
            self.input_params.queue_c_mean_service_time,
            self.routing_probs.prob_a_to_b,
            self.routing_probs.prob_b_to_c,
            self.routing_probs.prob_c_to_a,
            self.clock, event_type,
            self.curr_num_in_queue_a_system, self.curr_num_in_queue_b_system,
            self.curr_num_in_queue_c_system
        ],
                        index=self.time_series.columns)
        self.time_series.loc[len(self.time_series)] = row

    def parameter_intervention(self, evidence_var, evidence_val):
        """Function to change parameters when there is an intervention."""
        logger.debug(f'Parameter Intervention: {evidence_var} = {evidence_val}')
        logger.debug(f'Current parameters: {self.input_params}')
        # Extract the queue_id from the evidence variable
        if evidence_var in ['Lambdaqa', 'Muqa', 'Lambdaqb', 'Muqb', 
                            'Lambdaqc', 'Muqc']:
            queue_id = evidence_var[-1]
        elif evidence_var in ['Rab', 'Rbc', 'Rca']:
            queue_id = None
        if evidence_var == f'Lambdaq{queue_id}':
            setattr(self.input_params, f'queue_{queue_id}_mean_arrival_time',
                    evidence_val)
            # Update the next arrival time based on the new interarrival time
            # for all events that occur after the intervention
            if getattr(self, f't_arrival_queue_{queue_id}') >= self.clock:
                setattr(self, f't_arrival_queue_{queue_id}',
                        (self.clock + self.gen_int_arr(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_arrival_time'))))
        elif evidence_var == f'Muq{queue_id}':
            setattr(self.input_params, f'queue_{queue_id}_mean_service_time',
                    evidence_val)
            # Update the next departure time based on the new service time,
            # if the job is being served
            if getattr(self, f't_departure_queue_{queue_id}') != float('inf'):
                setattr(self, f't_departure_queue_{queue_id}',
                        (self.clock + self.gen_service_time(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_service_time'))))
        elif evidence_var == 'Rab': 
            setattr(self.routing_probs, 'prob_a_to_b', evidence_val)
        elif evidence_var == 'Rbc':
            setattr(self.routing_probs, 'prob_b_to_c', evidence_val)
        elif evidence_var == 'Rca':
            setattr(self.routing_probs, 'prob_c_to_a', evidence_val)
        logger.debug(f'Updated parameters: {self.input_params}')
        logger.debug(f'Updated routing probabilities: {self.routing_probs}')
        self.log_query("Parameter Intervention")

    def queue_condition(self, evidence_var, evidence_val):
        """Function to check whether the queue condition is satisfied."""
        if evidence_var == 'Lqa':
            if self.curr_num_in_queue_a_system == evidence_val:
                self.log_query("Conditional")
        elif evidence_var == 'Lqb':
            if self.curr_num_in_queue_b_system == evidence_val:
                self.log_query("Conditional")
        elif evidence_var == 'Lqc':
            if self.curr_num_in_queue_c_system == evidence_val:
                self.log_query("Conditional")

    def queue_intervention(self, evidence_var, evidence_val):
        """Function to intervene on the queue lengths."""
        # Determine the queue_id from evidence_var
        queue_id = evidence_var[-1]

        if evidence_var == f'Lq{queue_id}':
            if evidence_val == 0:
                # The system has to be cleared immediately, current service is dropped
                setattr(self, f'curr_num_in_queue_{queue_id}', 0)
                setattr(self, f'curr_num_in_queue_{queue_id}_system', 0)
                setattr(self.server_states, f'state_server_queue_{queue_id}',
                        False)
                setattr(self, f't_departure_queue_{queue_id}', float('inf'))
                setattr(
                    self, f't_arrival_queue_{queue_id}',
                    self.clock + self.gen_int_arr(
                        getattr(self.input_params,
                                f'queue_{queue_id}_mean_arrival_time')))
            elif evidence_val > 0:
                # Check if there are any active jobs being serviced in the system
                # and intervene accordingly
                if getattr(self.server_states,
                           f'state_server_queue_{queue_id}'):
                    setattr(self, f'curr_num_in_queue_{queue_id}',
                            max(evidence_val - 1, 0))
                    setattr(self, f'curr_num_in_queue_{queue_id}_system',
                            evidence_val)
                    setattr(
                        self, f't_arrival_queue_{queue_id}',
                        self.clock + self.gen_int_arr(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_arrival_time')))
                else:
                    # Service the new job if the parent is idle
                    setattr(self.server_states,
                            f'state_server_queue_{queue_id}', True)
                    setattr(
                        self, f't_departure_queue_{queue_id}',
                        self.clock + self.gen_service_time(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_service_time')))
                    # Add the remaining jobs to the queue
                    setattr(self, f'curr_num_in_queue_{queue_id}',
                            max(evidence_val - 1, 0))
                    setattr(self, f'curr_num_in_queue_{queue_id}_system',
                            evidence_val)
                    setattr(
                        self, f't_arrival_queue_{queue_id}',
                        self.clock + self.gen_int_arr(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_arrival_time')))
        self.log_query("Intervention")

    def queue_modintervention(self, evidence_var, evidence_val, evidence_type):
        """Function to add or subtract from the queue lengths."""
        # Determine the queue_id from evidence_var
        queue_id = evidence_var[-1]
        logger.debug('Values of the system before intervention')
        logger.debug(f'Queue {queue_id} system: {getattr(self, f"curr_num_in_queue_{queue_id}_system")}')
        if evidence_var == f'Lq{queue_id}':
            if evidence_type == 'additive':
                if getattr(self, f'curr_num_in_queue_{queue_id}_system') == 0:
                    # Add to empty system
                    setattr(self, f'curr_num_in_queue_{queue_id}',
                            (getattr(self, f'curr_num_in_queue_{queue_id}') +
                             evidence_val - 1))
                    setattr(
                        self, f'curr_num_in_queue_{queue_id}_system',
                        (getattr(self, f'curr_num_in_queue_{queue_id}_system') +
                         evidence_val))
                    setattr(self.server_states,
                            f'state_server_queue_{queue_id}', True)
                    setattr(
                        self, f't_departure_queue_{queue_id}',
                        self.clock + self.gen_service_time(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_service_time')))
                    setattr(
                        self, f't_arrival_queue_{queue_id}',
                        self.clock + self.gen_int_arr(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_arrival_time')))
                else:  # If the system has jobs already
                    setattr(self, f'curr_num_in_queue_{queue_id}',
                            (getattr(self, f'curr_num_in_queue_{queue_id}') +
                             evidence_val - 1))
                    setattr(
                        self, f'curr_num_in_queue_{queue_id}_system',
                        (getattr(self, f'curr_num_in_queue_{queue_id}_system') +
                         evidence_val))
                    setattr(
                        self, f't_arrival_queue_{queue_id}',
                        self.clock + self.gen_int_arr(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_arrival_time')))
            elif evidence_type == 'subtractive':
                if getattr(self, f'curr_num_in_queue_{queue_id}_system') == 0:
                    pass
                elif getattr(
                        self,
                        f'curr_num_in_queue_{queue_id}_system') <= evidence_val:
                    # Remove all jobs
                    setattr(self, f'curr_num_in_queue_{queue_id}', 0)
                    setattr(self, f'curr_num_in_queue_{queue_id}_system', 0)
                    setattr(self.server_states,
                            f'state_server_queue_{queue_id}', False)
                    setattr(self, f't_departure_queue_{queue_id}', float('inf'))
                    setattr(
                        self, f't_arrival_queue_{queue_id}',
                        self.clock + self.gen_int_arr(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_arrival_time')))
                elif (getattr(self, f'curr_num_in_queue_{queue_id}_system') - evidence_val) == 1:
                    # Only keep one job in the system which is being served
                    setattr(self, f'curr_num_in_queue_{queue_id}', 0)
                    setattr(self, f'curr_num_in_queue_{queue_id}_system', 1)
                    setattr(self.server_states,
                            f'state_server_queue_{queue_id}', True)
                    setattr(
                        self, f't_arrival_queue_{queue_id}',
                        self.clock + self.gen_int_arr(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_arrival_time')))
                else:  # Remove the specified number of jobs from the system
                    setattr(self, f'curr_num_in_queue_{queue_id}',
                            (getattr(self, f'curr_num_in_queue_{queue_id}') -
                             evidence_val - 1))
                    setattr(
                        self, f'curr_num_in_queue_{queue_id}_system',
                        (getattr(self, f'curr_num_in_queue_{queue_id}_system') -
                         evidence_val))
                    setattr(
                        self, f't_arrival_queue_{queue_id}',
                        self.clock + self.gen_int_arr(
                            getattr(self.input_params,
                                    f'queue_{queue_id}_mean_arrival_time')))
        logger.debug('Values of the system after intervention')
        logger.debug(f'Queue {queue_id} system: {getattr(self, f"curr_num_in_queue_{queue_id}_system")}')
        self.log_query("Mod Intervention")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Generate interventional data for the Markovian queue network.")
    # Example call - python src/simulator_interventions.py
    # --config_file configs/queries.json --experiment_number 1 -v

    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help=
        "Path to the configuration file (e.g. configs/queries.json)",
        default="configs/queries.json")
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

    lambda_qa = query_details['start_parameters']['Lambdaqa']
    mu_qa = query_details['start_parameters']['Muqa']
    lambda_qb = query_details['start_parameters']['Lambdaqb']
    mu_qb = query_details['start_parameters']['Muqb']
    lambda_qc = query_details['start_parameters']['Lambdaqc']
    mu_qc = query_details['start_parameters']['Muqc']
    simulation_end_time = query_details['query_time']
    gt_replications = query_details['gt_replications']
    rp_a_to_b = query_details['start_parameters']['Rab']
    rp_b_to_c = query_details['start_parameters']['Rbc']
    rp_c_to_a = query_details['start_parameters']['Rca']

    df_list = [
    ]  # Global dataframe that is used to log the events of the simulator
    start_time = time.time()
    RUN = 1

    for i in tqdm(range(gt_replications)):
        q = QueueSimulation(simulation_end=simulation_end_time,
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
        "Run", "Lambda_qA", "Mu_qA", "Lambda_qB", "Mu_qB",
        "Lambda_qC", "Mu_qC",
        "Rp_ab", "Rp_bc", "Rp_ca", 
        "Time", "Event",
        "QueueAL", "QueueBL", "QueueCL"
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
