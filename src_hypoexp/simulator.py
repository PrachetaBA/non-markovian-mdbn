"""
Simulation model of a HypoExp/M/1 queue
We can approximate multiple general distributions to the hypoexponential

Input parameters:
Arrival distributions:
    Hypoexponential distribution: 
        num_phases 
        lambda_first
        lambda_second: None
        lambda_third: None
Departure distribution: 
    Exponential distribution:
        mean: mu
    
Events:
1. Arrival (follow Hypoexponential distribution)
2. Departure (follows Exponential distribution)
"""

import argparse
import time
import logging
import dataclasses
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from math import ceil, sqrt
import matplotlib.pyplot as plt
from scipy.stats import gamma
from helper_functions import get_optimal_subsampling_interval

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InputParameters:
    """
    Input parameters for the HypoExp/M/1 simulation
    Hypoexponential distribution: 
        num_phases 
        lambda_first
        lambda_second: None
        lambda_third: None
    Exponential distribution:
        mean: mu
    """
    phase_rates: list 
    mean_service_rate: float # mean service rate (mu) for exponential distribution
    simulation_end: float


@dataclasses.dataclass
class NextEventTimes:
    """
    Dataclass to store the next event times
    """
    t_arrival: float
    t_departure: float


@dataclasses.dataclass
class ServerState:
    """
    Dataclass to store the server state
    """
    busy: bool


class HypoExpM1Simulation:
    """
    Class for HypoExp/M/1 queue simulation (single server)
    """
    def __init__(self,
                phase_rates: list[float],
                mean_service_rate: float,
                simulation_end: float,
                seed=0,
                initial_queue_length=0):
        self.clock = 0.0  # simulation clock
        self.rng = np.random.default_rng(seed)  # random number generator stream

        self.input_params = InputParameters(phase_rates=phase_rates,
                                            mean_service_rate=mean_service_rate,
                                            simulation_end=simulation_end)
    
        self.num_phases = len(self.input_params.phase_rates)  # number of phases in hypoexponential distribution
        # track current phase for arrivals
        self.phase_counter = 1

        # next event times (initialize first phase arrival)
        first_phase_arrival = self.gen_phase_time()
        if initial_queue_length == 0:
            self.next_event_times = NextEventTimes(
                t_arrival=self.clock + first_phase_arrival,
                t_departure=float('inf')
            )
            self.server = ServerState(busy=False)
            self.curr_num_in_queue = 0  
            self.curr_num_in_system = 0
        else:
            self.curr_num_in_system = initial_queue_length
            self.curr_num_in_queue = initial_queue_length - 1
            service_time = self.gen_service_time(self.input_params.mean_service_rate)
            self.next_event_times = NextEventTimes(
                t_arrival=self.clock + first_phase_arrival,
                t_departure=self.clock + service_time
            )
            self.server = ServerState(busy=True)

        # bookkeeping
        self.num_arrivals = 0
        self.num_of_departures = 0

        # time-series logging
        self.time_series = pd.DataFrame(columns=["Time", "Event", "Queue_Length", "Current_Phase"])
        self.log_event("Initialization")

        # some logger info to keep
        logger.debug(f"Initialized HypoExp with phase rates: {self.input_params.phase_rates}")


    def time_adv(self):
        """
        Go to next event and call arrival or departure
        """
        # next event
        t_next = min(self.next_event_times.t_arrival, self.next_event_times.t_departure)
        self.clock = t_next

        # stop if time exceeds simulation end
        if t_next >= self.input_params.simulation_end:
            self.log_event("Simulation_End")
            return True

        # call appropriate event
        if self.next_event_times.t_arrival <= self.next_event_times.t_departure:
            self.arrival_phase()
        else:
            self.departure()

        return False


    def arrival_phase(self):
        """
        handle the hypoexponential arrival phase
        trigger an arrival when n phases complete
        """
        if self.phase_counter < self.num_phases:
            # move to next phase
            self.phase_counter += 1
            next_phase_time = self.gen_phase_time()
            self.next_event_times.t_arrival = self.clock + next_phase_time
            self.log_event("Phase_Progress")
            return

        # n phases completed so actual arrival
        self.phase_counter = 1
        self.num_arrivals += 1
        self.curr_num_in_system += 1

        # If server idle and no queue
        if self.curr_num_in_queue == 0 and not self.server.busy:
            self.server.busy = True
            service = self.gen_service_time(self.input_params.mean_service_rate)
            self.next_event_times.t_departure = self.clock + service
        else:
            # or join the queue
            self.curr_num_in_queue += 1

        # schedule next phase
        next_phase_time = self.gen_phase_time()
        self.next_event_times.t_arrival = self.clock + next_phase_time
        self.log_event("Arrival")


    def departure(self):
        """
        Handle departure
        """
        self.num_of_departures += 1
        self.curr_num_in_system -= 1

        if self.curr_num_in_queue > 0:
            # next from queue to service
            service = self.gen_service_time(self.input_params.mean_service_rate)
            self.next_event_times.t_departure = self.clock + service
            self.curr_num_in_queue -= 1
        else:
            # if no one waiting
            self.next_event_times.t_departure = float('inf')
            self.server.busy = False

        self.log_event("Departure")


    def gen_phase_time(self) -> float:
        """
        Generate a single exponential phase time for hypoexponential process
        Uses the current phase index to pick the correct rate from the list
        """
        # phase_counter is 1-based, list index is 0-based
        phase_index = max(0, self.phase_counter - 1)

        # if phase_counter out of range
        if phase_index >= len(self.input_params.phase_rates):
            phase_index = len(self.input_params.phase_rates) - 1
        phase_lambda = self.input_params.phase_rates[phase_index]

        return float(self.rng.exponential(scale=1.0 / phase_lambda))


    def gen_service_time(self, mean_service_rate: float) -> float:
        """
        Generate service time using exponential distribution
        """
        return float(self.rng.exponential(scale=1.0 / mean_service_rate))


    def log_event(self, event_type: str):
        """
        Log the events of the simulation
        """
        row = pd.Series([self.clock, event_type, self.curr_num_in_system, self.phase_counter],
                        index=self.time_series.columns)
        # use loc to append (similar to simulator.py)
        self.time_series.loc[len(self.time_series)] = row    


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", "-c", type=str, required=True,
                        help="Path to YAML config file (e.g. configs/simulator.yaml)")
    parser.add_argument("--experiment_number", "-e", type=int, default=1,
                        help="Experiment number to run the script")
    parser.add_argument("--verbose", "-v",
                        help="Increase output verbosity",
                        action="store_true",
                        default=False,
                        required=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # load the config
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)[f"experiment_{args.experiment_number}"]

    # Extract simulation input parameters
    arrival_distributions = config["arrival_distributions"]
    #phase_rates = config["phase_rates"] # Gives us a list of lists
    service_rates = config["service_rates"]
    max_iql = int(config.get("max_iql", 0))
    runs = config["runs"]
    sim_end = config["simulation_end"]

    # ensure outputs go to project_root/data/<output_folder>
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / config.get("output_folder", "data/simulation")
    output_root.mkdir(parents=True, exist_ok=True)
    outpath = output_root / f'hypoexp-m1-{args.experiment_number}.csv'

    base_seed = config.get("seed", 0)  # optional
    start_time = time.time()
    df_list = []
    RUN = 1

    max_queue_len_obs = 0
    min_delta_observed = float("inf")
    # decide iql values to use based on varying_iql bool
    iql_values = range(0, max_iql + 1) if config['varying_iql'] else [0]
    
    for dist in arrival_distributions:
        phase_rate_set = dist["phase_rates"]
        alpha = dist["alpha"]
        theta = dist["theta"]
        for mu in service_rates:

            for iql in iql_values:
                logger.info(f"Running Hypexp/M/1 for phase_rates={phase_rate_set}, mu={mu}, iql={iql}") 
                for i in range(runs):
                    seed = base_seed + i
                    sim = HypoExpM1Simulation(phase_rates=phase_rate_set,
                                              mean_service_rate=mu,
                                              simulation_end=sim_end,
                                              seed=seed,
                                              initial_queue_length=iql)

                    while True:
                        stop = sim.time_adv()
                        if stop:
                            break

                    run_max = int(sim.time_series["Queue_Length"].max())
                    max_queue_len_obs = max(max_queue_len_obs, run_max)

                    # attach run metadata and collect timeseries
                    df_list.append(sim.time_series.assign(Run=RUN, Alpha=alpha,Theta=theta, Phase_Rates=str(phase_rate_set), Mu=mu, IQL=iql, End=sim_end))
                    RUN += 1
            
            # get optimal subsampling interval for this config
            delta = get_optimal_subsampling_interval(max_queue_len_obs, phase_rate_set, mu)
            logger.info(f"Optimal subsampling interval for alpha={alpha}, theta={theta}, mu={mu} = {delta}")
            min_delta_observed = min(min_delta_observed, delta)

    logger.info(f"Minimum optimal subsampling interval observed across all configs: {min_delta_observed}")
    print(f"Minimum optimal subsampling interval observed across all configs: {min_delta_observed}")

    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True)
        df = df[["Run", "Alpha", "Theta", "Phase_Rates", "Mu", "Current_Phase", "IQL", "End", "Time", "Event", "Queue_Length"]]
        df.to_csv(outpath, index=False)
        logger.info(f"Wrote time series to {outpath}")
    else:
        logger.warning("No data generated.")

    end_time = time.time()
    logger.info(f"Total runtime: {end_time - start_time:.3f} seconds")

