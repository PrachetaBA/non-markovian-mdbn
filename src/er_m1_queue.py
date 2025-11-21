"""
Simulation model of a Er/M/1 queue

Input parameters: 
Interarrival time mean 
Teller service time mean
Number of phases (k) for Erlang distribution 

Events: 
1. Arrival 
2. Departure 
"""

import argparse
import time
import logging
import dataclasses
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InputParameters:
    """
    Input parameters for the Er/M/1 simulation
    """
    mean_interarrival_time: float
    mean_service_time: float
    k: int
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



class ErM1Simulation:
    """
    Class for Er/M/1 queue simulation (single server)
    """
    def __init__(self,
                 mean_interarrival_time: float,
                 mean_service_time: float,
                 k: int,
                 simulation_end: float,
                 seed=0,
                 initial_queue_length=0):
        self.clock = 0.0  # simulation clock
        self.rng = np.random.default_rng(seed)  # random number generator stream

        # ensure k (total phases) is at least 1
        if k < 1:
            logging.warning(f"provided k={k} is invalid, resetting it to 1")
            k = 1

        self.input_params = InputParameters(mean_interarrival_time=mean_interarrival_time,
                                     mean_service_time=mean_service_time,
                                     k=k,
                                     simulation_end=simulation_end)

        # track current phase for Erlang arrivals
        self.phase_counter = 1

        # next event times (initialize first phase arrival)
        first_phase_arrival = self.gen_phase_time(self.input_params.mean_interarrival_time)
        if initial_queue_length == 0:
            self.next_event_times = NextEventTimes(
                t_arrival=self.clock + first_phase_arrival,
                t_departure=float('inf')
            )
            self.server = ServerState(busy=False)
            self.curr_num_in_q = 0
            self.curr_num_in_system = 0
        else:
            self.curr_num_in_system = initial_queue_length
            self.curr_num_in_q = initial_queue_length - 1
            service_time = self.gen_service_time(self.input_params.mean_service_time)
            self.next_event_times = NextEventTimes(
                t_arrival=self.clock + first_phase_arrival,
                t_departure=self.clock + service_time
            )
            self.server = ServerState(busy=True)

        # bookkeeping
        self.num_arrivals = 0
        self.num_of_departures = 0
        self.dep_sum = 0.0               
        self.total_wait_time = 0.0       
        self.total_joined_queue = 0      

        # time-series logging
        self.time_series = pd.DataFrame(columns=["Time", "Event", "Queue_Length", "Current_Phase"])
        self.log_event("Initialization")


    def time_adv(self):
        """
        Go to next event and call arrival or departure
        """
        # next event
        t_next = min(self.next_event_times.t_arrival, self.next_event_times.t_departure)
        self.total_wait_time += self.curr_num_in_q * (t_next - self.clock)
        self.clock = t_next

        # stop if time exceeds simulation end
        if t_next >= self.input_params.simulation_end:
            return True

        # call appropriate event
        if self.next_event_times.t_arrival <= self.next_event_times.t_departure:
            self.arrival_phase()
        else:
            self.departure()

        return False


    def arrival_phase(self):
        """
        handle the Erlang arrival phase
        trigger an arrival when k phases complete
        """
        if self.phase_counter < self.input_params.k:
            # move to next phase
            self.phase_counter += 1
            next_phase_time = self.gen_phase_time(self.input_params.mean_interarrival_time)
            self.next_event_times.t_arrival = self.clock + next_phase_time
            self.log_event("Phase_Progress")
            return

        # k phases completed so actual arrival
        self.phase_counter = 1
        self.num_arrivals += 1
        self.curr_num_in_system += 1

        # If server idle and no queue
        if self.curr_num_in_q == 0 and not self.server.busy:
            self.server.busy = True
            service = self.gen_service_time(self.input_params.mean_service_time)
            self.dep_sum += service
            self.next_event_times.t_departure = self.clock + service
        else:
            # or join the queue
            self.curr_num_in_q += 1
            self.total_joined_queue += 1

        # schedule next phase
        next_phase_time = self.gen_phase_time(self.input_params.mean_interarrival_time)
        self.next_event_times.t_arrival = self.clock + next_phase_time
        self.log_event("Arrival")


    def departure(self):
        """
        Handle departure
        """
        self.num_of_departures += 1
        self.curr_num_in_system -= 1

        if self.curr_num_in_q > 0:
            # next from queue to service
            service = self.gen_service_time(self.input_params.mean_service_time)
            self.dep_sum += service
            self.next_event_times.t_departure = self.clock + service
            self.curr_num_in_q -= 1
        else:
            # if no one waiting
            self.next_event_times.t_departure = float('inf')
            self.server.busy = False

        self.log_event("Departure")


    def gen_phase_time(self, mean_arrival_time: float) -> float:
        """
        generates a single exponential phase time for Erlang process
        """
        phase_lambda = self.input_params.k / mean_arrival_time
        return float(self.rng.exponential(scale=1.0 / phase_lambda))


    def gen_service_time(self, mean_service_time: float) -> float:
        """
        Generate service time using exponential distribution
        """
        return float(self.rng.exponential(scale=mean_service_time))


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

    # extract parameters
    arrival_rates = config["arrival_rates"]
    mu = config["service_rates"][0]
    k = config["total_phases"]
    runs = config["runs"]
    sim_end = config["simulation_end"]

    # ensure outputs go to project_root/data/<output_folder>
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "data" / config["output_folder"]
    output_root.mkdir(parents=True, exist_ok=True)
    outpath = output_root / f"erm1-simulation-results-{args.experiment_number}.csv"

    base_seed = config.get("seed", 0)  # optional

    start_time = time.time()
    df_list = []
    RUN = 1

    for lam in arrival_rates:
        mean_interarrival = 1.0 / lam
        mean_service = 1.0 / mu
        logger.info(f"Running Er/M/1 for lambda={lam}, mu={mu}, k={k}, meanIA={mean_interarrival}, meanS={mean_service}")

        for i in range(runs):
            seed = base_seed + i
            sim = ErM1Simulation(mean_interarrival_time=mean_interarrival,
                                 mean_service_time=mean_service,
                                 k=k,
                                 simulation_end=sim_end,
                                 seed=seed,
                                 initial_queue_length=0)

            while True:
                stop = sim.time_adv()
                if stop:
                    break

            # attach run metadata and collect timeseries
            df_list.append(sim.time_series.assign(Run=RUN, Lambda=lam, Mu=mu, K=k, End=sim_end))
            RUN += 1

        logger.info(f"Completed {runs} runs for lambda={lam}")

    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True)
        df = df[["Run", "Lambda", "Mu", "Current_Phase", "K", "End", "Time", "Event", "Queue_Length"]]
        df.to_csv(outpath, index=False)
        logger.info(f"Wrote time series to {outpath}")
    else:
        logger.warning("No data generated.")

    end_time = time.time()
    logger.info(f"Total runtime: {end_time - start_time:.3f} seconds")
