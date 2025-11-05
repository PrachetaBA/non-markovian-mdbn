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
        self.input_params = InputParameters(mean_interarrival_time=mean_interarrival_time,
                                     mean_service_time=mean_service_time,
                                     k=k,
                                     simulation_end=simulation_end)

        # next event times (initialize first arrival)
        first_arrival = self.gen_int_arr(self.input_params.mean_interarrival_time)
        # if initial queue >0, we consider first departure scheduled (busy server)
        if initial_queue_length == 0:
            self.next_event_times = NextEventTimes(
                t_arrival=self.clock + first_arrival,
                t_departure=float('inf')
            )
            self.server = ServerState(busy=False)
            self.curr_num_in_q = 0
            self.curr_num_in_system = 0
        else:
            # place initial customers: one in service and rest in queue
            self.curr_num_in_system = initial_queue_length
            self.curr_num_in_q = initial_queue_length - 1
            service_time = self.gen_service_time(self.input_params.mean_service_time)
            self.next_event_times = NextEventTimes(
                t_arrival=self.clock + first_arrival,
                t_departure=self.clock + service_time
            )
            self.server = ServerState(busy=True)

        # bookkeeping
        self.num_arrivals = 0
        self.num_of_departures = 0
        self.dep_sum = 0.0               # total service time (for utilization)
        self.total_wait_time = 0.0       # integral of queue length over time
        self.total_joined_queue = 0      # cumulative count of those who waited

        # time-series logging
        self.time_series = pd.DataFrame(columns=["Time", "Event", "Queue_Length"])
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
            self.arrival()
        else:
            self.departure()

        return False


    def arrival(self):
        """
        Handle an arrival event
        """
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

        # next arrival (Erlang)
        inter = self.gen_int_arr(self.input_params.mean_interarrival_time)
        self.next_event_times.t_arrival = self.clock + inter
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


    def gen_int_arr(self, mean_arrival_time: float) -> float:
        """
        Generate interarrival time using Erlang(k, lambda) distribution
        """
        # using gamma with integer shape k
        k = max(1, int(self.input_params.k))
        return float(self.rng.gamma(shape=k, scale=(mean_arrival_time / k)))


    def gen_service_time(self, mean_service_time: float) -> float:
        """
        Generate service time using exponential distribution
        """
        return float(self.rng.exponential(scale=mean_service_time))


    def log_event(self, event_type: str):
        """
        Log the events of the simulation
        """
        row = pd.Series([self.clock, event_type, self.curr_num_in_system],
                        index=self.time_series.columns)
        # use loc to append (similar to simulator.py)
        self.time_series.loc[len(self.time_series)] = row


    def summary(self):
        """
        Get summary of the simulation run
        """
        T = min(self.clock, self.input_params.simulation_end)
        util = (self.dep_sum / T) if T > 0 else 0.0
        avg_queue = (self.total_wait_time / T) if T > 0 else 0.0
        avg_wait_per_served = (self.total_wait_time / self.num_of_departures) if self.num_of_departures > 0 else 0.0
        avg_service_observed = (self.dep_sum / self.num_of_departures) if self.num_of_departures > 0 else 0.0
        return {
            "SimTime": T,
            "Arrivals": self.num_arrivals,
            "Departures": self.num_of_departures,
            "Utilization": util,
            "AvgQueueLength": avg_queue,
            "AvgWaitPerServed": avg_wait_per_served,
            "AvgServiceObserved": avg_service_observed,
        }



def parse_args():
    """
    Parse the input arguments
    """
    parser = argparse.ArgumentParser(description="Er/M/1 queue simulator")

    # add necessary arguments
    parser.add_argument("--arr", nargs='+', type=float, required=True,
                        help="Arrival rates lambda (e.g. 0.2 0.4). Script uses mean interarrival = 1/lambda")
    parser.add_argument("--ser", nargs=1, type=float, required=True,
                        help="Service rate mu (e.g. 1.0). Script uses mean service time = 1/mu")
    parser.add_argument("--k", nargs=1, type=int, default=[1],
                        help="Erlang shape k (integer >=1). k=1 -> exponential.")
    parser.add_argument("--runs", nargs=1, type=int, default=[1],
                        help="Number of replications per arrival rate")
    parser.add_argument("--end", nargs=1, type=float, required=True,
                        help="Simulation end time (e.g. 10.0)")
    parser.add_argument("--seed", nargs=1, type=int, default=[123],
                        help="Base RNG seed (each replication will add offset)")
    parser.add_argument("--out", nargs=1, type=str, default=["data/erm1-timeseries.csv"],
                        help="Output CSV filepath")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    arrival_rates = args.arr
    mu = args.ser[0]
    k = args.k[0]
    runs = args.runs[0]
    sim_end = args.end[0]
    base_seed = args.seed[0]
    outpath = args.out[0]

    logging.basicConfig(level=logging.INFO)
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

            # run until simulation end
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
        # reorder columns to match mm1-like output
        df = df[["Run", "Lambda", "Mu", "K", "End", "Time", "Event", "Queue_Length"]]
        df.to_csv(outpath, index=False)
        logger.info(f"Wrote time series to {outpath}")
    else:
        logger.warning("No data generated.")

    end_time = time.time()
    logger.info(f"Total runtime: {end_time - start_time:.3f} seconds")
