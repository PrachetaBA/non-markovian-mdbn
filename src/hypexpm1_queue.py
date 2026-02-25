"""
Simulation model of a G/M/1 queue
It approximates Gamma (general) distribution to an hypoexponential one

Input parameters:
Gamma shape
Gamma dist variable mean (tau)
Teller service time mean

Events:
1. Arrival (follows hypoexp dist)
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
from math import ceil, sqrt
import matplotlib.pyplot as plt
from scipy.stats import gamma

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InputParameters:
    """
    Input parameters for the HypoExp/M/1 simulation
    j   = gamma shape parameter
    tau = gamma mean interarrival time
    """
    tau: float
    mean_service_time: float
    j: float
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


class GammaM1Simulation:
    """
    Class for HypoExp (Gamma -> Hypoexponential) / M / 1 queue simulation (single server)
    """
    def __init__(self,
                tau: float,
                mean_service_time: float,
                j: float,
                simulation_end: float,
                seed=0,
                initial_queue_length=0):
        self.clock = 0.0  # simulation clock
        self.rng = np.random.default_rng(seed)  # random number generator stream

        self.input_params = InputParameters(tau=tau,
                                            mean_service_time=mean_service_time,
                                            j=j,
                                            simulation_end=simulation_end)

        # get gamma parameters and check validity
        j = float(self.input_params.j)
        tau = float(self.input_params.tau)
        if j <= 0:
            raise ValueError("shape parameter j must be positive")

        # get total phases and store the list of phase rates
        # if j is very small handle case separately
        if abs(j - 1.0) < 1e-12:
            self.num_phases = 1
            self.phase_rates = [1.0 / tau]

        else:
            n = max(ceil(j), 2)
            self.num_phases = int(n)

            # rate for first n-2 phases
            lambda_base = n / tau

            # calculate final two rates nu_f and mu_f using paper formulas
            inside = (n / (2.0 * j)) * (n - j)
            if inside < 0: # make sure it is not negative
                inside = 0.0
            s = sqrt(inside)
            inv_nu_f = (tau / n) * (1.0 + s)
            inv_mu_f = (tau / n) * (1.0 - s)

            # make sure both are +ve
            if inv_mu_f <= 0 or inv_nu_f <= 0:
                raise ValueError(f"Invalid hypoexp parameters for j={j}, tau={tau}, got non-positive phase inverse(s)")

            nu_f = (1.0 / inv_nu_f)
            mu_f = (1.0 / inv_mu_f)

            # phase rates list: first n-2 are lambda_base, last two are nu_f and mu_f
            if n > 2:
                self.phase_rates = [lambda_base] * (n - 2) + [nu_f, mu_f]
            else:
                self.phase_rates = [nu_f, mu_f]

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

        # some logger info to keep
        logger.debug(f"Initialized HypoExp with shape j={j}, n={self.num_phases}, tau={tau}")
        logger.debug(f"Phase rates: {self.phase_rates}")


    def plot_distributions(self, num_samples=100000, bins=100):
        """
        Plot the Gamma distribution and the Hypoexponential approximation on the same graph
        """
        # gamma dist pdf
        shape = self.input_params.j
        scale = self.input_params.tau / shape  # mean = tau, so scale = tau / shape
        x = np.linspace(0, self.input_params.tau * 5, 1000)
        gamma_pdf = gamma.pdf(x, a=shape, scale=scale)

        # hypoexp samples
        hypo_samples = []
        for _ in range(num_samples):
            sample = sum(self.rng.exponential(1.0 / rate) for rate in self.phase_rates)
            hypo_samples.append(sample)
        hypo_samples = np.array(hypo_samples)

        # histogram to approximate PDF
        hist_vals, bin_edges = np.histogram(hypo_samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # plot the distributions together
        plt.figure(figsize=(8, 5))
        plt.plot(x, gamma_pdf, label="Gamma PDF", color="blue", lw=2)
        plt.plot(bin_centers, hist_vals, label="Hypoexp Approx (samples)", color="orange", lw=2, alpha=0.7)
        plt.title(f"Gamma vs Hypoexponential Approximation (j={shape}, tau={self.input_params.tau})")
        plt.xlabel("Time")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)
        plt.show()


    def compute_cross_entropy(self, num_gamma_samples=100000, bins=100, epsilon=1e-12):
        """
        Compute approximate cross-entropy H(Gamma || Hypoexp)
        """
        # get samples from gamma
        shape = self.input_params.j
        scale = self.input_params.tau / shape
        #gamma_samples = np.random.default_rng().gamma(shape, scale, num_gamma_samples)
        gamma_samples = self.rng.gamma(shape, scale, num_gamma_samples)

        # hypoexp samples and estimate pdf
        hypo_samples = np.array([sum(self.rng.exponential(1.0 / rate) for rate in self.phase_rates)
                                for _ in range(num_gamma_samples)])
        hist_vals, bin_edges = np.histogram(hypo_samples, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # hypoexp PDF at Gamma samples and avoid log(0)
        hypo_pdf_at_gamma = np.interp(gamma_samples, bin_centers, hist_vals, left=epsilon, right=epsilon)
        hypo_pdf_at_gamma = np.maximum(hypo_pdf_at_gamma, epsilon)

        # get cross-entropy
        cross_entropy = -np.mean(np.log(hypo_pdf_at_gamma))
        return cross_entropy


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
        next_phase_time = self.gen_phase_time()
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


    def gen_phase_time(self) -> float:
        """
        Generate a single exponential phase time for hypoexponential process
        Uses the current phase index to pick the correct rate from the list
        """
        # phase_counter is 1-based, list index is 0-based
        phase_index = max(0, self.phase_counter - 1)

        # if phase_counter out of range
        if phase_index >= len(self.phase_rates):
            phase_index = len(self.phase_rates) - 1
        phase_lambda = self.phase_rates[phase_index]

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


    def generate_Q_matrix(self, max_queue_length: int):
        """
        builds generator matrix Q for HypoExp/M/1 queue
        state = (phase, queue_length)
        """
        num_phases = self.num_phases
        service_rate = 1.0 / self.input_params.mean_service_time
        num_states = num_phases * (max_queue_length + 1)
        Q = np.zeros((num_states, num_states))

        for queue_len in range(max_queue_length + 1):
            for phase in range(1, num_phases + 1):

                state_idx = queue_len * num_phases + (phase - 1)

                # arrival transition: phase advance or arrival after final phase
                if phase < num_phases:
                    # transition to next phase in same queue length
                    next_state_idx = queue_len * num_phases + phase
                    Q[state_idx, next_state_idx] = self.phase_rates[phase - 1]
                else:
                    # customer arrives (after last phase)
                    if queue_len < max_queue_length:
                        next_state_idx = (queue_len + 1) * num_phases
                        Q[state_idx, next_state_idx] = self.phase_rates[num_phases - 1]

                # service completed
                if queue_len > 0:
                    next_state_idx = (queue_len - 1) * num_phases + (phase - 1)
                    Q[state_idx, next_state_idx] = service_rate

                # set diagonal entries so rows add up to 0
                Q[state_idx, state_idx] = -np.sum(Q[state_idx, :])

        return Q


    def get_optimal_subsampling_interval(self, max_queue_length: int):
        """
        return optimal subsampling interval delta
        reference: https://www.jstor.org/stable/24340803?seq=4
        """
        Q = self.generate_Q_matrix(max_queue_length)
        diagonal_entries = np.diagonal(Q)
        max_diag_abs = np.max(np.abs(diagonal_entries))
        delta = 1.0 / max_diag_abs

        return delta
    


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
    taus = config["taus"]
    service_rates = config["service_rates"]
    shapes = config["shapes"]
    max_iql = int(config.get("max_iql", 0))
    runs = config["runs"]
    sim_end = config["simulation_end"]

    # ensure outputs go to project_root/data/<output_folder>
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "data" / config["output_folder"]
    output_root.mkdir(parents=True, exist_ok=True)
    outpath = output_root / f"hypexp_m1-simulation-results-{args.experiment_number}.csv"

    base_seed = config.get("seed", 0)  # optional

    start_time = time.time()
    df_list = []
    RUN = 1

    for tau in taus:
        for mu in service_rates:
            for j in shapes:
                
                mean_service = 1.0 / mu
                max_queue_len_obs = 0

                for iql in range(0, max_iql + 1):
                    logger.info(f"Running Hypexp/M/1 for tau={tau}, mu={mu}, j={j}, iql={iql}, meanS={mean_service}")

                    for i in range(runs):
                        seed = base_seed + i
                        sim = GammaM1Simulation(tau=tau,
                                            mean_service_time=mean_service,
                                            j=j,
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
                        df_list.append(sim.time_series.assign(Run=RUN, Tau=tau, Mu=mu, J=j, IQL=iql, End=sim_end))
                        RUN += 1

                sim.plot_distributions()
                #ce_loss = sim.compute_cross_entropy()
                #print(f"\nCE loss = {ce_loss}\n")
                Q = sim.generate_Q_matrix(max_queue_len_obs)
                print(Q)
                delta = sim.get_optimal_subsampling_interval(max_queue_len_obs)
                print(f"sub interval = {delta}")

                logger.info(f"Completed {runs} runs for tau={tau}")

    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True)
        df = df[["Run", "Tau", "Mu", "Current_Phase", "J", "IQL", "End", "Time", "Event", "Queue_Length"]]
        df.to_csv(outpath, index=False)
        logger.info(f"Wrote time series to {outpath}")
    else:
        logger.warning("No data generated.")

    end_time = time.time()
    logger.info(f"Total runtime: {end_time - start_time:.3f} seconds")

