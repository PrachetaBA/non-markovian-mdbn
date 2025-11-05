"""Script to calculate the sampling interval given a set of 
simulation input parameters."""

# Import libraries
import itertools
import numpy as np
import yaml

# Read the configuration file and extract the parameters
config_file = 'configs/simulator.yaml'
experiment_number = 11
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

# Compute the optimal sampling interval based on a provided value of the maximum queue length.abs
max_ql = 5


def q_values(m, m_prime, sim_input_parameters):
    """Returns the q values for the transition from state m to state m_prime.
    Args:
        m: the current state (vector of size 3)
        m_prime: the next state (vector of size 3)
    """
    lambda_qa = sim_input_parameters['lambda_qa']
    mu_qa = sim_input_parameters['mu_qa']
    lambda_qb = sim_input_parameters['lambda_qb']
    mu_qb = sim_input_parameters['mu_qb']
    lambda_qc = sim_input_parameters['lambda_qc']
    mu_qc = sim_input_parameters['mu_qc']
    r_ab = sim_input_parameters['r_ab']
    r_bc = sim_input_parameters['r_bc']
    r_ca = sim_input_parameters['r_ca']

    # Determine how many states have changed between m and m_prime
    vector_subtraction = np.subtract(m_prime, m)
    # If the sum of the vector > 1 or < -1, then there are multiple events
    # that have occurred in the network.
    if sum(vector_subtraction) > 1 or sum(vector_subtraction) < -1:
        return 0.0
    # if each element in the vector has changed more than 1, then there are multiple events
    # that have occurred in the network
    elif any(abs(vector_subtraction) > 1):
        return 0.0
    else:
        # Exogenous arrivals
        if np.array_equal(vector_subtraction, np.array([1, 0, 0])):
            # Event: Customer arrives at node 1
            return lambda_qa
        elif np.array_equal(vector_subtraction, np.array([0, 1, 0])):
            # Event: Customer arrives at node 2
            return lambda_qb
        elif np.array_equal(vector_subtraction, np.array([0, 0, 1])):
            # Event: Customer arrives at node 3
            return lambda_qc
        # Exogenous departures
        elif np.array_equal(vector_subtraction, np.array([-1, 0, 0])):
            # Event: Customer departs from node 1 exogenously
            return mu_qa * (1 - r_ab)
        elif np.array_equal(vector_subtraction, np.array([0, -1, 0])):
            # Event: Customer departs from node 2 exogenously
            return mu_qb * (1 - r_bc)
        elif np.array_equal(vector_subtraction, np.array([0, 0, -1])):
            # Event: Customer departs from node 3 exogenously
            return mu_qc * (1 - r_ca)
        # Endogenous events
        elif np.array_equal(vector_subtraction, np.array([-1, 1, 0])):
            # Event: Customer departs from node 1 and arrives at node 2
            return mu_qa * r_ab
        elif np.array_equal(vector_subtraction, np.array([0, -1, 1])):
            # Event: Customer departs from node 2 and arrives at node 3
            return mu_qb * r_bc
        elif np.array_equal(vector_subtraction, np.array([1, 0, -1])):
            # Event: Customer departs from node 3 and arrives at node 1
            return mu_qc * r_ca
        # All other endogenous events have a rate of 0
        else:
            return 0.0


# Function to compute the optimal sampling interval for a particular set of simulation
# input parameters
def optimal_delta(sim_input_parameters):
    """Compute the optimal sampling interval for a given set of simulation input parameters."""

    possible_states = list(itertools.product(range(max_ql + 1), repeat=3))

    # We will use one array to store the diagonal values of the matrix
    diagonal_values = np.zeros(len(possible_states))

    for i, m in enumerate(possible_states):
        for j, m_prime in enumerate(possible_states):
            if m != m_prime:
                diagonal_values[i] += q_values(m, m_prime, sim_input_parameters)

    # Find the max of absolute values of all the diagonal values
    max_diag = max(abs(diagonal_values))

    # Find the max value of delta, delta < 1/max_diag
    delta = 1 / max_diag
    return delta


# For each combination of simulation input parameters, calculate the optimal sampling interval
sampling_intervals = []

for lambda_qa in mean_interarrival_rates_queue_a:
    for mu_qa in mean_service_rates_queue_a:
        for lambda_qb in mean_interarrival_rates_queue_b:
            for mu_qb in mean_service_rates_queue_b:
                for lambda_qc in mean_interarrival_rates_queue_c:
                    for mu_qc in mean_service_rates_queue_c:
                        for r_ab in rp_a_to_b:
                            for r_bc in rp_b_to_c:
                                for r_ca in rp_c_to_a:
                                    sim_input_parameters = {
                                        'lambda_qa': lambda_qa,
                                        'mu_qa': mu_qa,
                                        'lambda_qb': lambda_qb,
                                        'mu_qb': mu_qb,
                                        'lambda_qc': lambda_qc,
                                        'mu_qc': mu_qc,
                                        'r_ab': r_ab,
                                        'r_bc': r_bc,
                                        'r_ca': r_ca
                                    }
                                    print(
                                        f'Simulation input parameters: {sim_input_parameters}'
                                    )
                                    delta = optimal_delta(sim_input_parameters)
                                    print(
                                        f'The optimal sampling interval is: {delta}'
                                    )
                                    sampling_intervals.append(delta)

# Find the minimum sampling interval
min_sampling_interval = min(sampling_intervals)
print(f'The minimum sampling interval is: {min_sampling_interval}')
