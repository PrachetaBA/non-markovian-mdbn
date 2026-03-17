# pylint: disable=logging-fstring-interpolation, pointless-string-statement, invalid-name
"""Script to generate a query workload by sampling from a possible set of interventions."""

# Import necessary libraries
import logging
import json
import argparse
import os
import numpy as np
import yaml

logger = logging.getLogger('expt_config_generator')
logging.basicConfig(level=logging.INFO)

def single_intervention(config_file,
                        simulation_config_file='configs/simulator.yaml',
                        simulation_config_expt=1,
                        time_discretization_experiment=1,
                        results_suffix='queries',
                        gt_suffix='queries_gt',
                        maximum_ql=14,
                        max_ql_queries=10,
                        inference_algorithm='LazyPropagation',
                        total_queries=50,
                        expt_name=None,
                        optimal_delta=None,
                        rng=np.random.default_rng(seed=188)):
    """Randomly samples from a possible set of parameters to add a new experiment."""
    # Step 1. Read the yaml configuration file and extract the last experiment number
    with open(config_file, 'r', encoding='utf-8') as cf:
        config = json.load(cf)

    if not config:
        last_expt_num = 0
        config = {}
    else:
        curr_expt_nums = [int(x.split('_')[1]) for x in list(config.keys())]
        # Sort and find the highest experiment number
        curr_expt_nums.sort()
        last_expt_num = curr_expt_nums[-1]
        logger.info(f'Last experiment number: {last_expt_num}')

    # Using the simulator, extract the values of possible parameters
    with open(simulation_config_file, 'r', encoding='utf-8') as file:
        all_simulators = yaml.safe_load(file)
    sim_config = all_simulators[f'experiment_{simulation_config_expt}']

    # Extract the possible values of the parameters
    unq_alpha = tuple(sorted(sim_config['alphas']))
    unq_theta = tuple(sorted(sim_config['thetas']))
    unq_mu = tuple(sorted(sim_config['service_rates']))
    max_time = sim_config['simulation_end']
    possible_qls = list(range(0, max_ql_queries + 1))
    types_of_interventions = [
        'conditional', 'interventional', 'additive', 'subtractive',
        'parameter_intervention'
    ]
    queue_variables = ['QueueLength']
    parameter_variables = ['Alpha', 'Theta', 'Mu']
    max_iql = sim_config['max_iql']

    # Step 2. Generate new experiments
    start_exp_num = last_expt_num + 1

    # Step 3: Define the starting parameters for the values
    for query_num in range(start_exp_num, start_exp_num + total_queries):
        start_alpha = rng.choice(unq_alpha)
        start_theta = rng.choice(unq_theta)
        start_mu = rng.choice(unq_mu)
        # Let us keep the starting queue length small for now
        start_ql = rng.choice(list(range(0, max_iql + 1)))

        # Pick a type of intervention (assuming 500 queries, ~100 of each type)
        if query_num <= 100:
            intervention_type = 'conditional'
        elif query_num > 100 and query_num <= 200:
            intervention_type = 'parameter_intervention'
        elif query_num > 200 and query_num <= 300:
            intervention_type = 'interventional'
        else:
            intervention_type = rng.choice(['additive', 'subtractive'],
                                           size=1)[0]

        if intervention_type in [
                'conditional', 'interventional', 'additive', 'subtractive'
        ]:
            # Pick an intervention variable
            intervention_variable = 'QueueLength'
            # Pick a query variable
            query_variable = 'QueueLength'
            # Pick an intervention value
            intervention_value = int(rng.choice(possible_qls))
        else:
            # Pick an intervention parameter
            intervention_variable = rng.choice(parameter_variables)
            # Pick a query variable
            query_variable = 'QueueLength'
            # Pick an intervention value depending on the parameter,
            # that is necessarily different from the starting value
            if intervention_variable == 'Alpha':
                intervention_value = rng.choice([
                    val for val in unq_alpha
                    if val != start_alpha
                ])
            elif intervention_variable == 'Theta':
                intervention_value = rng.choice([
                    val for val in unq_theta
                    if val != start_theta
                ])
            elif intervention_variable == 'Mu':
                intervention_value = rng.choice([
                    val for val in unq_mu
                    if val != start_mu
                ])
        logger.info(f'Intervention type is {intervention_type}, '
                    f'intervention variable is {intervention_variable}, '
                    f'intervention value is {intervention_value}, '
                    f'query variable is {query_variable}')

        # Pick query time (an integer between 1 and max_time)
        # query_time = float(rng.integers(2, max_time + 1))
        # Pick a query time that is a multiple of delta that lies between 1.0 and max_time
        possible_timepoints = [
            round(x, 3) for x in np.arange(0.0, max_time + optimal_delta, optimal_delta)
        ]
        query_time = rng.choice(possible_timepoints[2:])
        # Pick an intervention time that is less than the query time
        # intervention_time = float(rng.integers(1, query_time))
        intervention_time = rng.choice(
            possible_timepoints[1:possible_timepoints.index(query_time)])
        logger.info(
            f'Query time is {query_time}, intervention time is {intervention_time}'
        )

        # Current config_details
        exp_skeleton = {
            'time_discretization_experiment':
                int(time_discretization_experiment),
            'dbn_output_folder':
                'data/dbns',
            'maximum_ql':
                int(maximum_ql),
            'start_parameters': {
                'Alpha': start_alpha,
                'Theta': start_theta,
                'Mu': start_mu,
                'QueueLength': int(start_ql)
            },
            'inference_algorithm':
                inference_algorithm,
            'interventions':
                None,
            'query_variable':
                query_variable,
            'query_time':
                query_time,
            'results_folder':
                f'output/{results_suffix}',
            'gt_replications':
                50000,
            'gt_results_folder':
                f'data/{gt_suffix}',
            'figures_folder':
                f'figures/{results_suffix}',
            'expt_name':
                expt_name
        }
        if inference_algorithm == 'GibbsSampling': 
            exp_skeleton['burn_in'] = 500
            exp_skeleton['epsilon'] = 1e-3
        elif inference_algorithm == 'MonteCarloSampling': 
            exp_skeleton['epsilon'] = 1e-2
        else:
            # Do nothing
            pass

        exp_skeleton['interventions'] = [{
            'intervention_variable': intervention_variable,
            'intervention_type': intervention_type,
            'intervention_start': intervention_time,
            'intervention_value': intervention_value
        }]
        # Add the exp_skeleton to the config dictionary
        config[f'experiment_{query_num}'] = exp_skeleton
        logger.info(f'Experiment_{query_num} added!')

    # Step 3. Write the new configuration to the yaml file
    with open(config_file, 'w', encoding='utf-8') as rf:
        json.dump(config, rf, indent=4)


if __name__ == '__main__':
    # Create a set of 500 queries for the experiment.
    parser = argparse.ArgumentParser(description='Generate query workload.')
    parser.add_argument('--experiment_number', '-e', type=int, default=1, help='Experiment number to determine the type of queries to be generated.')
    args = parser.parse_args()
    EXPT_NUM = args.experiment_number
    
    # Create a random number generator
    SEED = 1603
    rng = np.random.default_rng(seed=SEED)

    expt_dets = {
        1: {
           'time_discretization_experiment': 8,
           'simulation_config_expt': 8,
           'max_ql': 9, 
           'inference_algorithm': 'LazyPropagation',
           'dbn_name': 'dbn_hypoexpm1_exp8_crosstab'
        }, 
        2: {
            'time_discretization_experiment': 8,
            'simulation_config_expt': 8,
            'max_ql': 9,
            'inference_algorithm': 'GibbsSampling',
            'dbn_name': 'dbn_hypoexpm1_exp8_crosstab'
        },
        3: {
            'time_discretization_experiment': 8,
            'simulation_config_expt': 8,
            'max_ql': 9,
            'inference_algorithm': 'MonteCarloSampling',
            'dbn_name': 'dbn_hypoexpm1_exp8_crosstab'
        },
        4: {
            'time_discretization_experiment': 8,
            'simulation_config_expt': 8,
            'max_ql': 9,
            'inference_algorithm': 'LazyPropagation',
            'dbn_name': 'dbn_hypoexpm1_exp8_crosstab'
        },
        5: { # running bif 9
            'time_discretization_experiment': 9,
            'simulation_config_expt': 9,
            'max_ql': 9,
            'inference_algorithm': 'LazyPropagation',
            'dbn_name': 'dbn_hypoexpm1_exp9'
        },
        6: { # running bif 11
            'time_discretization_experiment': 11,
            'simulation_config_expt': 11,
            'max_ql': 16,
            'inference_algorithm': 'LazyPropagation',
            'dbn_name': 'dbn_hypoexpm1_exp11'
        }
    }
    
    # Obtain optimal delta from the corresponding time discretization experiment
    time_discretization_filename = 'config/hypoexp_time_discretization.yaml'
    with open(time_discretization_filename, 'r', encoding='utf-8') as f: 
        time_discretization_config = yaml.safe_load(f)
        time_discretization_params = time_discretization_config[
            f'experiment_{expt_dets[EXPT_NUM]["time_discretization_experiment"]}']
    sampling_interval = time_discretization_params['sampling_interval']

    NUM_QUERIES = 500
    MAX_QL_QUERIES = 7  

    # Test query workload construction
    CONFIG_NAME = (
        f'query_workload_exp-{EXPT_NUM}')
    # If the file does not exist, create a JSON file with an empty dictionary
    if not os.path.exists(f'config/{CONFIG_NAME}.json'):
        with open(f'config/{CONFIG_NAME}.json', 'w', encoding='utf-8') as f:
            json.dump({}, f)
    single_intervention(
        f'config/{CONFIG_NAME}.json',
        simulation_config_file='config/gamma_simulator.yaml',
        simulation_config_expt=expt_dets[EXPT_NUM]['simulation_config_expt'],
        time_discretization_experiment=expt_dets[EXPT_NUM]
        ['time_discretization_experiment'],
        results_suffix='queries',
        gt_suffix='queries_gt',
        maximum_ql=expt_dets[EXPT_NUM]['max_ql'],
        max_ql_queries=MAX_QL_QUERIES,  # Only for query purposes
        total_queries=NUM_QUERIES,
        inference_algorithm=expt_dets[EXPT_NUM].get('inference_algorithm', 'LazyPropagation'),
        expt_name=expt_dets[EXPT_NUM]['dbn_name'],
        optimal_delta=sampling_interval,
        rng=rng)