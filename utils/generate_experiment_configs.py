# pylint: disable=logging-fstring-interpolation, pointless-string-statement, invalid-name
"""Script to generate a query workload by sampling from a possible set of interventions."""

# Import necessary libraries
import logging
import json
import os
import numpy as np
import yaml

logger = logging.getLogger('expt_config_generator')
logging.basicConfig(level=logging.INFO)

# Create a random number generator
SEED = 1603
rng = np.random.default_rng(seed=SEED)


def single_intervention(config_file,
                        simulation_config_file='configs/simulator.yaml',
                        simulation_config_expt=1,
                        time_discretization_experiment=1,
                        edges=None,
                        results_suffix='queries',
                        gt_suffix='queries_gt',
                        maximum_ql=14,
                        max_ql_queries=10,
                        inference_algorithm='LazyPropagation',
                        total_queries=50,
                        expt_name=None):
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
    unq_arr_qa = set(sim_config['queue_a_arrival_rates'])
    unq_ser_qa = set(sim_config['queue_a_service_rates'])
    unq_arr_qb = set(sim_config['queue_b_arrival_rates'])
    unq_ser_qb = set(sim_config['queue_b_service_rates'])
    unq_arr_qc = set(sim_config['queue_c_arrival_rates'])
    unq_ser_qc = set(sim_config['queue_c_service_rates'])
    unq_prob_ab = set(sim_config['routing_probabilities']['a_to_b'])
    unq_prob_bc = set(sim_config['routing_probabilities']['b_to_c'])
    unq_prob_ca = set(sim_config['routing_probabilities']['c_to_a'])
    max_time = sim_config['simulation_end']
    possible_qls = set(list(range(0, max_ql_queries + 1)))
    types_of_interventions = [
        'conditional', 'interventional', 'additive', 'subtractive',
        'parameter_intervention'
    ]
    queue_variables = set(['Lqa', 'Lqb', 'Lqc'])
    parameter_variables = set([
        'Lambdaqa', 'Muqa', 'Lambdaqb', 'Muqb', 'Lambdaqc', 'Muqc', 'Rab',
        'Rbc', 'Rca'
    ])
    max_iql = sim_config['max_iql']

    # Step 2. Generate new experiments
    start_exp_num = last_expt_num + 1

    # Step 3: Define the starting parameters for the values
    for query_num in range(start_exp_num, start_exp_num + total_queries):
        start_lambda_qa = rng.choice(list(unq_arr_qa))
        start_mu_qa = rng.choice(list(unq_ser_qa))
        start_lambda_qb = rng.choice(list(unq_arr_qb))
        start_mu_qb = rng.choice(list(unq_ser_qb))
        start_lambda_qc = rng.choice(list(unq_arr_qc))
        start_mu_qc = rng.choice(list(unq_ser_qc))
        start_prob_ab = rng.choice(list(unq_prob_ab))
        start_prob_bc = rng.choice(list(unq_prob_bc))
        start_prob_ca = rng.choice(list(unq_prob_ca))
        # Let us keep the starting queue length small for now
        start_qla = rng.choice(list(range(0, max_iql + 1)))
        start_qlb = rng.choice(list(range(0, max_iql + 1)))
        start_qlc = rng.choice(list(range(0, max_iql + 1)))

        logger.info(
            f'Starting parameters are Lambdaqa = {start_lambda_qa}, '
            f'Muqa = {start_mu_qa}, Lambdaqb = {start_lambda_qb}, '
            f'Muqb = {start_mu_qb}, Lambdaqc = {start_lambda_qc}, '
            f'Muqc ={start_mu_qc}, Rab = {start_prob_ab}, Rbc = {start_prob_bc}, '
            f'Rca = {start_prob_ca}, Lqa0 = {start_qla}, Lqb0 ={start_qlb}, '
            f'Lqc0 = {start_qlc}')

        # Pick a type of intervention (assuming 200 queries, ~50 of each type)
        if query_num <= 50:
            intervention_type = 'conditional'
        elif query_num > 50 and query_num <= 100:
            intervention_type = 'parameter_intervention'
        elif query_num > 100 and query_num <= 150:
            intervention_type = 'interventional'
        else:
            intervention_type = rng.choice(['additive', 'subtractive'],
                                           size=1)[0]
        # Random choice
        # intervention_type = rng.choice(types_of_interventions)

        if intervention_type in [
                'conditional', 'interventional', 'additive', 'subtractive'
        ]:
            # Pick an intervention variable
            intervention_variable = rng.choice(list(queue_variables))
            # Pick a query variable
            query_variable = rng.choice(list(queue_variables))
            # Pick an intervention value
            intervention_value = int(rng.choice(list(possible_qls)))
        else:
            # Pick an intervention parameter
            intervention_variable = rng.choice(list(parameter_variables))
            # Pick a query variable
            query_variable = rng.choice(list(queue_variables))
            # Pick an intervention value depending on the parameter,
            # that is necessarily different from the starting value
            if intervention_variable == 'Lambdaqa':
                intervention_value = rng.choice(
                    list(unq_arr_qa.difference({start_lambda_qa})))
            elif intervention_variable == 'Muqa':
                intervention_value = rng.choice(
                    list(unq_ser_qa.difference({start_mu_qa})))
            elif intervention_variable == 'Lambdaqb':
                intervention_value = rng.choice(
                    list(unq_arr_qb.difference({start_lambda_qb})))
            elif intervention_variable == 'Muqb':
                intervention_value = rng.choice(
                    list(unq_ser_qb.difference({start_mu_qb})))
            elif intervention_variable == 'Lambdaqc':
                intervention_value = rng.choice(
                    list(unq_arr_qc.difference({start_lambda_qc})))
            elif intervention_variable == 'Muqc':
                intervention_value = rng.choice(
                    list(unq_ser_qc.difference({start_mu_qc})))
            elif intervention_variable == 'Rab':
                intervention_value = rng.choice(
                    list(unq_prob_ab.difference({start_prob_ab})))
            elif intervention_variable == 'Rbc':
                intervention_value = rng.choice(
                    list(unq_prob_bc.difference({start_prob_bc})))
            elif intervention_variable == 'Rca':
                intervention_value = rng.choice(
                    list(unq_prob_ca.difference({start_prob_ca})))
        logger.info(f'Intervention type is {intervention_type}, '
                    f'intervention variable is {intervention_variable}, '
                    f'intervention value is {intervention_value}, '
                    f'query variable is {query_variable}')

        # Pick query time (an integer between 1 and max_time)
        # query_time = float(rng.integers(2, max_time + 1))
        # Pick a query time that is a multiple of 0.2 that lies between 1.0 and max_time
        possible_timepoints = [
            round(x, 1) for x in np.arange(0.0, max_time + 0.2, 0.2)
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
            'dbn_edges':
                edges,
            'dbn_output_folder':
                'data/dbns',
            'maximum_ql':
                int(maximum_ql),
            'start_parameters': {
                'Lambdaqa': start_lambda_qa,
                'Muqa': start_mu_qa,
                'Lambdaqb': start_lambda_qb,
                'Muqb': start_mu_qb,
                'Lambdaqc': start_lambda_qc,
                'Muqc': start_mu_qc,
                'Rab': start_prob_ab,
                'Rbc': start_prob_bc,
                'Rca': start_prob_ca,
                'Lqa': int(start_qla),
                'Lqb': int(start_qlb),
                'Lqc': int(start_qlc)
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
                5000,
            'gt_results_folder':
                f'data/{gt_suffix}',
            'figures_folder':
                f'figures/{results_suffix}',
            'expt_name':
                expt_name
        }

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
    # Generate for a specific delta, we will reuse these queries for the other deltas
    # and for different inference algorithms
    """
    EXPT_NUM = 1
    expt_dets = {1: {'nreps': 30,
                    'time_discretization_experiment': 1,
                    'simulation_config_expt': 1,
                    'edges': [],
                    'max_ql': 14},
                 2: {'nreps': 30,
                    'time_discretization_experiment': 1,
                    'simulation_config_expt': 1,
                    'edges': ['indicator_ql'],
                    'max_ql': 14},
                 }
    NUM_QUERIES = 200
    INFERENCE_ALGORITHM_KEYWORD = 'exact-lazyprop'
    INFERENCE_ALGORITHM = 'LazyPropagation'
    if expt_dets[EXPT_NUM]['edges'] == ["indicator_ql"]:
        INDICATOR_KEYWORD = 'True'
    else:
        INDICATOR_KEYWORD = 'False'

    # Test query workload construction
    CONFIG_NAME = (f'indicator-{INDICATOR_KEYWORD}_inference-{INFERENCE_ALGORITHM_KEYWORD}'
                   f'_nreps-{expt_dets[EXPT_NUM]["nreps"]}')
    # If the file does not exist, create a JSON file with an empty dictionary
    if not os.path.exists(f'configs/{CONFIG_NAME}.json'):
        with open(f'configs/{CONFIG_NAME}.json', 'w', encoding='utf-8') as f:
            json.dump({}, f)
    single_intervention(f'configs/{CONFIG_NAME}.json',
                simulation_config_file='configs/simulator.yaml',
                simulation_config_expt=expt_dets[EXPT_NUM]['simulation_config_expt'],
                time_discretization_experiment=expt_dets[EXPT_NUM][
                    'time_discretization_experiment'],
                edges=expt_dets[EXPT_NUM]['edges'],
                results_suffix='queries',
                gt_suffix='queries_gt',
                maximum_ql=expt_dets[EXPT_NUM]['max_ql'],
                max_ql_queries=10,  # Only for query purposes
                inference_algorithm=INFERENCE_ALGORITHM,
                total_queries=NUM_QUERIES,
                expt_name=CONFIG_NAME)
    """
    # Create a set of 200 queries for the experiment with data pooling possible.
    EXPT_NUM = 1
    expt_dets = {
        1: {
            'nreps': 30,
            'time_discretization_experiment': 12,
            'simulation_config_expt': 12,
            'edges': [],
            'max_ql': 12
        }
    }
    NUM_QUERIES = 200
    NREPS = 30
    INFERENCE_ALGORITHM_KEYWORD = 'exact-lazyprop'
    INFERENCE_ALGORITHM = 'LazyPropagation'
    INDICATOR_KEYWORD = 'False'
    POOLING = 'True'

    # Test query workload construction
    CONFIG_NAME = (
        f'indicator-{INDICATOR_KEYWORD}_inference-{INFERENCE_ALGORITHM_KEYWORD}'
        f'_nreps-{NREPS}_pooled-{POOLING}')
    # If the file does not exist, create a JSON file with an empty dictionary
    if not os.path.exists(f'configs/{CONFIG_NAME}.json'):
        with open(f'configs/{CONFIG_NAME}.json', 'w', encoding='utf-8') as f:
            json.dump({}, f)
    single_intervention(
        f'configs/{CONFIG_NAME}.json',
        simulation_config_file='configs/simulator.yaml',
        simulation_config_expt=expt_dets[EXPT_NUM]['simulation_config_expt'],
        time_discretization_experiment=expt_dets[EXPT_NUM]
        ['time_discretization_experiment'],
        edges=expt_dets[EXPT_NUM]['edges'],
        results_suffix='queries',
        gt_suffix='queries_gt',
        maximum_ql=expt_dets[EXPT_NUM]['max_ql'],
        max_ql_queries=10,  # Only for query purposes
        inference_algorithm=INFERENCE_ALGORITHM,
        total_queries=NUM_QUERIES,
        expt_name=CONFIG_NAME)
