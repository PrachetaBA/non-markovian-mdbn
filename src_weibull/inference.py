# pylint: disable=pointless-string-statement, logging-fstring-interpolation
"""Script to run inference on the constructed DBN. 

This script reads the DBN specified by the dbn_name. It then runs inference on the DBN to 
predict the state of the system at a future time step for any query that
is specified.
"""
# Import libraries
import argparse
import logging
import math
import os
from pathlib import Path
import pickle
import time
import re
import warnings

import pyAgrum as gm
import pyAgrum.lib.dynamicBN as gdynbn
import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set logger
logger = logging.getLogger('inference_dbn_logger')


def param_val_to_str(param_val):
    """Function used to read the float parameter values and convert them to strings.
    
    For example, if the parameter value is 1.0, it is converted to "1".
    Otherwise, the parameter value 0.2 is converted to _0_2.
    """
    if float(param_val).is_integer():
        return str(int(param_val))
    else:
        param_val = str(param_val).replace(".", "_")
        param_val = f"_{param_val}"
    return param_val


def run_inference_hypoexp(dbn, query, delta):
    """Function to run inference on the weibull HypoExp/M/1 DBN.

    This function runs inference on the DBN to predict the state of the system at a
    future time step for any query that is specified.

    Args:
        dbn (str): The constructed DBN.
        query (str): Query to run inference on the DBN.
        delta (float): Time interval between time slices.
    Returns: 
        posterior (object): The posterior distribution of the query.
    """
    # Unroll the DBN up till the query slice
    query_slice = math.floor(query['query_time'] / delta)
    dbn_unrolled = gdynbn.unroll2TBN(dbn, query_slice + 1)

    # Set the starting parameters
    for key, val in query['start_parameters'].items():
        # Initial queue length
        if key == 'QueueLength':
            dbn_unrolled.cpt('QueueLength0')[:] = 0.0
            dbn_unrolled.cpt('QueueLength0')[{'QueueLength0': int(val)}] = 1.0

        # Initial phase
        elif key == 'CurrentPhase':
            dbn_unrolled.cpt('CurrentPhase0')[:] = 0.0
            dbn_unrolled.cpt('CurrentPhase0')[{'CurrentPhase0': str(val)}] = 1.0

        # Simulation input parameters
        elif key in ['WBshape', 'WBscale', 'Mu']:
            val = param_val_to_str(val) # Change to readable format
            for slice_num in range(query_slice + 1):
                dbn_unrolled.cpt(f"{key}{slice_num}")[:] = 0.0
                dbn_unrolled.cpt(f"{key}{slice_num}")[{
                    f"{key}{slice_num}": str(val)
                }] = 1.0
    # Get the maximum queue length vals
    mql = dbn_unrolled.variable('QueueLength0').domainSize() - 1

    # Store the conditional variables and values for inference
    conditional_vars = []
    conditional_vals = []

    # store inverse intervention variables and values for inference
    inverse_vars = []
    inverse_vals = []

    # Interventions
    if query['interventions']:
        for iv in query['interventions']:

            # Parameter intervention
            if iv['intervention_type'] == 'parameter_intervention':
                intervention_val = param_val_to_str(iv['intervention_value'])
                # Step 1. Convert the intervention times to the corresponding slices
                param_intervention_slice = math.floor(iv['intervention_start'] / delta)
                # Step 2. Set the intervention values for all slices after the intervention
                for slice_num in range(param_intervention_slice,
                                       query_slice + 1):
                    dbn_unrolled.cpt(f"{iv['intervention_variable']}{slice_num}")[:] = 0.0
                    dbn_unrolled.cpt(
                        f"{iv['intervention_variable']}{slice_num}")[{
                            f"{iv['intervention_variable']}{slice_num}":
                                str(intervention_val)
                        }] = 1.0
                # Step 3. Set the current phase to 1 when the parameters have changed only
                # for the intervention slice 
                dbn_unrolled.cpt(f"CurrentPhase{param_intervention_slice}")[:] = 0.0
                dbn_unrolled.cpt(f"CurrentPhase{param_intervention_slice}")[{f"CurrentPhase{param_intervention_slice}": "1"}] = 1.0

            # intervention on state var
            elif iv['intervention_type'] == 'interventional':
                ev_var = f"{iv['intervention_variable']}{math.floor(iv['intervention_start']/delta)}"
                ev_val = int(iv['intervention_value'])

                # Step 1. Erase the arcs from the parents of the intervention var
                for parent in dbn_unrolled.parents(ev_var):
                    parent_var = dbn_unrolled.variable(parent).name()
                    dbn_unrolled.eraseArc(parent_var, ev_var)

                # Step 2. Set the evidence to the corresponding value
                dbn_unrolled.cpt(ev_var)[:] = 0.0
                dbn_unrolled.cpt(ev_var)[ev_val] = 1.0

            # Conditional intervention
            elif iv['intervention_type'] == 'conditional':
                cond_var = f"{iv['intervention_variable']}{math.floor(iv['intervention_start']/delta)}"
                cond_val = int(iv['intervention_value'])
                conditional_vars.append(cond_var)
                conditional_vals.append(cond_val)

            # Additive or subtractive intervention
            elif iv['intervention_type'] in ['additive', 'subtractive']:
                ev_var = f"{iv['intervention_variable']}{math.floor(iv['intervention_start']/delta)}"
                ev_val = int(iv['intervention_value'])

                if iv['intervention_type'] == 'additive':
                    shifted_dbn = gm.BayesNet(dbn_unrolled)
                    for ql in range(0, ev_val):
                        shifted_dbn.cpt(ev_var)[{ev_var: ql}] = 0.0
                    for ql in range(ev_val, mql + 1):
                        shifted_dbn.cpt(ev_var)[{
                            ev_var: ql
                        }] = dbn_unrolled.cpt(ev_var)[{
                            ev_var: ql - ev_val
                        }]
                    # Add the remaining probabilities to the last column
                    remainder = 0.0
                    for rem in range(mql - ev_val + 1, mql + 1):
                        remainder += dbn_unrolled.cpt(ev_var)[{ev_var: rem}]
                    shifted_dbn.cpt(ev_var)[{ev_var: mql}] += remainder
                    dbn_unrolled = gm.BayesNet(shifted_dbn)

                elif iv['intervention_type'] == 'subtractive':
                    shifted_dbn = gm.BayesNet(dbn_unrolled)
                    for ql in range(mql - ev_val + 1, mql + 1):
                        shifted_dbn.cpt(ev_var)[{ev_var: ql}] = 0.0
                    for ql in range(0, mql - ev_val + 1):
                        shifted_dbn.cpt(ev_var)[{
                            ev_var: ql
                        }] = dbn_unrolled.cpt(ev_var)[{
                            ev_var: ql + ev_val
                        }]
                    # Add the remaining probabilities to the first column
                    remainder = 0.0
                    for rem in range(0, ev_val):
                        remainder += dbn_unrolled.cpt(ev_var)[{ev_var: rem}]
                    shifted_dbn.cpt(ev_var)[{ev_var: 0}] += remainder
                    dbn_unrolled = gm.BayesNet(shifted_dbn)

            # Inverse intervention
            elif iv['intervention_type'] == 'inverse_intervention':
                # record queue length evidence for inverse query
                obs_slice = math.floor(iv['intervention_start'] / delta)
                inv_var = f"QueueLength{obs_slice}"
                inv_val = iv['intervention_value']   # could be int or a list

                inverse_vars.append(inv_var)
                inverse_vals.append(inv_val)

                # record which parameter we want posterior for
                inv_param = iv['query_parameter']

    # Choose the type of inference algorithm to use
    inference_engine = getattr(gm, f"{query['inference_algorithm']}")(dbn_unrolled)
    logger.info(f"Running inference using {query['inference_algorithm']} ...")
    # Set the parameters of the inference algorithm 
    if query['inference_algorithm'] == 'GibbsSampling':
        inference_engine.setBurnIn(query['burn_in'])
        inference_engine.setEpsilon(query['epsilon'])
    elif query['inference_algorithm'] == 'MonteCarloSampling': 
        inference_engine.setEpsilon(query['epsilon'])
    else: 
        pass # Do nothing (possible that we have not specified the inference_algorithm)

    # For all interventions of type 'conditional', set the evidence
    for ev_var, ev_val in zip(conditional_vars, conditional_vals):
        inference_engine.addEvidence(ev_var, ev_val)

    # new inverse evidence
    for inv_var, inv_val in zip(inverse_vars, inverse_vals):
        inference_engine.addEvidence(inv_var, inv_val)

    heavy_computation_start_time = time.time()
    inference_engine.makeInference()

    # Extract the posterior probability distribution and return the results
    compute_posterior_start_time = time.time()
    #posterior = inference_engine.posterior(dbn_unrolled.idFromName(f"{query['query_variable']}{query_slice}"))
    if inverse_vars:  # we have inverse query
        posterior = inference_engine.posterior(
            dbn_unrolled.idFromName(f"{inv_param}0")
        )
    else:
        posterior = inference_engine.posterior(
            dbn_unrolled.idFromName(f"{query['query_variable']}{query_slice}")
        )
    compute_posterior_end_time = time.time()

    posterior_dist = {}
    for i in posterior.loopIn():
        key = str(i)
        key = re.search(r"\:(.*?)\>", key)
        posterior_dist[int(key.group(1))] = float(posterior.get(i))

    full_inference_time = compute_posterior_end_time - heavy_computation_start_time
    inference_time = compute_posterior_end_time - compute_posterior_start_time

    return posterior_dist, query_slice, inference_time, full_inference_time


if __name__ == "__main__":
    """Function to construct the DBN and run inference on it.

    Function call: 
    python src/inference.py --config_file configs/queries.json
    --experiment_number 1 -v
    """
    parser = argparse.ArgumentParser(
        description="Construct the DBN for the Markovian queueing system and run inference."
    )
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help=
        "Path to the configuration file (e.g. config/queries.json)",
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
    parser.add_argument(
        '--time_disc_config',
        '-t',
        type=str,
        help='Path to the time discretization configuration file',
        default='config/weibull_hypoexp_time_discretization.yaml',
        required=False)
    parser.add_argument(
        '--dbn_name',
        '-d',
        type=str,
        help='Name of the DBN file to load',
        required=True
    )

    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    experiment_number = args.experiment_number
    dbn_name = args.dbn_name

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read the configuration file and extract the parameters
    with open(config_file, 'r', encoding='utf-8') as file:
        all_configs = yaml.safe_load(file)
    config = all_configs[f'experiment_{experiment_number}']

    # Extract the parameters
    time_discretization_experiment = config['time_discretization_experiment']
    constructed_dbn_folder = config['dbn_output_folder']
    maximum_queue_length = config['maximum_ql']
    expt_name = config['expt_name']

    # Extract the time discretization parameters
    with open(args.time_disc_config, 'r',encoding='utf-8') as time_discretization_file:
        time_discretization_config = yaml.safe_load(time_discretization_file)
        time_discretization_params = time_discretization_config[f'experiment_{time_discretization_experiment}']

    sampling_interval = time_discretization_params['sampling_interval']

    # Print the parameters
    logger.info(f"Sampling Interval: {sampling_interval}")
    logger.info(f"Maximum Queue Length: {maximum_queue_length}")

    project_root = Path(__file__).resolve().parents[1]

    # Constructed DBN filename
    CONSTRUCTED_DBN_FILENAME = project_root / constructed_dbn_folder / f"{dbn_name}.bif"
    logger.info(f"Constructed DBN filename: {CONSTRUCTED_DBN_FILENAME}")

    # Load the DBN specified by the BIF file
    logger.info("Loading the BN specified by the BIF file ...")
    constructed_dbn = gm.BayesNet()
    constructed_dbn.loadBIF(str(CONSTRUCTED_DBN_FILENAME))

    # Run erm1 inference
    logger.info(f"Running inference for experiment {experiment_number}")
    query_dist, total_slices, inference_time, full_inference_time = run_inference_hypoexp(
        constructed_dbn, config, sampling_interval)
    logger.info(f"Posterior distribution: {query_dist}")

    # Create the posterior output folder if it does not exist
    query_workload_name = f"{config_file.split('/')[-1].split('.')[0]}"
    results_folder = f"{config['results_folder']}/{query_workload_name}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save the posterior distribution as a dictionary to a file
    output_dict = {
        'Posterior': query_dist,
        'InferenceTime': inference_time,
        'FullInferenceTime': full_inference_time,
        'TotalSlices': total_slices
    }

    output_filename = f"{results_folder}/weibull-posterior-exp-{experiment_number}.pkl"
    with open(output_filename, 'wb') as file:
        pickle.dump(output_dict, file)


