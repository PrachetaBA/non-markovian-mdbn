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
import pickle
import time
import re
import warnings

#import pyagrum as gm
import pyAgrum as gm
#import pyagrum.lib.dynamicBN as gdynbn
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
    if param_val == 1.0:
        param_val = "1"
    else:
        param_val = str(param_val).replace(".", "_")
        param_val = f"_{param_val}"
    return param_val


def run_inference_erm1(dbn, query, delta):
    """Function to run inference on the ER/M/1 DBN.

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

        # # Initial phase
        # elif key == 'CurrentPhase':
        #     dbn_unrolled.cpt('CurrentPhase0')[:] = 0.0
        #     dbn_unrolled.cpt('CurrentPhase0')[{'CurrentPhase0': str(val)}] = 1.0

        # # parameters
        # elif key in ['Lambda', 'Mu', 'K']:
        #     if key != "K":
        #         val = param_val_to_str(val)
        #     for slice_num in range(query_slice + 1):
        #         dbn_unrolled.cpt(f"{key}{slice_num}")[:] = 0.0
        #         dbn_unrolled.cpt(f"{key}{slice_num}")[{
        #             f"{key}{slice_num}": str(val)
        #         }] = 1.0

        elif key == 'phase_rates':
            # val is a list of floats: the phase rates
            K = len(val)
            
            # Update K variable in DBN
            dbn_unrolled.cpt('K0')[:] = 0.0
            dbn_unrolled.cpt('K0')[{'K0': K}] = 1.0
            
            # CurrentPhase always starts at 1
            dbn_unrolled.cpt('CurrentPhase0')[:] = 0.0
            dbn_unrolled.cpt('CurrentPhase0')[{'CurrentPhase0': 1}] = 1.0
            
            # Lambda per phase (map float to closest DBN label)
            domain_labels = [str(v) for v in dbn_unrolled.variable('Lambda0').labels()]
            
            matched_labels = []
            for rate in val:
                closest_label = min(domain_labels, key=lambda l: abs(rate - float(l.lstrip('_').replace('_','.'))))
                matched_labels.append(closest_label)
            
            # Assign phase rate to slices (cycle through phases)
            for slice_num in range(query_slice + 1):
                phase_idx = slice_num % K
                cpt_var = f"Lambda{slice_num}"
                dbn_unrolled.cpt(cpt_var)[:] = 0.0
                dbn_unrolled.cpt(cpt_var)[{cpt_var: matched_labels[phase_idx]}] = 1.0
            
            # Mu assignment (same for all slices)
            mu_val = query['start_parameters']['Mu']
            mu_val_str = param_val_to_str(mu_val)
            for slice_num in range(query_slice + 1):
                cpt_var = f"Mu{slice_num}"
                dbn_unrolled.cpt(cpt_var)[:] = 0.0
                dbn_unrolled.cpt(cpt_var)[{cpt_var: mu_val_str}] = 1.0

    # Get the maximum queue length vals
    mql = dbn_unrolled.variable('QueueLength0').domainSize() - 1

    # Store the conditional variables and values for inference
    conditional_vars = []
    conditional_vals = []

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

    # Choose the type of inference algorithm to use
    inference_engine = getattr(gm, f"{query['inference_algorithm']}")(dbn_unrolled)
    logger.info(f"Running inference using {query['inference_algorithm']} ...")
    heavy_computation_start_time = time.time()
    inference_engine.makeInference()

    # For all interventions of type 'conditional', set the evidence
    for ev_var, ev_val in zip(conditional_vars, conditional_vals):
        inference_engine.addEvidence(ev_var, ev_val)

    # Extract the posterior probability distribution and return the results
    compute_posterior_start_time = time.time()
    posterior = inference_engine.posterior(dbn_unrolled.idFromName(f"{query['query_variable']}{query_slice}"))
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

    # is this required !?
    # parser.add_argument('--sim_config',
    #                     '-s',
    #                     type=str,
    #                     help='Path to the simulation configuration file',
    #                     default='configs/simulator.yaml',
    #                     required=False)

    parser.add_argument(
        '--time_disc_config',
        '-t',
        type=str,
        help='Path to the time discretization configuration file',
        default='configs/time_discretization.yaml',
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

    # Constructed DBN filename
    CONSTRUCTED_DBN_FILENAME = f"{constructed_dbn_folder}/{dbn_name}.bif"
    logger.info(f"Constructed DBN filename: {CONSTRUCTED_DBN_FILENAME}")

    # Load the DBN specified by the BIF file
    logger.info("Loading the BN specified by the BIF file ...")
    constructed_dbn = gm.BayesNet()
    constructed_dbn.loadBIF(CONSTRUCTED_DBN_FILENAME)

    # Run erm1 inference
    logger.info(f"Running inference for experiment {experiment_number}")
    query_dist, total_slices, inference_time, full_inference_time = run_inference_erm1(
        constructed_dbn, config, sampling_interval)
    logger.info(f"Posterior distribution: {query_dist}")

    # Create the posterior output folder if it does not exist
    results_folder = f"{config['results_folder']}/{config['expt_name']}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Save the posterior distribution as a dictionary to a file
    output_dict = {
        'Posterior': query_dist,
        'InferenceTime': inference_time,
        'FullInferenceTime': full_inference_time,
        'TotalSlices': total_slices
    }

    output_filename = f"{results_folder}/posterior-exp-{experiment_number}.pkl"
    with open(output_filename, 'wb') as file:
        pickle.dump(output_dict, file)



# if __name__ == "__main__":
#     """Function to construct the DBN and run inference on it.

#     Function call: 
#     python src/erm1_inference.py --config_file configs/queries.json
#     --experiment_number 1 -v
#     """
#     parser = argparse.ArgumentParser(
#         description=
#         "Construct the DBN for the Er/M/1 Markovian queueing system and run inference."
#     )

#     parser.add_argument(
#         "--config_file",
#         "-c",
#         type=str,
#         help=
#         "Path to the configuration file (e.g. configs/queries.json)",
#         default="configs/queries.json")
#     parser.add_argument("--experiment_number",
#                         "-e",
#                         type=int,
#                         help="Experiment number (e.g. 1)",
#                         default=1)
#     parser.add_argument('--verbose',
#                         '-v',
#                         help='Increase output verbosity',
#                         action='store_true',
#                         default=False,
#                         required=False)
#     # parser.add_argument('--sim_config',
#     #                     '-s',
#     #                     type=str,
#     #                     help='Path to the simulation configuration file',
#     #                     default='configs/simulator.yaml',
#     #                     required=False)
#     parser.add_argument(
#         '--time_disc_config',
#         '-t',
#         type=str,
#         help='Path to the time discretization configuration file',
#         default='configs/time_discretization.yaml',
#         required=False)
#     parser.add_argument(
#         '--dbn_name',
#         '-d',
#         type=str,
#         help='Name of the DBN file to load',
#         required=True
#     )

#     # Parse the arguments
#     args = parser.parse_args()
#     config_file = args.config_file
#     experiment_number = args.experiment_number
#     dbn_name = args.dbn_name

#     if args.verbose:
#         logging.basicConfig(level=logging.DEBUG)
#     else:
#         logging.basicConfig(level=logging.INFO)

#     # Read the configuration file and extract the parameters
#     with open(config_file, 'r', encoding='utf-8') as file:
#         all_configs = yaml.safe_load(file)
#     config = all_configs[f'experiment_{experiment_number}']

#     # Extract the parameters
#     time_discretization_experiment = config['time_discretization_experiment']
#     constructed_dbn_folder = config['dbn_output_folder']
#     maximum_queue_length = config['maximum_ql']
#     dbn_edges = config['dbn_edges']
#     expt_name = config['expt_name']

#     # Extract the simulation input parameters
#     with open(args.time_disc_config, 'r',
#               encoding='utf-8') as time_discretization_file:
#         time_discretization_config = yaml.safe_load(time_discretization_file)
#         time_discretization_params = time_discretization_config[
#             f'experiment_{time_discretization_experiment}']

#     with open(args.sim_config, 'r', encoding='utf-8') as sim_file:
#         sim_config = yaml.safe_load(sim_file)
#         sim_params = sim_config[
#             f"experiment_{time_discretization_params['time_series_experiment']}"]

#     # Extract and print the parameters
#     experimental_design = sim_params['experimental_design']
#     mean_interarrival_rates_queue_a = sim_params['queue_a_arrival_rates']
#     mean_service_rates_queue_a = sim_params['queue_a_service_rates']
#     mean_interarrival_rates_queue_b = sim_params['queue_b_arrival_rates']
#     mean_service_rates_queue_b = sim_params['queue_b_service_rates']
#     mean_interarrival_rates_queue_c = sim_params['queue_c_arrival_rates']
#     mean_service_rates_queue_c = sim_params['queue_c_service_rates']
#     rp_a_to_b = sim_params['routing_probabilities']['a_to_b']
#     rp_b_to_c = sim_params['routing_probabilities']['b_to_c']
#     rp_c_to_a = sim_params['routing_probabilities']['c_to_a']
#     simulation_reps = sim_params['replications']
#     simulation_end_time = sim_params['simulation_end']
#     num_configs = sim_params['configurations']
#     varying_iql = sim_params['varying_iql']
#     max_iql = sim_params['max_iql']
#     sampling_interval = time_discretization_params['sampling_interval']

#     # Print the parameters
#     logger.info(f"Experimental Design: {experimental_design}")
#     logger.info(
#         f"Mean Interarrival Rates Queue A: {mean_interarrival_rates_queue_a}")
#     logger.info(f"Mean Service Rate Queue A: {mean_service_rates_queue_a}")
#     logger.info(
#         f"Mean Interarrival Rates Queue B: {mean_interarrival_rates_queue_b}")
#     logger.info(f"Mean Service Rate Queue B: {mean_service_rates_queue_b}")
#     logger.info(f"Mean Interarrival Rates Queue C: {mean_interarrival_rates_queue_c}")
#     logger.info(f"Mean Service Rate Queue C: {mean_service_rates_queue_c}")
#     logger.info(f"Routing Probabilities A to B: {rp_a_to_b}")
#     logger.info(f"Routing Probabilities B to C: {rp_b_to_c}")
#     logger.info(f"Routing Probabilities C to A: {rp_c_to_a}")
#     logger.info(f"Number of Replications: {simulation_reps}")
#     logger.info(f"Simulation End Time: {simulation_end_time}")
#     logger.info(f"Number of Configurations: {num_configs}")
#     logger.info(f"Varying Initial Queue Length: {varying_iql}")
#     logger.info(f"Max Initial Queue Length: {max_iql}")
#     logger.info(f"Sampling Interval: {sampling_interval}")

#     # config_id = config_file.split("/")[-1].split(".")[0]
#     # CONSTRUCTED_DBN_FILENAME = f"{constructed_dbn_folder}/dbn_{config_id}.bif"
#     CONSTRUCTED_DBN_FILENAME = f"{constructed_dbn_folder}/{dbn_name}.bif"
#     logger.info(f"Constructed DBN filename: {CONSTRUCTED_DBN_FILENAME}")

#     # Load the DBN specified by the BIF file
#     logger.info("Loading the BN specified by the BIF file ...")
#     constructed_dbn = gm.BayesNet()
#     constructed_dbn.loadBIF(CONSTRUCTED_DBN_FILENAME)

#     logger.info(f"Running inference for experiment {experiment_number}")

#     query_dist, total_slices, inference_time, full_inference_time = run_inference(
#         constructed_dbn, config, sampling_interval)
#     logger.info(f"Posterior distribution: {query_dist}")

#     # Create the posterior output folder if it does not exist
#     results_folder = f"{config['results_folder']}/{config['expt_name']}"
#     if not os.path.exists(results_folder):
#         os.makedirs(results_folder)

#     # Save the posterior distribution as a dictionary to a file
#     output_dict = {
#         'Posterior': query_dist,
#         'InferenceTime': inference_time,
#         'FullInferenceTime': full_inference_time,
#         'TotalSlices': total_slices
#     }
#     output_filename = f"{results_folder}/posterior-exp-{experiment_number}.pkl"
#     with open(output_filename, 'wb') as file:
#         pickle.dump(output_dict, file)





    # ####################### Trial Figures ########################
    # # check if figures dir exists
    # notebook_dir = Path.cwd()
    # project_root = notebook_dir.parent
    # figures_dir = project_root / "figures"
    # figures_dir.mkdir(exist_ok=True)

    # # plot for each lambda, mu, k
    # bn_id = dbn.idFromName("QueueLengtht")
    # Lt_values = list(range(0, int(max_ql)+1))
    # unique_params = data_bn[['Lambda0', 'Mu0', 'K0']].drop_duplicates().reset_index(drop=True)

    # for idx, row in unique_params.iterrows():
    #     fixed_lambda = row["Lambda0"]
    #     fixed_mu     = row["Mu0"]
    #     fixed_k      = row["K0"]

    #     plt.figure(figsize=(8, 6))
    #     colors = plt.cm.viridis(np.linspace(0, 1, len(Lt_values)))

    #     for ql0, color in zip(Lt_values, colors):
    #         plot_dist = dbn.cpt(bn_id)[{'Lambdat': str(fixed_lambda), 'Mut': str(fixed_mu), 'CurrentPhase0': str(fixed_k), 'QueueLength0': ql0}]
    #         plt.plot(range(len(plot_dist)), plot_dist, color=color, label=f"L0={ql0}")

    #     plt.xlabel("Queue Length at t")
    #     plt.ylabel("P(Lt | L0)")
    #     if use_crosstab:
    #         plt.title(f"corsstab used: λ={fixed_lambda}, μ={fixed_mu}, k={fixed_k}")
    #     else:
    #         plt.title(f"tml used: λ={fixed_lambda}, μ={fixed_mu}, k={fixed_k}")
    #     plt.legend()
    #     plt.grid(True)

    #     # save the figure in the figures directory
    #     if use_crosstab:
    #         fname = f"experiment{exp_no}_crosstab_lambda{fixed_lambda}_mu{fixed_mu}_k{fixed_k}.png"
    #     else:
    #         fname = f"experiment{exp_no}_lambda{fixed_lambda}_mu{fixed_mu}_k{fixed_k}.png"
    #     plt.savefig(figures_dir / fname, dpi=300, bbox_inches="tight")
    #     plt.close()
    # ####################### Trial Figures ########################


