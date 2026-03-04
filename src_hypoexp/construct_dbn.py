# pylint: disable=pointless-string-statement, logging-fstring-interpolation
"""Script to construct the DBN from a simulated Hypoexp/M/1 queue.

This script constructs the DBN with the specified structure and learns the
CPDs using the simulator data. It saves the DBN so that it can be reused for
similar queries in the future.

Model choices:
 - State variables: QueueLength_t (L_t) and CurrentPhase_t (i_t)
 - Parameters: Lambda_t, Mu_t, K_t
 - Dependencies:
     L_t  <- L_{t-1}, CurrentPhase_{t-1}, Mu_t
     i_t  <- i_{t-1}, Lambda_t
"""
import argparse
import logging
import os
from pathlib import Path
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pyAgrum as gm
#import pyagrum as gm
import yaml

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger('construct_dbn_hypoexp_m1_logger') # set logger


def tm_l(bn2t_data, l0_val, prev_state_col, curr_state_col):
    """Function to learn the transition matrix for the queue lengths depending on
    its structure.

    Args:
        bn2t_data (pd.DataFrame): 2TBN data
        l0_val (int): init queue length value
        prev_state_col (str): Column name for previous queue length
        curr_state_col (str): Column name for current queue length

    Returns:
        dict: transition matrix
    """
    tm_l_prob = defaultdict(int)
    # Count observed increments for rows matching the l0_val bucket
    for i in range(len(bn2t_data)):
        curr_state = bn2t_data.iloc[i][prev_state_col]
        if l0_val == 0:
            if curr_state != 0:
                continue
        elif l0_val == 1:
            if curr_state != 1:
                continue
        elif l0_val > 1:
            if curr_state <= 1:
                continue
        next_state = bn2t_data.iloc[i][curr_state_col]
        difference = int(next_state - curr_state)
        tm_l_prob[difference] += 1

    # if no data observed return empty dict
    if len(tm_l_prob) == 0:
        return {}

    # Normalize to probabilities
    total = float(sum(tm_l_prob.values()))
    tm_l_prob = {k: v / total for k, v in tm_l_prob.items()}
    return tm_l_prob


def change_to_queuelength_distribution(change_dist, prev_queue_length, max_queue_length):
    """
    Convert change distribution to absolute queue length distribution
    """
    # init probabilities for all possible queue lengths
    next_queue_probs = {q_len: 0.0 for q_len in range(0, max_queue_length + 1)}

    # Go through each change and add prob
    for change, prob in change_dist.items():
        next_queue_length = prev_queue_length + int(change)
        next_queue_length = max(0, min(max_queue_length, next_queue_length))
        next_queue_probs[next_queue_length] += prob

    # get total prob and normalize
    total_prob = sum(next_queue_probs.values())
    if total_prob <= 0.0:
        next_queue_probs = {q_len: 0.0 for q_len in range(0, max_queue_length + 1)}
        next_queue_probs[prev_queue_length] = 1.0
    else: # normalize
        next_queue_probs = {q_len: prob / total_prob for q_len, prob in next_queue_probs.items()}

    return next_queue_probs


def learn_cpd_using_crosstab(dbn, var_name, data, logger=None):
    """
    Learn the CPD of a DBN variable from data using crosstab function.
    The parent set is inferred from the DBN structure.

    dbn: The DBN containing the variable and its parents.
    var_name: Name of the variable whose CPD should be learned.
    data: Dataframe (DBN or 2TBN) containing the variable and its parents.
    logger: To log the messages.
    """
    # check if variable present
    if var_name not in dbn.names():
        if logger:
            logger.warning(f"Variable {var_name} not in DBN")
        return

    # get the necessary variables
    bn_id = dbn.idFromName(var_name)
    parents = list(reversed(dbn.cpt(bn_id).names))
    domains = [dbn[p].domainSize() for p in parents]
    parents.pop()
    if logger:
        logger.debug(f"Learning CPD for {var_name} with parents {parents}")

    # use cross-tab depending on number of parents
    if len(parents) > 0 and all(p in data.columns for p in parents + [var_name]):
        ctab = pd.crosstab(data[var_name], [data[p] for p in parents], dropna=False, normalize='columns')
    elif var_name in data.columns:
        ctab = data[var_name].value_counts(normalize=True)
    else:
        if logger:
            logger.warning(f"No data present for {var_name}")
        return

    # reshape to match CPT dims
    reshaped_cpt = np.array(ctab.transpose()).reshape(*domains)
    dbn.cpt(bn_id)[:] = reshaped_cpt


def construct_dbn(bn_file,
                  dbn_file,
                  edges,
                  manual_maxql,
                  store_dbn=False,
                  constructed_dbn_filename=None,
                  use_crosstab=True,
                  exp_no=1):
    """Function to construct the DBN using the simulator data (HypoexpM1).

    This function constructs the DBN with the specified structure and learns the
    CPDs using the simulator data.

    Args:
        bn_file (str): File path to the 2TBN data file.
        dbn_file (str): File path to the DBN data file.
        ** edges (list): skipped for now fix later
        manual_maxql (int): maximum queue length.
        store_dbn (bool): Flag to store the constructed DBN.
        constructed_dbn_filename (str): File path to store the constructed DBN or None if not storing.
    Returns:
        constructed_dbn (object): The constructed DBN object.
    """
    # Read the 2TBN file
    if os.path.exists(bn_file):
        data_bn = pd.read_csv(bn_file)
    else:
        raise FileNotFoundError(f"{bn_file} does not exist.")

    # Read the time series data file
    if os.path.exists(dbn_file):
        data_dbn = pd.read_csv(dbn_file, index_col=0)
    else:
        raise FileNotFoundError(f"{dbn_file} does not exist.")

    # Determine maximum queue length seen in the 2TBN data (prev and current)
    max_ql_obs = int(max(data_bn[['QueueLength_tprev', 'QueueLength']].max()))
    max_ql = max(max_ql_obs, manual_maxql)
    logger.info(f"Maximum queue length ever observed or specified: {max_ql}")

    # Define domains for Lambda, Mu, K, CurrentPhase using 2TBN
    unique_lambda = sorted(data_bn['Lambda_tprev'].unique())
    unique_mu = sorted(data_bn['Mu_tprev'].unique())
    unique_k = [float(x) for x in sorted(data_bn['K_tprev'].unique())] # read as floats
    unique_phases = [float(x) for x in sorted(data_bn['CurrentPhase_tprev'].unique())]

    # Create the variables of the DBN (naming: suffix "0" for previous slice, suffix "t" for current slice)
    # Previous time-slice variables
    lambda_tprev = gm.NumericalDiscreteVariable("Lambda0", "Arrival rate (t - 1)", unique_lambda)
    mu_tprev = gm.NumericalDiscreteVariable("Mu0", "Service rate (t - 1)", unique_mu)
    k_tprev = gm.NumericalDiscreteVariable("K0", "Erlang phases (t - 1)", unique_k)
    phase_tprev = gm.NumericalDiscreteVariable("CurrentPhase0", "Current phase (t - 1)", unique_phases)
    ql_tprev = gm.RangeVariable("QueueLength0", "Queue length (t - 1)", 0, max_ql)
    # Current time-slice variables
    lambda_t = gm.NumericalDiscreteVariable("Lambdat", "Arrival rate (t)", unique_lambda)
    mu_t = gm.NumericalDiscreteVariable("Mut", "Service rate (t)", unique_mu)
    k_t = gm.NumericalDiscreteVariable("Kt", "Erlang phases (t)", unique_k)
    phase_t = gm.NumericalDiscreteVariable("CurrentPhaset", "Current phase (t)", unique_phases)
    ql_t = gm.RangeVariable("QueueLengtht", "Queue length (t)", 0, max_ql)

    # Create DBN and add variables
    dbn = gm.BayesNet()
    (lambda0, mu0, k0, phase0, ql0) = [
        dbn.add(x) for x in [lambda_tprev, mu_tprev, k_tprev, phase_tprev, ql_tprev]
    ]
    (lambdat, mut, kdat, phaset, qlt) = [
        dbn.add(x) for x in [lambda_t, mu_t, k_t, phase_t, ql_t]
    ]

    # Add fixed arcs as per the specified Hypoexp M1 dependencies:
    # L_t  <- L_{t-1}, CurrentPhase_{t-1}, Lambda_t, Mu_t
    dbn.addArc(ql0, qlt)
    dbn.addArc(phase0, qlt)
    dbn.addArc(mut, qlt)
    # i_t  <- i_{t-1}, Lambda_t
    dbn.addArc(phase0, phaset)
    dbn.addArc(lambdat, phaset)

    # Print the DBN
    logger.debug(f"DBN: {dbn}")

 
    # Step 1. Rename the columns of the 2TBN dataframe to match the DBN variable names
    data_bn.columns = [
        "Lambda0", "Mu0", "K0", "CurrentPhase0", "QueueLength0",
        "Lambdat", "Mut", "Kt", "CurrentPhaset", "QueueLengtht"
    ]

    # Step 2. Set the domain of all relevant variables in the pandas dataframes
    data_bn["Lambda0"] = pd.Categorical(data_bn["Lambda0"], categories=unique_lambda)
    data_bn["Mu0"] = pd.Categorical(data_bn["Mu0"], categories=unique_mu)
    data_bn["K0"] = pd.Categorical(data_bn["K0"], categories=unique_k)
    data_bn["CurrentPhase0"] = pd.Categorical(data_bn["CurrentPhase0"], categories=unique_phases)
    data_bn["QueueLength0"] = pd.Categorical(data_bn["QueueLength0"], categories=range(0, max_ql + 1))
    data_bn["Lambdat"] = pd.Categorical(data_bn["Lambdat"], categories=unique_lambda)
    data_bn["Mut"] = pd.Categorical(data_bn["Mut"], categories=unique_mu)
    data_bn["Kt"] = pd.Categorical(data_bn["Kt"], categories=unique_k)
    data_bn["CurrentPhaset"] = pd.Categorical(data_bn["CurrentPhaset"], categories=unique_phases)
    data_bn["QueueLengtht"] = pd.Categorical(data_bn["QueueLengtht"], categories=range(0, max_ql + 1))

    # For data_dbn (time-series DBN file): set categories for the "initial" slice columns we will use
    # We expect data_dbn to contain columns for slice 0 named "Lambda0","Mu0","K0","CurrentPhase0","QueueLength0"
    for col, cats in [
        ("Lambda0", unique_lambda),
        ("Mu0", unique_mu),
        ("K0", unique_k),
        ("CurrentPhase0", unique_phases),
        ("QueueLength0", range(0, max_ql + 1))
    ]:
        if col in data_dbn.columns:
            data_dbn[col] = pd.Categorical(data_dbn[col], categories=cats)
        else:
            logger.debug(f"Column {col} not present in data_dbn")


    # step 3: initial queue lenth and phase
    initial_state_vars = ["QueueLength0", "CurrentPhase0"]
    for init_var in initial_state_vars:
        learn_cpd_using_crosstab(dbn, init_var, data_dbn, logger)

    # step 4: CPD for phase
    learn_cpd_using_crosstab(dbn, "CurrentPhaset", data_bn, logger)

    # step 5: CPD for QueueLength (group by phases and service rates)
    for mu_value in unique_mu:
        for phase_value in unique_phases:

            # group by mu and phase
            df_group = data_bn[
                (data_bn['Mut'] == mu_value) &
                (data_bn['CurrentPhase0'] == phase_value)
            ][['QueueLength0', 'QueueLengtht']]

            # get distributions for the 3 buckets
            tm_0 = tm_l(df_group, 0, 'QueueLength0', 'QueueLengtht')
            tm_1 = tm_l(df_group, 1, 'QueueLength0', 'QueueLengtht')
            tm_ge2 = tm_l(df_group, 2, 'QueueLength0', 'QueueLengtht')

            # incase we get empty dicts
            if not tm_0:
                tm_0 = {0: 1.0}
            if not tm_1:
                tm_1 = {0: 1.0}
            if not tm_ge2:
                tm_ge2 = {0: 1.0}

            # assign CPTs
            for queue_len in range(0, max_ql + 1):

                if queue_len == 0:
                    abs_prob = change_to_queuelength_distribution(tm_0, queue_len, max_ql)
                elif queue_len == 1:
                    abs_prob = change_to_queuelength_distribution(tm_1, queue_len, max_ql)
                else: # pooled for >=2
                    abs_prob = change_to_queuelength_distribution(tm_ge2, queue_len, max_ql)

                prob_list = [abs_prob[ql] for ql in range(0, max_ql + 1)]

                dbn.cpt('QueueLengtht')[{
                    'Mut': str(mu_value),
                    'CurrentPhase0': str(phase_value),
                    'QueueLength0': queue_len
                }] = prob_list

    logger.debug('Finished processing the queue length variable QueueLengtht')
    logger.info("DBN constructed successfully")

    # Save the DBN to a file
    if store_dbn and constructed_dbn_filename is not None:
        gm.saveBN(dbn, str(constructed_dbn_filename), allowModificationWhenSaving=True)
        logger.info(f"DBN saved to {constructed_dbn_filename}")


if __name__ == "__main__":
    """Construct and save the Hypoexp/M/1 DBN for a given config.

    Usage example:
      python construct_dbn.py --config_file configs/queries.yaml --experiment_number 1 -v
    """
    parser = argparse.ArgumentParser(description="Construct the DBN for the Hypoexp M1 queueing system.")
    parser.add_argument("--config_file",
                        "-c",
                        type=str,
                        help="Path to the configuration file (e.g. configs/queries.json)",
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
    parser.add_argument('--sim_config',
                        '-s',
                        type=str,
                        help='Path to the simulation configuration file',
                        default='config/hypoexp_simulator.yaml',
                        required=False)
    parser.add_argument('--time_disc_config',
                        '-t',
                        type=str,
                        help='Path to the time discretization configuration file',
                        default='config/hypoexp_time_discretization.yaml',
                        required=False)

    args = parser.parse_args()
    config_file = args.config_file
    experiment_number = args.experiment_number

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Read YAML config and extract parameters
    with open(config_file, 'r', encoding='utf-8') as file:
        all_configs = yaml.safe_load(file)
    config = all_configs[f'experiment_{experiment_number}']

    # config fields
    time_discretization_experiment = config['time_discretization_experiment']
    constructed_dbn_folder = config['dbn_output_folder']
    maximum_queue_length = config['maximum_ql']
    dbn_edges = config.get('dbn_edges', [])
    expt_name = config.get('expt_name', 'hypoexpm1_experiment')
    use_crosstab = config.get('use_crosstab', False)

    # Read time discretization and simulation configs
    with open(args.time_disc_config, 'r', encoding='utf-8') as time_discretization_file:
        time_discretization_config = yaml.safe_load(time_discretization_file)
        time_discretization_params = time_discretization_config[f'experiment_{time_discretization_experiment}']

    with open(args.sim_config, 'r', encoding='utf-8') as sim_file:
        sim_config = yaml.safe_load(sim_file)
        sim_params = sim_config[f"experiment_{time_discretization_params['hypoexp_m1_time_series_experiment']}"]

    # logging ome parameters
    simulation_reps = sim_params.get('runs', None)
    simulation_end_time = sim_params.get('simulation_end', None)
    sampling_interval = time_discretization_params.get('sampling_interval', None)
    logger.info(f"Experiment: {expt_name}")
    logger.info(f"Simulation replications: {simulation_reps}")
    logger.info(f"Simulation end time: {simulation_end_time}")
    logger.info(f"Sampling interval: {sampling_interval}")

    # build the filenames
    bn_filename = (
        f"{time_discretization_params['time_discretization_folder']}"
        f"/hypoexp-discrete-time-2tbn-exp-{time_discretization_experiment}.csv")
    dbn_filename = (
        f"{time_discretization_params['time_discretization_folder']}"
        f"/hypoexp-discrete-time-dbn-exp-{time_discretization_experiment}.csv")

    logger.info(f"2TBN filename: {bn_filename}")
    logger.info(f"DBN filename: {dbn_filename}")

    # get constructed dbn file name
    project_root = Path(__file__).resolve().parents[1]
    if use_crosstab:
        CONSTRUCTED_DBN_FILENAME = project_root / f"data/discrete_time/dbn_hypoexpm1_exp{time_discretization_experiment}_crosstab.bif"
    else:
        CONSTRUCTED_DBN_FILENAME = project_root / f"data/discrete_time/dbn_hypoexpm1_exp{time_discretization_experiment}.bif"
    logger.info(f'Constructed DBN filename: {CONSTRUCTED_DBN_FILENAME}')

    # construct the dbn
    if os.path.exists(CONSTRUCTED_DBN_FILENAME):
        logger.info(f"DBN file {CONSTRUCTED_DBN_FILENAME} already exists")
    else:
        start = time.time()
        construct_dbn(project_root / f"data/{bn_filename}",
                      project_root / f"data/{dbn_filename}",
                      dbn_edges,
                      maximum_queue_length,
                      store_dbn=True,
                      constructed_dbn_filename=CONSTRUCTED_DBN_FILENAME,
                      use_crosstab=use_crosstab,
                      exp_no=time_discretization_experiment)
        end = time.time()
        logger.info(f"Time taken to construct the Hypoexp M1 DBN: {end - start: .2f} seconds")
