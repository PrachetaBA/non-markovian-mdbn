# pylint: disable=pointless-string-statement, logging-fstring-interpolation
"""Script to construct the DBN from a simulated Er/M/1 queue.

This script constructs the DBN with the specified structure and learns the
CPDs using the simulator data. It saves the DBN so that it can be reused for
similar queries in the future.

Model choices:
 - State variables: QueueLength_t (L_t) and CurrentPhase_t (i_t)
 - Parameters: Lambda_t, Mu_t, K_t
 - Dependencies:
     L_t  <- L_{t-1}, CurrentPhase_{t-1}, Lambda_t, Mu_t
     i_t  <- i_{t-1}, Lambda_t
 - No indicator_ql functionality yet
 - Raw frequency CPDs used (no smoothing)
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
#import pyAgrum as gm
import pyagrum as gm
import yaml

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set logger
logger = logging.getLogger('construct_dbn_erm1_logger')


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


def construct_dbn(bn_file,
                  dbn_file,
                  edges,
                  manual_maxql,
                  store_dbn=False,
                  constructed_dbn_filename=None):
    """Function to construct the DBN using the simulator data (ERM1).

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

    # Define domains for Lambda, Mu, K, CurrentPhase using 2TBN (use tprev unique values)
    unique_lambda = sorted(data_bn['Lambda_tprev'].unique())
    unique_mu = sorted(data_bn['Mu_tprev'].unique())
    #unique_k = sorted(data_bn['K_tprev'].unique())
    #unique_phases = sorted(data_bn['CurrentPhase_tprev'].unique())
    # read as floats
    unique_k = [float(x) for x in sorted(data_bn['K_tprev'].unique())]
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

    # Add fixed arcs as per the specified ERM1 dependencies:
    # L_t  <- L_{t-1}, CurrentPhase_{t-1}, Lambda_t, Mu_t
    # i_t  <- i_{t-1}, Lambda_t
    dbn.addArc(ql0, qlt)            # QueueLength0 -> QueueLengtht
    dbn.addArc(phase0, qlt)         # CurrentPhase0 -> QueueLengtht
    dbn.addArc(lambdat, qlt)        # Lambdat -> QueueLengtht
    dbn.addArc(mut, qlt)            # Mut -> QueueLengtht

    dbn.addArc(phase0, phaset)      # CurrentPhase0 -> CurrentPhaset
    dbn.addArc(lambdat, phaset)     # Lambdat -> CurrentPhaset

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

    # # Step 3. Get empirical counts of all observed values of initial queue lengths
    # # ** add phase length0 too right ?
    # if "QueueLength0" in dbn.names():
    #     name = "QueueLength0"
    #     bn_id = dbn.idFromName(name)
    #     logger.debug(f"Processing variable {name} with id {bn_id}")
    #     parents = list(reversed(dbn.cpt(bn_id).names))
    #     domains = [dbn[p].domainSize() for p in parents]
    #     parents.pop()

    #     # ** shd we add debug prints ? eg. all(p in data_dbn.columns for p in parents)
    #     if len(parents) > 0 and all(p in data_dbn.columns for p in parents):
    #         ctab = pd.crosstab(data_dbn[name], [data_dbn[p] for p in parents], dropna=False, normalize='columns')
    #     elif name in data_dbn.columns:
    #         ctab = data_dbn[name].value_counts(normalize=True)
    #     else:
    #         logger.warning("No data_dbn present for QueueLength0; using uniform marginal.")

    #     reshaped_cpt = np.array((ctab).transpose()).reshape(*domains)
    #     dbn.cpt(bn_id)[:] = reshaped_cpt


    # step 3: initial queue lenth and phase
    # can we do this !?
    initial_state_vars = ["QueueLength0", "CurrentPhase0"]
    for init_var in initial_state_vars:
        if init_var not in dbn.names():
            continue

        name = init_var
        bn_id = dbn.idFromName(name)
        logger.debug(f"Processing variable {name} with id {bn_id}")
        parents = list(reversed(dbn.cpt(bn_id).names))
        domains = [dbn[p].domainSize() for p in parents]
        parents.pop()

        if len(parents) > 0 and all(p in data_dbn.columns for p in parents):
            ctab = pd.crosstab(data_dbn[name], [data_dbn[p] for p in parents], dropna=False, normalize='columns')
        elif name in data_dbn.columns:
            ctab = data_dbn[name].value_counts(normalize=True)
        else:
            logger.warning(f"No data_dbn present for {name}")
            continue

        reshaped_cpt = np.array(ctab.transpose()).reshape(*domains)
        dbn.cpt(bn_id)[:] = reshaped_cpt


    # Step 4. Get empirical counts of all observed values of the parameter variables
    #exclude_names = {"QueueLength0", "QueueLengtht", "CurrentPhase0", "CurrentPhaset"}
    exclude_names = {"QueueLength0", "CurrentPhase0"}
    for name in dbn.names():
        if name in exclude_names:
            continue

        bn_id = dbn.idFromName(name)
        logger.debug(f"Processing parameter variable {name} with id {bn_id}")
        parents = list(reversed(dbn.cpt(bn_id).names))
        domains = [dbn[p].domainSize() for p in parents]
        parents.pop()

        if len(parents) > 0 and all(p in data_bn.columns for p in parents):
            ctab = pd.crosstab(data_bn[name], [data_bn[p] for p in parents], dropna=False, normalize='columns')
        elif name in data_bn.columns:
            ctab = data_bn[name].value_counts(normalize=True)
        else:
            logger.warning(f"No data for parameter {name}: using uniform distribution.")

        # Normalize the CPTs
        reshaped_cpt = np.array((ctab).transpose()).reshape(*domains)
        dbn.cpt(bn_id)[:] = reshaped_cpt

    # Step 5. Learn the probabilities for the phase variables for all time slices != 0
    logger.debug('Learning CPD for CurrentPhaset')
    # We can use pd.crosstab to compute conditional distribution P(CurrentPhaset | CurrentPhase0, Lambdat)
    if {'CurrentPhaset', 'CurrentPhase0', 'Lambdat'}.issubset(set(data_bn.columns)):
        # Build crosstab: rows = current, cols = parents combined (phase0, lambdat)
        ctab = pd.crosstab(data_bn['CurrentPhaset'], [data_bn['CurrentPhase0'], data_bn['Lambdat']], dropna=False, normalize='columns')
        # Shape domains consistent with dbn.cpt ordering
        bn_id = dbn.idFromName('CurrentPhaset')
        parents = list(reversed(dbn.cpt(bn_id).names))
        domains = [dbn[p].domainSize() for p in parents]
        parents.pop()
        reshaped_cpt = np.array((ctab).transpose()).reshape(*domains)
        dbn.cpt(bn_id)[:] = reshaped_cpt
    else:
        logger.warning("Insufficient columns to learn CurrentPhaset CPD from 2TBN.")


    unique_params = data_bn[['Lambda0', 'Mu0', 'K0']].drop_duplicates().reset_index(drop=True)
    fixed_lambda = unique_params.loc[0, 'Lambda0']
    fixed_mu = unique_params.loc[0, 'Mu0']
    fixed_k = unique_params.loc[0, 'K0']

    #max_ql_obs = data_fixed['QueueLength0'].astype(float).max()
    Lt_values = list(range(0, int(max_ql_obs)+1, 2))

    # Set up plot
    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(Lt_values)))

    bn_id = dbn.idFromName("QueueLengtht")
    print(dbn.cpt(bn_id))

    gm.saveBN(dbn,
                  "trial.bif",
                  allowModificationWhenSaving=True)

    #plot_dist = dbn.cpt(bn_id)[{'Lambdat': str(fixed_lambda), 'Mut': str(fixed_mu), 'CurrentPhase0': str(fixed_k), 'QueueLength0': 2}]

    
    for ql0, color in zip(Lt_values, colors):

        plot_dist = dbn.cpt(bn_id)[{'Lambdat': str(fixed_lambda), 'Mut': str(fixed_mu), 'CurrentPhase0': str(fixed_k), 'QueueLength0': ql0}]

        plt.plot(range(len(plot_dist)), plot_dist, color=color, label=f"queuelength={ql0}")

    plt.xlabel("length")
    plt.ylabel("P(Lt | L0)")
    plt.legend()
    plt.grid(True)
    plt.show()



    # # add Lt and then plot
    # # ========================================== testing the plot

    # # picking the first lambda, Mu, K for now
    # unique_params = data_bn[['Lambda0', 'Mu0', 'K0']].drop_duplicates().reset_index(drop=True)
    # fixed_lambda = unique_params.loc[0, 'Lambda0']
    # fixed_mu = unique_params.loc[0, 'Mu0']
    # fixed_k = unique_params.loc[0, 'K0']

    # # get corresponding data
    # data_fixed = data_bn[
    #     (data_bn['Lambda0'] == fixed_lambda) &
    #     (data_bn['Mu0'] == fixed_mu) &
    #     (data_bn['K0'] == fixed_k)
    # ]

    # # x-axis
    # max_ql_obs = data_fixed['QueueLength0'].astype(float).max()
    # Lt_values = list(range(0, int(max_ql_obs)+1, 2))

    # # Set up plot
    # plt.figure(figsize=(8, 6))
    # colors = plt.cm.viridis(np.linspace(0, 1, len(Lt_values)))

    # for i, lt in enumerate(Lt_values):
    #     next_qls = data_fixed[data_fixed['QueueLength0'] == lt]['QueueLengtht']
        
    #     if len(next_qls) == 0:
    #         continue  # skip if no data for this queue length
        
    #     counts = next_qls.value_counts().sort_index()
    #     probs = counts / counts.sum()
        
    #     # all the lengths till max that is observed
    #     all_qls = range(0, int(max_ql_obs)+1)
    #     probs = probs.reindex(all_qls, fill_value=0)
        
    #     plt.plot(all_qls, probs, marker='o', color=colors[i], label=f'L_t = {lt}')

    # plt.xlabel('Queue Lengths')
    # plt.ylabel('P(L(t+1) | L(t))')
    # plt.title(f'lambda = {fixed_lambda}, mu = {fixed_mu}, k = {fixed_k}')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # ========================================== testing the plot



    # # Step 6. Learn the probabilities for the queue length variables for all time slices != 0
    # logger.debug('Learning CPD for QueueLengtht')
    # # Collect relevant columns
    # data_bn_ql = data_bn[[
    #     'QueueLength0', 'CurrentPhase0', 'Lambdat', 'Mut', 'QueueLengtht'
    # ]]

    # for lambda_curr_val in unique_lambda:
    #     for mu_curr_val in unique_mu:
    #         for phase_prev_val in unique_phases:
    #             # For each possible previous queue length state (we will call tm_l with the bucketed state)
    #             for state in range(0, max_ql + 1):
    #                 # Filter the dbn data to only contain specific values of the parameters
    #                 bn2t_data = data_bn_ql[
    #                     (data_bn_ql['Lambdat'] == lambda_curr_val) &
    #                     (data_bn_ql['Mut'] == mu_curr_val) &
    #                     (data_bn_ql['CurrentPhase0'] == phase_prev_val)
    #                 ]

    #                 # Learn the transition matrix for the queue lengths
    #                 bn2t_data = bn2t_data[['QueueLength0', 'QueueLengtht']]
    #                 tm_lt_prob = tm_l(bn2t_data,
    #                                   state,
    #                                   prev_state_col='QueueLength0',
    #                                   curr_state_col='QueueLengtht')

    #                 # ** shd we handle this case or not needed?
    #                 if not tm_lt_prob:
    #                     tm_lt_prob = {0: 1.0}

    #                 if state == 0:
    #                     # Fill in the missing states with 0 probability
    #                     for x in range(max_ql + 1):
    #                         if x not in tm_lt_prob:
    #                             tm_lt_prob[x] = 0.0
    #                 else:
    #                     # Add state to the keys to account for the difference
    #                     tm_lt_prob = {k + state: v for k, v in tm_lt_prob.items()}
    #                     # Fill in the missing states with 0 probability
    #                     for x in range(max_ql + 1):
    #                         if x not in tm_lt_prob:
    #                             tm_lt_prob[x] = 0.0
    #                     # Keep only the keys from 0 to max_ql
    #                     tm_lt_prob = {k: v for k, v in tm_lt_prob.items() if k in range(max_ql + 1)}

    #                 # Sort the dictionary by keys
    #                 tm_lt_prob = dict(sorted(tm_lt_prob.items()))

    #                 # Set the CPT values
    #                 dbn.cpt('QueueLengtht')[{
    #                     'Lambdat': str(lambda_curr_val),
    #                     'Mut': str(mu_curr_val),
    #                     'CurrentPhase0': phase_prev_val,
    #                     'QueueLength0': state
    #                 }] = list(tm_lt_prob.values())

    # logger.debug('Finished processing the queue length variable QueueLengtht')

    # logger.info("ERM1 DBN constructed successfully")

    # # Save the DBN to a file (if requested)
    # if store_dbn and constructed_dbn_filename is not None:
    #     gm.saveBN(dbn, constructed_dbn_filename, allowModificationWhenSaving=True)
    #     logger.info(f"DBN saved to {constructed_dbn_filename}")

    return dbn


if __name__ == "__main__":
    # """Construct and save the ERM1 DBN for a given config.

    # Usage example:
    #   python erm1_construct_dbn.py --config_file configs/queries.yaml --experiment_number 1 -v
    # """
    # parser = argparse.ArgmentParser(
    #     description="Construct the DBN for the ERM1 queueing system.")

    # parser.add_argment(
    #     "--config_file",
    #     "-c",
    #     type=str,
    #     help="Path to the configuration file (e.g. configs/queries.json)",
    #     default="configs/queries.json")
    # parser.add_argment("--experiment_number",
    #                     "-e",
    #                     type=int,
    #                     help="Experiment number (e.g. 1)",
    #                     default=1)
    # parser.add_argment('--verbose',
    #                     '-v',
    #                     help='Increase output verbosity',
    #                     action='store_true',
    #                     default=False,
    #                     required=False)
    # parser.add_argment('--sim_config',
    #                     '-s',
    #                     type=str,
    #                     help='Path to the simulation configuration file',
    #                     default='configs/simulator.yaml',
    #                     required=False)
    # parser.add_argment(
    #     '--time_disc_config',
    #     '-t',
    #     type=str,
    #     help='Path to the time discretization configuration file',
    #     default='configs/time_discretization.yaml',
    #     required=False)

    # args = parser.parse_args()
    # config_file = args.config_file
    # experiment_number = args.experiment_number

    # if args.verbose:
    #     logging.basicConfig(level=logging.DEBUG)
    # else:
    #     logging.basicConfig(level=logging.INFO)

    # # Read YAML config and extract parameters
    # with open(config_file, 'r', encoding='utf-8') as file:
    #     all_configs = yaml.safe_load(file)
    # config = all_configs[f'experiment_{experiment_number}']

    # # Basic config fields we expect (same names as your PI config)
    # time_discretization_experiment = config['time_discretization_experiment']
    # constructed_dbn_folder = config['dbn_output_folder']
    # maximum_queue_length = config['maximum_ql']
    # dbn_edges = config.get('dbn_edges', [])
    # expt_name = config.get('expt_name', 'erm1_experiment')

    # # Read time discretization and simulation configs (to find file names)
    # with open(args.time_disc_config, 'r', encoding='utf-8') as time_discretization_file:
    #     time_discretization_config = yaml.safe_load(time_discretization_file)
    #     time_discretization_params = time_discretization_config[
    #         f'experiment_{time_discretization_experiment}']

    # with open(args.sim_config, 'r', encoding='utf-8') as sim_file:
    #     sim_config = yaml.safe_load(sim_file)
    #     sim_params = sim_config[
    #         f"experiment_{time_discretization_params['time_series_experiment']}"]

    # # Extract some sim params for logging (optional)
    # simulation_reps = sim_params.get('replications', None)
    # simulation_end_time = sim_params.get('simulation_end', None)
    # sampling_interval = time_discretization_params.get('sampling_interval', None)

    # logger.info(f"Experiment: {expt_name}")
    # logger.info(f"Simulation replications: {simulation_reps}")
    # logger.info(f"Simulation end time: {simulation_end_time}")
    # logger.info(f"Sampling interval: {sampling_interval}")

    # # Build file names (match PI conventions)
    # bn_filename = (
    #     f"{time_discretization_params['time_discretization_folder']}"
    #     f"/discrete-time-2tbn-exp-{time_discretization_experiment}.csv")
    # dbn_filename = (
    #     f"{time_discretization_params['time_discretization_folder']}"
    #     f"/discrete-time-dbn-exp-{time_discretization_experiment}.csv")

    # logger.info(f"2TBN filename: {bn_filename}")
    # logger.info(f"DBN filename: {dbn_filename}")

    # if not os.path.exists(constructed_dbn_folder):
    #     os.makedirs(constructed_dbn_folder)
    # config_id = config_file.split('/')[-1].split('.')[0]
    # CONSTRUCTED_DBN_FILENAME = f'{constructed_dbn_folder}/dbn_{config_id}_erm1.bif'
    # logger.info(f'Constructed DBN filename: {CONSTRUCTED_DBN_FILENAME}')

    # if os.path.exists(CONSTRUCTED_DBN_FILENAME):
    #     logger.info(f"DBN file {CONSTRUCTED_DBN_FILENAME} already exists")
    # else:
    #     start = time.time()
    #     construct_dbn(bn_filename,
    #                   dbn_filename,
    #                   dbn_edges,
    #                   maximum_queue_length,
    #                   store_dbn=True,
    #                   constructed_dbn_filename=CONSTRUCTED_DBN_FILENAME)
    #     end = time.time()
    #     logger.info(f"Time taken to construct the ERM1 DBN: {end - start: .2f} seconds")

    project_root = Path(__file__).resolve().parents[1]
    construct_dbn(
        bn_file= project_root / "data/discrete_time/discrete-time-2tbn-erm1-exp-1.csv",
        dbn_file= project_root / "data/discrete_time/discrete-time-dbn-exp-1.csv",
        edges=[],
        manual_maxql=10,
        store_dbn=True,
        constructed_dbn_filename="erlang-queue-mdbn/data/discrete_time/dbn_erm1.bif"
    )
