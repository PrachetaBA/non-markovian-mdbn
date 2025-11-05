# pylint: disable=pointless-string-statement, logging-fstring-interpolation
"""
This script is used to learn a DBN model and explore the impact of three strategies
1. Parameter extrapolation 
2. Data pooling (combining training data)
3. Data pooling (using indicator variables)

We mainly use this to construct DBNs and visualize the resulting CPDs. It is 
analogous to the script `construct_dbn.py` but with additional functionalities
for plotting and comparing CPDs. 
"""

# Import libraries
import argparse
import logging
import os
import time
import warnings

from collections import defaultdict
import numpy as np
import pandas as pd
import pyAgrum as gm
import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set logger
logger = logging.getLogger(__name__)


def cpt_w_extrapolation(bn2t_data, l0_val, prev_state_col, curr_state_col):
    """Function to learn the transition matrix for the queue lengths depending on
    its structure. 
    
    Args:
        bn2t_data (object): 2TBN data.
        l0_val (str): Initial queue length value.
        prev_state_col (str): Previous state column.
        curr_state_col (str): Current state column.
    Returns:
        tm_l_prob (dict): Transition matrix for the queue lengths.
    """
    tm_l_prob = defaultdict(int)
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
    # Normalize the probabilities
    tm_l_prob = {k: v / sum(tm_l_prob.values()) for k, v in tm_l_prob.items()}
    return tm_l_prob


def construct_dbn(bn_file,
                  dbn_file,
                  edges,
                  manual_maxql,
                  extrapolation=False,
                  store_dbn=False,
                  constructed_dbn_filename=None):
    """Function to construct the DBN using the simulator data.

    This function constructs the DBN with the specified structure and learns the
    CPDs using the simulator data.

    Args:
        bn_file (str): File path to the 2TBN data file.
        dbn_file (str): File path to the DBN data file.
        edges (list): List of edges in the DBN.
        manual_maxql (int): Maximum number of time slices to consider for the DBN.
        store_dbn (bool): Flag to store the constructed DBN.
        dbn_filename (str): File path to store the constructed DBN or None if not storing.
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

    # Specify a large unique value for queue length
    # Find the maximum queue length in the data across the following columns
    max_ql = int(
        max(data_bn[[
            "L_qa_tprev", "L_qb_tprev", "L_qc_tprev", "L_qa", "L_qb", "L_qc"
        ]].max()))
    # or manually set the maximum queue length
    max_ql = max(max_ql, manual_maxql)
    logger.info(f"Maximum queue length ever observed or specified: {max_ql}")

    # Define domain for the simulation input parameters
    unique_lambda_qa = sorted(data_bn['Lambda_qa_tprev'].unique())
    unique_mu_qa = sorted(data_bn['Mu_qa_tprev'].unique())
    unique_lambda_qb = sorted(data_bn['Lambda_qb_tprev'].unique())
    unique_mu_qb = sorted(data_bn['Mu_qb_tprev'].unique())
    unique_lambda_qc = sorted(data_bn['Lambda_qc_tprev'].unique())
    unique_mu_qc = sorted(data_bn['Mu_qc_tprev'].unique())
    unique_r_ab = sorted(data_bn['R_ab_tprev'].unique())
    unique_r_bc = sorted(data_bn['R_bc_tprev'].unique())
    unique_r_ca = sorted(data_bn['R_ca_tprev'].unique())

    # Create the variables of the DBN
    # NOTE: Naming convention really matters, we cannot have other integers in the
    # variable names. We should have a 2TBN relevant naming convention where
    # all variables are suffixed by "0" or "t" to indicate the previous and
    # current time slices.

    # Technically, we have other parameters such as the routing probabilities into
    # each of the queues that are controlling the transitions between the queues.
    # However, in this case, we have provided a single default value of 0.5 to all
    # the routing probabilities. Hence we chose not to represent them as variables
    # in the DBN.

    # Previous time slice variables
    lambda_qa_tprev = gm.NumericalDiscreteVariable(
        "Lambdaqa0", "Queue A arrival rates (t-delta)", unique_lambda_qa)
    mu_qa_tprev = gm.NumericalDiscreteVariable(
        "Muqa0", "Queue A service rates (t-delta)", unique_mu_qa)
    lambda_qb_tprev = gm.NumericalDiscreteVariable(
        "Lambdaqb0", "Queue B arrival rates (t-delta)", unique_lambda_qb)
    mu_qb_tprev = gm.NumericalDiscreteVariable(
        "Muqb0", "Queue B service rates (t-delta)", unique_mu_qb)
    lambda_qc_tprev = gm.NumericalDiscreteVariable(
        "Lambdaqc0", "Queue C arrival rates (t-delta)", unique_lambda_qc)
    mu_qc_tprev = gm.NumericalDiscreteVariable(
        "Muqc0", "Queue C service rates (t-delta)", unique_mu_qc)
    r_ab_tprev = gm.NumericalDiscreteVariable(
        "Rab0", "Routing probability from A to B (t-delta)", unique_r_ab)
    r_bc_tprev = gm.NumericalDiscreteVariable(
        "Rbc0", "Routing probability from B to C (t-delta)", unique_r_bc)
    r_ca_tprev = gm.NumericalDiscreteVariable(
        "Rca0", "Routing probability from C to A (t-delta)", unique_r_ca)
    qa_ql_tprev = gm.RangeVariable("Lqa0", "Queue A queue length (t-delta)", 0,
                                   max_ql)
    qb_ql_tprev = gm.RangeVariable("Lqb0", "Queue B queue length (t-delta)", 0,
                                   max_ql)
    qc_ql_tprev = gm.RangeVariable("Lqc0", "Queue C queue length (t-delta)", 0,
                                   max_ql)

    # Current time slice variables
    lambda_qa = gm.NumericalDiscreteVariable("Lambdaqat",
                                             "Queue A arrival rates",
                                             unique_lambda_qa)
    mu_qa = gm.NumericalDiscreteVariable("Muqat", "Queue A service rates",
                                         unique_mu_qa)
    lambda_qb = gm.NumericalDiscreteVariable("Lambdaqbt",
                                             "Queue B arrival rates",
                                             unique_lambda_qb)
    mu_qb = gm.NumericalDiscreteVariable("Muqbt", "Queue B service rates",
                                         unique_mu_qb)
    lambda_qc = gm.NumericalDiscreteVariable("Lambdaqct",
                                             "Queue C arrival rates",
                                             unique_lambda_qc)
    mu_qc = gm.NumericalDiscreteVariable("Muqct", "Queue C service rates",
                                         unique_mu_qc)
    r_ab = gm.NumericalDiscreteVariable("Rabt",
                                        "Routing probability from A to B",
                                        unique_r_ab)
    r_bc = gm.NumericalDiscreteVariable("Rbct",
                                        "Routing probability from B to C",
                                        unique_r_bc)
    r_ca = gm.NumericalDiscreteVariable("Rcat",
                                        "Routing probability from C to A",
                                        unique_r_ca)
    qa_ql = gm.RangeVariable("Lqat", "Queue A queue length", 0, max_ql)
    qb_ql = gm.RangeVariable("Lqbt", "Queue B queue length", 0, max_ql)
    qc_ql = gm.RangeVariable("Lqct", "Queue C queue length", 0, max_ql)

    # Create the BN
    dbn = gm.BayesNet()
    # Add the variables to the DBN
    lambdaqa0, muqa0, lambdaqb0, muqb0, lambdaqc0, muqc0, rab0, rbc0, rca0, lqa0, lqb0, lqc0 = [
        dbn.add(x) for x in [
            lambda_qa_tprev, mu_qa_tprev, lambda_qb_tprev, mu_qb_tprev,
            lambda_qc_tprev, mu_qc_tprev, r_ab_tprev, r_bc_tprev, r_ca_tprev,
            qa_ql_tprev, qb_ql_tprev, qc_ql_tprev
        ]
    ]
    lambdaqat, muqat, lambdaqbt, muqbt, lambdaqct, muqct, rabt, rbct, rcat, lqat, lqbt, lqct = [
        dbn.add(x) for x in [
            lambda_qa, mu_qa, lambda_qb, mu_qb, lambda_qc, mu_qc, r_ab, r_bc,
            r_ca, qa_ql, qb_ql, qc_ql
        ]
    ]

    # Add the fixed (known) arcs between the variables
    # MM1-related arcs Queue A
    dbn.addArc(lambdaqa0, lqa0)
    dbn.addArc(muqa0, lqa0)
    dbn.addArc(lambdaqat, lqat)
    dbn.addArc(muqat, lqat)
    dbn.addArc(lqa0, lqat)
    # Routing probabilities and Queue C inputs to Queue A
    dbn.addArc(rca0, lqa0)
    dbn.addArc(rcat, lqat)
    dbn.addArc(lambdaqc0, lqa0)
    dbn.addArc(lambdaqct, lqat)
    dbn.addArc(muqc0, lqa0)
    dbn.addArc(muqct, lqat)

    # MM1-related Queue B
    dbn.addArc(lambdaqb0, lqb0)
    dbn.addArc(muqb0, lqb0)
    dbn.addArc(lambdaqbt, lqbt)
    dbn.addArc(muqbt, lqbt)
    dbn.addArc(lqb0, lqbt)
    # Routing probabilities and Queue A inputs to Queue B
    dbn.addArc(rab0, lqb0)
    dbn.addArc(rabt, lqbt)
    dbn.addArc(lambdaqa0, lqb0)
    dbn.addArc(muqa0, lqb0)
    dbn.addArc(lambdaqat, lqbt)
    dbn.addArc(muqat, lqbt)

    # MM1-related Queue C
    dbn.addArc(lambdaqc0, lqc0)
    dbn.addArc(muqc0, lqc0)
    dbn.addArc(lambdaqct, lqct)
    dbn.addArc(muqct, lqct)
    dbn.addArc(lqc0, lqct)
    # Routing probabilities and Queue B inputs to Queue C
    dbn.addArc(lambdaqb0, lqc0)
    dbn.addArc(muqb0, lqc0)
    dbn.addArc(rbc0, lqc0)
    dbn.addArc(lambdaqbt, lqct)
    dbn.addArc(muqbt, lqct)
    dbn.addArc(rbct, lqct)

    if 'indicator_ql' in edges:
        # Create the indicator queue length variables
        qai_ql_tprev = gm.NumericalDiscreteVariable(
            "Lqai0", "Queue A queue length indicator (t-delta)", [0, 1])
        qbi_ql_tprev = gm.NumericalDiscreteVariable(
            "Lqbi0", "Queue B queue length indicator (t-delta)", [0, 1])
        qci_ql_tprev = gm.NumericalDiscreteVariable(
            "Lqci0", "Queue C queue length indicator (t-delta)", [0, 1])
        qai_ql = gm.NumericalDiscreteVariable("Lqait",
                                              "Queue A queue length indicator",
                                              [0, 1])
        qbi_ql = gm.NumericalDiscreteVariable("Lqbit",
                                              "Queue B queue length indicator",
                                              [0, 1])
        qci_ql = gm.NumericalDiscreteVariable("Lqcit",
                                              "Queue C queue length indicator",
                                              [0, 1])
        lqai0, lqbi0, lqci0 = [
            dbn.add(x) for x in [qai_ql_tprev, qbi_ql_tprev, qci_ql_tprev]
        ]
        lqait, lqbit, lqcit = [dbn.add(x) for x in [qai_ql, qbi_ql, qci_ql]]
        # Add the arcs between the indicator variables
        dbn.addArc(lqa0, lqai0)
        dbn.addArc(lqat, lqait)
        dbn.addArc(lqb0, lqbi0)
        dbn.addArc(lqbt, lqbit)
        dbn.addArc(lqc0, lqci0)
        dbn.addArc(lqct, lqcit)
        # Add the arcs between the indicator variable of the previous
        # queue to the queue length of the current queue
        dbn.addArc(lqai0, lqbt)
        dbn.addArc(lqbi0, lqct)
        dbn.addArc(lqci0, lqat)
    else:  # Add the arcs between the queue lengths of the previous and current queues
        dbn.addArc(lqa0, lqbt)
        dbn.addArc(lqb0, lqct)
        dbn.addArc(lqc0, lqat)

    # Print the DBN
    logger.debug(f"DBN: {dbn}")

    # Populate the CPTs of the DBN using the Pandas cross tab function
    # Step 1. Rename the columns of the data_bn dataframe to match the variable names
    data_bn.columns = [
        "Lambdaqa0", "Muqa0", "Lambdaqb0", "Muqb0", "Lambdaqc0", "Muqc0",
        "Rab0", "Rbc0", "Rca0", "Lqa0", "Lqb0", "Lqc0", "Lambdaqat", "Muqat",
        "Lambdaqbt", "Muqbt", "Lambdaqct", "Muqct", "Rabt", "Rbct", "Rcat",
        "Lqat", "Lqbt", "Lqct"
    ]

    # Step 2. Set the domain of all variables in the pandas dataframe
    data_bn["Lambdaqa0"] = pd.Categorical(data_bn["Lambdaqa0"],
                                          categories=unique_lambda_qa)
    data_bn["Muqa0"] = pd.Categorical(data_bn["Muqa0"], categories=unique_mu_qa)
    data_bn["Lambdaqb0"] = pd.Categorical(data_bn["Lambdaqb0"],
                                          categories=unique_lambda_qb)
    data_bn["Muqb0"] = pd.Categorical(data_bn["Muqb0"], categories=unique_mu_qb)
    data_bn["Lambdaqc0"] = pd.Categorical(data_bn["Lambdaqc0"],
                                          categories=unique_lambda_qc)
    data_bn["Muqc0"] = pd.Categorical(data_bn["Muqc0"], categories=unique_mu_qc)
    data_bn["Rab0"] = pd.Categorical(data_bn["Rab0"], categories=unique_r_ab)
    data_bn["Rbc0"] = pd.Categorical(data_bn["Rbc0"], categories=unique_r_bc)
    data_bn["Rca0"] = pd.Categorical(data_bn["Rca0"], categories=unique_r_ca)
    data_bn["Lqa0"] = pd.Categorical(data_bn["Lqa0"],
                                     categories=range(0, max_ql + 1))
    data_bn["Lqb0"] = pd.Categorical(data_bn["Lqb0"],
                                     categories=range(0, max_ql + 1))
    data_bn["Lqc0"] = pd.Categorical(data_bn["Lqc0"],
                                     categories=range(0, max_ql + 1))
    data_bn["Lambdaqat"] = pd.Categorical(data_bn["Lambdaqat"],
                                          categories=unique_lambda_qa)
    data_bn["Muqat"] = pd.Categorical(data_bn["Muqat"], categories=unique_mu_qa)
    data_bn["Lambdaqbt"] = pd.Categorical(data_bn["Lambdaqbt"],
                                          categories=unique_lambda_qb)
    data_bn["Muqbt"] = pd.Categorical(data_bn["Muqbt"], categories=unique_mu_qb)
    data_bn["Lambdaqct"] = pd.Categorical(data_bn["Lambdaqct"],
                                          categories=unique_lambda_qc)
    data_bn["Muqct"] = pd.Categorical(data_bn["Muqct"], categories=unique_mu_qc)
    data_bn["Rabt"] = pd.Categorical(data_bn["Rabt"], categories=unique_r_ab)
    data_bn["Rbct"] = pd.Categorical(data_bn["Rbct"], categories=unique_r_bc)
    data_bn["Rcat"] = pd.Categorical(data_bn["Rcat"], categories=unique_r_ca)
    data_bn["Lqat"] = pd.Categorical(data_bn["Lqat"],
                                     categories=range(0, max_ql + 1))
    data_bn["Lqbt"] = pd.Categorical(data_bn["Lqbt"],
                                     categories=range(0, max_ql + 1))
    data_bn["Lqct"] = pd.Categorical(data_bn["Lqct"],
                                     categories=range(0, max_ql + 1))
    # Do the same for the dbn data
    data_dbn["Lambdaqa0"] = pd.Categorical(data_dbn["Lambdaqa0"],
                                           categories=unique_lambda_qa)
    data_dbn["Muqa0"] = pd.Categorical(data_dbn["Muqa0"],
                                       categories=unique_mu_qa)
    data_dbn["Lambdaqb0"] = pd.Categorical(data_dbn["Lambdaqb0"],
                                           categories=unique_lambda_qb)
    data_dbn["Muqb0"] = pd.Categorical(data_dbn["Muqb0"],
                                       categories=unique_mu_qb)
    data_dbn["Lambdaqc0"] = pd.Categorical(data_dbn["Lambdaqc0"],
                                           categories=unique_lambda_qc)
    data_dbn["Muqc0"] = pd.Categorical(data_dbn["Muqc0"],
                                       categories=unique_mu_qc)
    data_dbn["Rab0"] = pd.Categorical(data_dbn["Rab0"], categories=unique_r_ab)
    data_dbn["Rbc0"] = pd.Categorical(data_dbn["Rbc0"], categories=unique_r_bc)
    data_dbn["Rca0"] = pd.Categorical(data_dbn["Rca0"], categories=unique_r_ca)
    data_dbn["Lqa0"] = pd.Categorical(data_dbn["Lqa0"],
                                      categories=range(0, max_ql + 1))
    data_dbn["Lqb0"] = pd.Categorical(data_dbn["Lqb0"],
                                      categories=range(0, max_ql + 1))
    data_dbn["Lqc0"] = pd.Categorical(data_dbn["Lqc0"],
                                      categories=range(0, max_ql + 1))

    # Create the additional rows of data_bn and data_dbn that correspond to the indicator variables
    if 'indicator_ql' in edges:
        # Create additional columns for data_bn, according to the existing values of Lqa0 and Lqat,
        # Lqb0 and Lqbt, Lqc0 and Lqct
        data_bn['Lqai0'] = np.where(data_bn['Lqa0'] == 0, 0, 1)
        data_bn['Lqait'] = np.where(data_bn['Lqat'] == 0, 0, 1)
        data_bn['Lqbi0'] = np.where(data_bn['Lqb0'] == 0, 0, 1)
        data_bn['Lqbit'] = np.where(data_bn['Lqbt'] == 0, 0, 1)
        data_bn['Lqci0'] = np.where(data_bn['Lqc0'] == 0, 0, 1)
        data_bn['Lqcit'] = np.where(data_bn['Lqct'] == 0, 0, 1)

    # Step 3. Get empirical counts of all observed values of initial queue lengths
    for name in ["Lqa0", "Lqb0", "Lqc0"]:
        bn_id = dbn.idFromName(name)
        logger.debug(f"Processing variable {name} with id {bn_id}")
        parents = list(reversed(dbn.cpt(bn_id).names))
        domains = [dbn[name].domainSize() for name in parents]
        parents.pop()

        if len(parents) > 0:
            ctab = pd.crosstab(data_dbn[name], [data_dbn[p] for p in parents],
                               dropna=False,
                               normalize='columns')
        else:
            ctab = data_dbn[name].value_counts(normalize=True)

        # Normalize the CPTs
        reshaped_cpt = np.array((ctab).transpose()).reshape(*domains)
        dbn.cpt(bn_id)[:] = reshaped_cpt

    # Step 4. Get empirical counts of all observed values of the parameter variables
    for name in dbn.names():
        if name not in [
                "Lqa0", "Lqb0", "Lqc0", "Lqat", "Lqbt", "Lqct", "Lqai0",
                "Lqbi0", "Lqci0"
        ]:
            bn_id = dbn.idFromName(name)
            logger.debug(f"Processing variable {name} with id {bn_id}")
            parents = list(reversed(dbn.cpt(bn_id).names))
            domains = [dbn[name].domainSize() for name in parents]
            parents.pop()  # Remove the same variable from the list

            if len(parents) > 0:
                ctab = pd.crosstab(data_bn[name], [data_bn[p] for p in parents],
                                   dropna=False,
                                   normalize='columns')
            else:
                ctab = data_bn[name].value_counts(normalize=True)

            # Normalize the CPTs
            reshaped_cpt = np.array((ctab).transpose()).reshape(*domains)
            dbn.cpt(bn_id)[:] = reshaped_cpt

    # Set the starting CPD for Lpi0 if necessary
    if 'indicator_ql' in edges:
        dbn.cpt('Lqai0')[:] = dbn.cpt('Lqait').toarray()
        dbn.cpt('Lqbi0')[:] = dbn.cpt('Lqbit').toarray()
        dbn.cpt('Lqci0')[:] = dbn.cpt('Lqcit').toarray()

    # Step 5. Learn the probabilities for the queue length variables for
    # all time slices != 0

    # Note that by default here we do data pooling, which means that
    # the user has to ensure that the simulation input parameters values for
    # all parameters have the same domain.

    # With indicator random variables.
    if 'indicator_ql' in edges:
        data_bn_qa = data_bn[[
            'Lqa0', 'Lqci0', 'Lambdaqat', 'Muqat', 'Lambdaqct', 'Muqct', 'Rcat',
            'Lqat'
        ]]
        data_bn_qb = data_bn[[
            'Lqb0', 'Lqai0', 'Lambdaqbt', 'Muqbt', 'Lambdaqat', 'Muqat', 'Rabt',
            'Lqbt'
        ]]
        data_bn_qc = data_bn[[
            'Lqc0', 'Lqbi0', 'Lambdaqct', 'Muqct', 'Lambdaqbt', 'Muqbt', 'Rbct',
            'Lqct'
        ]]

        # Pool the data by renaming the variables to generically be a the
        # parent and the child queue
        data_bn_qa_renamed = data_bn_qa.rename(
            columns={
                'Lqa0': 'Lcurrq0',
                'Lqci0': 'Lprevqi0',
                'Lambdaqat': 'Lambdacurrqt',
                'Muqat': 'Mucurrqt',
                'Lambdaqct': 'Lambdaprevqt',
                'Muqct': 'Muprevqt',
                'Rcat': 'Rt',
                'Lqat': 'Lcurrqt'
            })
        data_bn_qb_renamed = data_bn_qb.rename(
            columns={
                'Lqb0': 'Lcurrq0',
                'Lqai0': 'Lprevqi0',
                'Lambdaqbt': 'Lambdacurrqt',
                'Muqbt': 'Mucurrqt',
                'Lambdaqat': 'Lambdaprevqt',
                'Muqat': 'Muprevqt',
                'Rabt': 'Rt',
                'Lqbt': 'Lcurrqt'
            })
        data_bn_qc_renamed = data_bn_qc.rename(
            columns={
                'Lqc0': 'Lcurrq0',
                'Lqbi0': 'Lprevqi0',
                'Lambdaqct': 'Lambdacurrqt',
                'Muqct': 'Mucurrqt',
                'Lambdaqbt': 'Lambdaprevqt',
                'Muqbt': 'Muprevqt',
                'Rbct': 'Rt',
                'Lqct': 'Lcurrqt'
            })
        # Combine the data from the three queues, by horizontally stacking them
        data_bn_combined = pd.concat(
            [data_bn_qa_renamed, data_bn_qb_renamed, data_bn_qc_renamed])

        # For all the queues learn the conditional probability tables
        logger.debug(
            "Learning the CPTs using data pooling, extrapolation and indicator variables"
        )
        if extrapolation:
            for lambda_curr_val in unique_lambda_qa:
                for mu_curr_val in unique_mu_qa:
                    for lambda_prev_val in unique_lambda_qb:
                        for mu_prev_val in unique_mu_qb:
                            for r_curr_val in unique_r_ab:
                                for lqi0_val in [0, 1]:
                                    for state in range(0, max_ql + 1):
                                        # Filter the data to only contain the specific
                                        # values of these parameters
                                        bn2t_data = data_bn_combined[
                                            (data_bn_combined['Lambdacurrqt'] ==
                                             lambda_curr_val) &
                                            (data_bn_combined['Mucurrqt'] ==
                                             mu_curr_val) &
                                            (data_bn_combined['Lambdaprevqt']
                                             == lambda_prev_val) &
                                            (data_bn_combined['Muprevqt']
                                             == mu_prev_val) &
                                            (data_bn_combined['Rt']
                                             == r_curr_val) &
                                            (data_bn_combined['Lprevqi0']
                                             == lqi0_val)]
                                        # Learn the transition matrix for the queue lengths
                                        bn2t_data = bn2t_data[[
                                            'Lcurrq0', 'Lcurrqt'
                                        ]]
                                        if extrapolation:
                                            tm_lt_prob = cpt_w_extrapolation(
                                                bn2t_data,
                                                state,
                                                prev_state_col='Lcurrq0',
                                                curr_state_col='Lcurrqt')
                                            if state == 0:
                                                # Fill in the missing states with 0 probability
                                                for x in range(max_ql + 1):
                                                    if x not in tm_lt_prob:
                                                        tm_lt_prob[x] = 0.0
                                            elif state >= 1:
                                                # Add state to the keys to account for difference
                                                tm_lt_prob = {
                                                    k + state: v for k, v in
                                                    tm_lt_prob.items()
                                                }
                                                # Fill in the missing states with 0 probability
                                                for x in range(max_ql + 1):
                                                    if x not in tm_lt_prob:
                                                        tm_lt_prob[x] = 0.0
                                                # Keep only the keys from 0 to max_ql
                                                tm_lt_prob = {
                                                    k: v
                                                    for k, v in
                                                    tm_lt_prob.items()
                                                    if k in range(max_ql + 1)
                                                }
                                            # Sort the dictionary by keys
                                            tm_lt_prob = dict(
                                                sorted(tm_lt_prob.items()))
                                            # Set the CPT values for all the queues
                                            dbn.cpt('Lqat')[{
                                                'Lambdaqat':
                                                    str(lambda_curr_val),
                                                'Muqat':
                                                    str(mu_curr_val),
                                                'Lambdaqct':
                                                    str(lambda_prev_val),
                                                'Muqct':
                                                    str(mu_prev_val),
                                                'Rcat':
                                                    str(r_curr_val),
                                                'Lqci0':
                                                    lqi0_val,
                                                'Lqa0':
                                                    state
                                            }] = list(tm_lt_prob.values())
                                            dbn.cpt('Lqbt')[{
                                                'Lambdaqbt':
                                                    str(lambda_curr_val),
                                                'Muqbt':
                                                    str(mu_curr_val),
                                                'Lambdaqat':
                                                    str(lambda_prev_val),
                                                'Muqat':
                                                    str(mu_prev_val),
                                                'Rabt':
                                                    str(r_curr_val),
                                                'Lqai0':
                                                    lqi0_val,
                                                'Lqb0':
                                                    state
                                            }] = list(tm_lt_prob.values())
                                            dbn.cpt('Lqct')[{
                                                'Lambdaqct':
                                                    str(lambda_curr_val),
                                                'Muqct':
                                                    str(mu_curr_val),
                                                'Lambdaqbt':
                                                    str(lambda_prev_val),
                                                'Muqbt':
                                                    str(mu_prev_val),
                                                'Rbct':
                                                    str(r_curr_val),
                                                'Lqbi0':
                                                    lqi0_val,
                                                'Lqc0':
                                                    state
                                            }] = list(tm_lt_prob.values())
        else:
            # Pool the data and learn the CPTs for each queue individually
            data_bn_qa_pooled = data_bn_combined.rename(
                columns={
                    'Lcurrq0': 'Lqa0',
                    'Lprevqi0': 'Lqci0',
                    'Lambdacurrqt': 'Lambdaqat',
                    'Mucurrqt': 'Muqat',
                    'Lambdaprevqt': 'Lambdaqct',
                    'Muprevqt': 'Muqct',
                    'Rt': 'Rcat',
                    'Lcurrqt': 'Lqat'
                })
            # Learn the CPT for queue A
            bn_id = dbn.idFromName('Lqat')
            logger.debug(f"Processing variable Lqat with id {bn_id}")
            parents = list(reversed(dbn.cpt(bn_id).names))
            domains = [dbn[name].domainSize() for name in parents]
            parents.pop()  # Remove the same variable from the list
            ctab = pd.crosstab(data_bn_qa_pooled['Lqat'],
                               [data_bn_qa_pooled[p] for p in parents],
                               dropna=False,
                               normalize='columns')
            reshaped_cpt = np.array(
                (ctab).transpose()).reshape(*domains)  # Normalize the CPTs
            dbn.cpt(bn_id)[:] = reshaped_cpt

            # Learn the CPT for queue B
            data_bn_qb_pooled = data_bn_combined.rename(
                columns={
                    'Lcurrq0': 'Lqb0',
                    'Lprevqi0': 'Lqai0',
                    'Lambdacurrqt': 'Lambdaqbt',
                    'Mucurrqt': 'Muqbt',
                    'Lambdaprevqt': 'Lambdaqat',
                    'Muprevqt': 'Muqat',
                    'Rt': 'Rabt',
                    'Lcurrqt': 'Lqbt'
                })
            bn_id = dbn.idFromName('Lqbt')
            logger.debug(f"Processing variable Lqbt with id {bn_id}")
            parents = list(reversed(dbn.cpt(bn_id).names))
            domains = [dbn[name].domainSize() for name in parents]
            parents.pop()  # Remove the same variable from the list
            ctab = pd.crosstab(data_bn_qb_pooled['Lqbt'],
                               [data_bn_qb_pooled[p] for p in parents],
                               dropna=False,
                               normalize='columns')
            reshaped_cpt = np.array(
                (ctab).transpose()).reshape(*domains)  # Normalize the CPTs
            dbn.cpt(bn_id)[:] = reshaped_cpt

            # Learn the CPT for queue C
            data_bn_qc_pooled = data_bn_combined.rename(
                columns={
                    'Lcurrq0': 'Lqc0',
                    'Lprevqi0': 'Lqbi0',
                    'Lambdacurrqt': 'Lambdaqct',
                    'Mucurrqt': 'Muqct',
                    'Lambdaprevqt': 'Lambdaqbt',
                    'Muprevqt': 'Muqbt',
                    'Rt': 'Rbct',
                    'Lcurrqt': 'Lqct'
                })
            bn_id = dbn.idFromName('Lqct')
            logger.debug(f"Processing variable Lqct with id {bn_id}")
            parents = list(reversed(dbn.cpt(bn_id).names))
            domains = [dbn[name].domainSize() for name in parents]
            parents.pop()  # Remove the same variable from the list
            ctab = pd.crosstab(data_bn_qc_pooled['Lqct'],
                               [data_bn_qc_pooled[p] for p in parents],
                               dropna=False,
                               normalize='columns')
            reshaped_cpt = np.array(
                (ctab).transpose()).reshape(*domains)  # Normalize the CPTs
            dbn.cpt(bn_id)[:] = reshaped_cpt
    else:  # No indicator variables
        data_bn_qa = data_bn[[
            'Lqa0', 'Lqc0', 'Lambdaqat', 'Muqat', 'Lambdaqct', 'Muqct', 'Rcat',
            'Lqat'
        ]]
        data_bn_qb = data_bn[[
            'Lqb0', 'Lqa0', 'Lambdaqbt', 'Muqbt', 'Lambdaqat', 'Muqat', 'Rabt',
            'Lqbt'
        ]]
        data_bn_qc = data_bn[[
            'Lqc0', 'Lqb0', 'Lambdaqct', 'Muqct', 'Lambdaqbt', 'Muqbt', 'Rbct',
            'Lqct'
        ]]

        # Pool the data by renaming the variables to generically be a the
        # parent and the child queue
        data_bn_qa_renamed = data_bn_qa.rename(
            columns={
                'Lqa0': 'Lcurrq0',
                'Lqc0': 'Lprevq0',
                'Lambdaqat': 'Lambdacurrqt',
                'Muqat': 'Mucurrqt',
                'Lambdaqct': 'Lambdaprevqt',
                'Muqct': 'Muprevqt',
                'Rcat': 'Rt',
                'Lqat': 'Lcurrqt'
            })
        data_bn_qb_renamed = data_bn_qb.rename(
            columns={
                'Lqb0': 'Lcurrq0',
                'Lqa0': 'Lprevq0',
                'Lambdaqbt': 'Lambdacurrqt',
                'Muqbt': 'Mucurrqt',
                'Lambdaqat': 'Lambdaprevqt',
                'Muqat': 'Muprevqt',
                'Rabt': 'Rt',
                'Lqbt': 'Lcurrqt'
            })
        data_bn_qc_renamed = data_bn_qc.rename(
            columns={
                'Lqc0': 'Lcurrq0',
                'Lqb0': 'Lprevq0',
                'Lambdaqct': 'Lambdacurrqt',
                'Muqct': 'Mucurrqt',
                'Lambdaqbt': 'Lambdaprevqt',
                'Muqbt': 'Muprevqt',
                'Rbct': 'Rt',
                'Lqct': 'Lcurrqt'
            })
        # Combine the data from the three queues, by horizontally stacking them
        data_bn_combined = pd.concat(
            [data_bn_qa_renamed, data_bn_qb_renamed, data_bn_qc_renamed])

        logger.debug(
            f"Data for the combined queues: \n{data_bn_combined.head()}")

        # For all the queues learn the conditional probability tables
        logger.debug(
            "Learning the CPTs using data pooling, extrapolation and no indicator variables"
        )
        if extrapolation:
            for lambda_curr_val in unique_lambda_qa:
                for mu_curr_val in unique_mu_qa:
                    for lambda_prev_val in unique_lambda_qb:
                        for mu_prev_val in unique_mu_qb:
                            for r_curr_val in unique_r_ab:
                                for lq_prev0_val in range(0, max_ql + 1):
                                    for state in range(0, max_ql + 1):
                                        # Filter the data to only contain the specific
                                        # values of these parameters
                                        bn2t_data = data_bn_combined[
                                            (data_bn_combined['Lambdacurrqt'] ==
                                             lambda_curr_val) &
                                            (data_bn_combined['Mucurrqt'] ==
                                             mu_curr_val) &
                                            (data_bn_combined['Lambdaprevqt']
                                             == lambda_prev_val) &
                                            (data_bn_combined['Muprevqt']
                                             == mu_prev_val) &
                                            (data_bn_combined['Rt']
                                             == r_curr_val) &
                                            (data_bn_combined['Lprevq0']
                                             == lq_prev0_val)]
                                        # Learn the transition matrix for the queue lengths
                                        bn2t_data = bn2t_data[[
                                            'Lcurrq0', 'Lcurrqt'
                                        ]]
                                        if extrapolation:
                                            tm_lt_prob = cpt_w_extrapolation(
                                                bn2t_data,
                                                state,
                                                prev_state_col='Lcurrq0',
                                                curr_state_col='Lcurrqt')
                                            if state == 0:
                                                # Fill in the missing states with 0 probability
                                                for x in range(max_ql + 1):
                                                    if x not in tm_lt_prob:
                                                        tm_lt_prob[x] = 0.0
                                            elif state >= 1:
                                                # Add state to the keys to account for difference
                                                tm_lt_prob = {
                                                    k + state: v for k, v in
                                                    tm_lt_prob.items()
                                                }
                                                # Fill in the missing states with 0 probability
                                                for x in range(max_ql + 1):
                                                    if x not in tm_lt_prob:
                                                        tm_lt_prob[x] = 0.0
                                                # Keep only the keys from 0 to max_ql
                                                tm_lt_prob = {
                                                    k: v
                                                    for k, v in
                                                    tm_lt_prob.items()
                                                    if k in range(max_ql + 1)
                                                }
                                            # Sort the dictionary by keys
                                            tm_lt_prob = dict(
                                                sorted(tm_lt_prob.items()))
                                            # Set the CPT values for all the queues
                                            dbn.cpt('Lqat')[{
                                                'Lambdaqat':
                                                    str(lambda_curr_val),
                                                'Muqat':
                                                    str(mu_curr_val),
                                                'Lambdaqct':
                                                    str(lambda_prev_val),
                                                'Muqct':
                                                    str(mu_prev_val),
                                                'Rcat':
                                                    str(r_curr_val),
                                                'Lqc0':
                                                    lq_prev0_val,
                                                'Lqa0':
                                                    state
                                            }] = list(tm_lt_prob.values())
                                            dbn.cpt('Lqbt')[{
                                                'Lambdaqbt':
                                                    str(lambda_curr_val),
                                                'Muqbt':
                                                    str(mu_curr_val),
                                                'Lambdaqat':
                                                    str(lambda_prev_val),
                                                'Muqat':
                                                    str(mu_prev_val),
                                                'Rabt':
                                                    str(r_curr_val),
                                                'Lqa0':
                                                    lq_prev0_val,
                                                'Lqb0':
                                                    state
                                            }] = list(tm_lt_prob.values())
                                            dbn.cpt('Lqct')[{
                                                'Lambdaqct':
                                                    str(lambda_curr_val),
                                                'Muqct':
                                                    str(mu_curr_val),
                                                'Lambdaqbt':
                                                    str(lambda_prev_val),
                                                'Muqbt':
                                                    str(mu_prev_val),
                                                'Rbct':
                                                    str(r_curr_val),
                                                'Lqb0':
                                                    lq_prev0_val,
                                                'Lqc0':
                                                    state
                                            }] = list(tm_lt_prob.values())
        else:
            # Pool the data and learn the CPTs for each queue individually
            data_bn_qa_pooled = data_bn_combined.rename(
                columns={
                    'Lcurrq0': 'Lqa0',
                    'Lprevq0': 'Lqc0',
                    'Lambdacurrqt': 'Lambdaqat',
                    'Mucurrqt': 'Muqat',
                    'Lambdaprevqt': 'Lambdaqct',
                    'Muprevqt': 'Muqct',
                    'Rt': 'Rcat',
                    'Lcurrqt': 'Lqat'
                })
            # Learn the CPT for queue A
            bn_id = dbn.idFromName('Lqat')
            logger.debug(f"Processing variable Lqat with id {bn_id}")
            parents = list(reversed(dbn.cpt(bn_id).names))
            domains = [dbn[name].domainSize() for name in parents]
            parents.pop()  # Remove the same variable from the list
            ctab = pd.crosstab(data_bn_qa_pooled['Lqat'],
                               [data_bn_qa_pooled[p] for p in parents],
                               dropna=False,
                               normalize='columns')
            reshaped_cpt = np.array(
                (ctab).transpose()).reshape(*domains)  # Normalize the CPTs
            dbn.cpt(bn_id)[:] = reshaped_cpt

            # Learn the CPT for queue B
            data_bn_qb_pooled = data_bn_combined.rename(
                columns={
                    'Lcurrq0': 'Lqb0',
                    'Lprevq0': 'Lqa0',
                    'Lambdacurrqt': 'Lambdaqbt',
                    'Mucurrqt': 'Muqbt',
                    'Lambdaprevqt': 'Lambdaqat',
                    'Muprevqt': 'Muqat',
                    'Rt': 'Rabt',
                    'Lcurrqt': 'Lqbt'
                })
            bn_id = dbn.idFromName('Lqbt')
            logger.debug(f"Processing variable Lqbt with id {bn_id}")
            parents = list(reversed(dbn.cpt(bn_id).names))
            domains = [dbn[name].domainSize() for name in parents]
            parents.pop()  # Remove the same variable from the list
            ctab = pd.crosstab(data_bn_qb_pooled['Lqbt'],
                               [data_bn_qb_pooled[p] for p in parents],
                               dropna=False,
                               normalize='columns')
            reshaped_cpt = np.array(
                (ctab).transpose()).reshape(*domains)  # Normalize the CPTs
            dbn.cpt(bn_id)[:] = reshaped_cpt

            # Learn the CPT for queue C
            data_bn_qc_pooled = data_bn_combined.rename(
                columns={
                    'Lcurrq0': 'Lqc0',
                    'Lprevq0': 'Lqb0',
                    'Lambdacurrqt': 'Lambdaqct',
                    'Mucurrqt': 'Muqct',
                    'Lambdaprevqt': 'Lambdaqbt',
                    'Muprevqt': 'Muqbt',
                    'Rt': 'Rbct',
                    'Lcurrqt': 'Lqct'
                })
            bn_id = dbn.idFromName('Lqct')
            logger.debug(f"Processing variable Lqct with id {bn_id}")
            parents = list(reversed(dbn.cpt(bn_id).names))
            domains = [dbn[name].domainSize() for name in parents]
            parents.pop()  # Remove the same variable from the list
            ctab = pd.crosstab(data_bn_qc_pooled['Lqct'],
                               [data_bn_qc_pooled[p] for p in parents],
                               dropna=False,
                               normalize='columns')
            reshaped_cpt = np.array(
                (ctab).transpose()).reshape(*domains)  # Normalize the CPTs
            dbn.cpt(bn_id)[:] = reshaped_cpt

    # Step 6. Save the constructed DBN
    logger.info("DBN constructed successfully")

    # Save the DBN to a file
    if store_dbn and constructed_dbn_filename is not None:
        gm.saveBN(dbn,
                  constructed_dbn_filename,
                  allowModificationWhenSaving=True)
        logger.info(f"DBN saved to {constructed_dbn_filename}")


if __name__ == "__main__":
    """Function to construct the DBN and save it for future loading.

    Function call: 
    python src/dbn_extrapolation_pooling.py --config_file configs/queries.yaml
    --experiment_number 1 -v
    """
    parser = argparse.ArgumentParser(
        description="Construct the DBN for the Markovian queueing system.")

    parser.add_argument(
        "--config_file",
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
                        default='configs/simulator.yaml',
                        required=False)
    parser.add_argument(
        '--time_disc_config',
        '-t',
        type=str,
        help='Path to the time discretization configuration file',
        default='configs/time_discretization.yaml',
        required=False)
    # Parse the arguments
    args = parser.parse_args()
    config_file = args.config_file
    experiment_number = args.experiment_number

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
    dbn_edges = config['dbn_edges']
    expt_name = config['expt_name']
    logger.debug(f"Experiment name is {expt_name}")

    # Extract the simulation input parameters
    with open(args.time_disc_config, 'r',
              encoding='utf-8') as time_discretization_file:
        time_discretization_config = yaml.safe_load(time_discretization_file)
        time_discretization_params = time_discretization_config[
            f'experiment_{time_discretization_experiment}']

    with open(args.sim_config, 'r', encoding='utf-8') as sim_file:
        sim_config = yaml.safe_load(sim_file)
        sim_params = sim_config[
            f"experiment_{time_discretization_params['time_series_experiment']}"]

    # Extract and print the parameters
    experimental_design = sim_params['experimental_design']
    mean_interarrival_rates_queue_a = sim_params['queue_a_arrival_rates']
    mean_service_rates_queue_a = sim_params['queue_a_service_rates']
    mean_interarrival_rates_queue_b = sim_params['queue_b_arrival_rates']
    mean_service_rates_queue_b = sim_params['queue_b_service_rates']
    mean_interarrival_rates_queue_c = sim_params['queue_c_arrival_rates']
    mean_service_rates_queue_c = sim_params['queue_c_service_rates']
    rp_a_to_b = sim_params['routing_probabilities']['a_to_b']
    rp_b_to_c = sim_params['routing_probabilities']['b_to_c']
    rp_c_to_a = sim_params['routing_probabilities']['c_to_a']
    simulation_reps = sim_params['replications']
    simulation_end_time = sim_params['simulation_end']
    num_configs = sim_params['configurations']
    varying_iql = sim_params['varying_iql']
    max_iql = sim_params['max_iql']
    sampling_interval = time_discretization_params['sampling_interval']

    # Print the parameters
    logger.info(f"Experimental Design: {experimental_design}")
    logger.info(
        f"Mean Interarrival Rates Queue A: {mean_interarrival_rates_queue_a}")
    logger.info(f"Mean Service Rate Queue A: {mean_service_rates_queue_a}")
    logger.info(
        f"Mean Interarrival Rates Queue B: {mean_interarrival_rates_queue_b}")
    logger.info(f"Mean Service Rate Queue B: {mean_service_rates_queue_b}")
    logger.info(
        f"Mean Interarrival Rates Queue C: {mean_interarrival_rates_queue_c}")
    logger.info(f"Mean Service Rate Queue C: {mean_service_rates_queue_c}")
    logger.info(f"Routing Probabilities A to B: {rp_a_to_b}")
    logger.info(f"Routing Probabilities B to C: {rp_b_to_c}")
    logger.info(f"Routing Probabilities C to A: {rp_c_to_a}")
    logger.info(f"Number of Replications: {simulation_reps}")
    logger.info(f"Simulation End Time: {simulation_end_time}")
    logger.info(f"Number of Configurations: {num_configs}")
    logger.info(f"Varying Initial Queue Length: {varying_iql}")
    logger.info(f"Max Initial Queue Length: {max_iql}")
    logger.info(f"Sampling Interval: {sampling_interval}")

    # Define specific filenames for the bn file, dbn file and the
    # constructed dbn
    bn_filename = (
        f"{time_discretization_params['time_discretization_folder']}"
        f"/discrete-time-2tbn-exp-{time_discretization_experiment}.csv")
    dbn_filename = (
        f"{time_discretization_params['time_discretization_folder']}"
        f"/discrete-time-dbn-exp-{time_discretization_experiment}.csv")

    # Print the filenames
    logger.info(f"2TBN filename: {bn_filename}")
    logger.info(f"DBN filename: {dbn_filename}")

    # We assume that for each queries.json file, we will have a unique
    # DBN that will be constructed.
    # All other parameters will be the same (like types of parent variables)

    # Extract the experiment name from the configuration file
    extrapolation = expt_name.split('_')[-1].split('-')[-1]
    if extrapolation == 'False':
        extrapolation = False
    elif extrapolation == 'True':
        extrapolation = True
    logger.debug(f"Extrapolation: {extrapolation}")
    logger.debug(f"Extrapolation type: {type(extrapolation)}")
    if not os.path.exists(constructed_dbn_folder):
        os.makedirs(constructed_dbn_folder)
    config_id = config_file.split('/')[-1].split('.')[0]
    CONSTRUCTED_DBN_FILENAME = (
        f'{constructed_dbn_folder}/dbn_{config_id}_extrapolation-{extrapolation}.bif')
    logger.info(f'Constructed DBN filename: {CONSTRUCTED_DBN_FILENAME}')

    if os.path.exists(CONSTRUCTED_DBN_FILENAME):
        logger.info(f"DBN file {CONSTRUCTED_DBN_FILENAME} already exists")
    else:  # Construct the DBN
        # Log the time taken to construct the DBN
        start = time.time()
        construct_dbn(bn_filename,
                      dbn_filename,
                      dbn_edges,
                      maximum_queue_length,
                      extrapolation=extrapolation,
                      store_dbn=True,
                      constructed_dbn_filename=CONSTRUCTED_DBN_FILENAME)
        end = time.time()
        logger.info(
            f"Time taken to construct the DBN: {end - start: .2f} seconds")
