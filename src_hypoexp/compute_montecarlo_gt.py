# pylint: disable=logging-fstring-interpolation, pointless-string-statement
"""Function to compute and store the ground truth distribution.

This script computes the ground truth distribution using 
the Monte Carlo simulations.
"""

# Import libraries
import argparse
import logging
import pickle
import os

import math
import pandas as pd
from scipy.stats import norm, beta
import yaml

logger = logging.getLogger('compute_montecarlo_gt')


def compute_gt_pd(query, exp_num, gt_folder=None, suffix=""):
    """Compute the ground truth probability distribution.

    Args:
        query: A dictionary containing the details of the query.
        exp_num: The experiment number.

    Returns:
        gt_pd: A pandas dataframe containing the ground truth probability distribution.
        conf_intervals: A dictionary containing the confidence intervals.
    """

    # Extract the query details
    query_variable = 'QueueLength'
    gt_filepath = f"{gt_folder}/gt-exp-{exp_num}{suffix}.csv"
    df = pd.read_csv(gt_filepath)

    # Use the query to determine how many conditional events there should exist and if they
    # are satisfied
    # Compute the number of conditional events in the query
    number_of_conditional_events = 0
    if query['interventions']:
        for iv in query['interventions']:
            if iv['intervention_type'] == 'conditional':
                number_of_conditional_events += 1
    logger.debug(
        f"Number of conditional events: {number_of_conditional_events}")

    # Compute the maximum queue length ever observed
    logger.info(
        f'Maximum queue length for Queue observed in data: {df["QueueLength"].max()}'
    )
    max_ql = query['maximum_ql']
    unique_l = list(range(max_ql + 1))
    query_dist = {k: 0 for k in unique_l}

    runs = df['Run'].unique()
    for run in runs:
        subdf = df.loc[df['Run'] == run]
        # Find the number of 'Conditional' events in the subdf
        cond_events_in_curr_run = subdf[subdf['Event'] ==
                                        'Conditional'].shape[0]
        if cond_events_in_curr_run == number_of_conditional_events:
            # Extract the query variable value
            query_variable_values = subdf[
                subdf['Event'] == 'Simulation End'][query_variable].values
            if query_variable_values[0] in query_dist:
                query_dist[query_variable_values[0]] += 1
            else:
                logger.debug(f'Query variable value {query_variable_values[0]} '
                             'not in the unique list of states')

    # Compute the total number of samples that is used to compute the probability distribution
    sum_total = sum(list(query_dist.values()))

    # Debug in case of division by zero
    logger.debug(f'Distribution of query variable: {query_dist}')
    logger.debug(f"Sum of total: {sum_total}")
    if sum_total == 0:
        sum_total = 1e-10  # Avoid division by zero

    # Compute the CI and prob dist using the Bonferroni correction
    conf_intervals = {k: 0 for k in unique_l}

    # 95% CI with Bonferroni inequality
    g = len(unique_l)
    # conf_pct = (1 - (0.05 / g)) * 100       # Bonferroni correction
    conf_pct = (1 - 0.05) * 100             # No Bonferroni correction
    z = norm.ppf(0.5 + (conf_pct / 200))

    for k, v in query_dist.items():
        p = v / sum_total
        hw = math.sqrt(p * (1 - p) / query['gt_replications']) * z
        query_dist[k] = p
        conf_intervals[k] = hw

    # We choose another method to compute the confidence intervals for the
    # empirical CDF instead of the PMF; we do this using Clopper-Pearson exact intervals
    alpha = 0.05
    empirical_cdf = {k: 0 for k in unique_l}
    cdf_pointwise_bounds = {k: (0, 0) for k in unique_l}
    for k, v in query_dist.items():
        empirical_cdf[k] = sum([query_dist[j] for j in range(k + 1)])
        binom_k = empirical_cdf[k]
        binom_n = sum_total
        # Use the Clopper-Pearson exact intervals to get p_l, p_u
        p_l, p_u = beta.ppf([alpha / 2, 1 - alpha / 2], [binom_k, binom_k + 1],
                            [binom_n - binom_k + 1, binom_n - binom_k])
        cdf_pointwise_bounds[k] = (p_l, p_u)

    logger.info(f"Query Distribution: {query_dist}")
    logger.info(f"CI Half width: {conf_intervals}")
    logger.info(f"Empirical CDF: {empirical_cdf}")
    logger.info(f"CDF Pointwise Bounds: {cdf_pointwise_bounds}")
    gt = {
        'query_dist': query_dist,
        'half_width': conf_intervals,
        'empirical_cdf': empirical_cdf,
        'cdf_pointwise_bounds': cdf_pointwise_bounds
    }
    return gt


if __name__ == '__main__':
    """Compare ground truth from Markovian queue and inferred DBN.
    
    Function call: 
    python src/compute_montecarlo_gt.py  
    --config_file configs/queries.yaml
    --experiment_number 1 -v
    """     # pylint: disable=pointless-string-statement
    parser = argparse.ArgumentParser(
        description="Compare ground truth to inferred DBN")
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="Path to the configuration file (e.g. configs/queries.yaml)",
        default="config/queries.yaml")
    parser.add_argument("--experiment_number",
                        "-e",
                        type=int,
                        help="Experiment number (e.g. 1)",
                        default=1)
    parser.add_argument('--verbose',
                        '-v',
                        help='Increase output verbosity',
                        action='store_true')

    args = parser.parse_args()
    config_file = args.config_file
    experiment_number = args.experiment_number

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load the evidence file
    with open(config_file, 'r', encoding='utf-8') as file:
        query_data = yaml.safe_load(file)
    query_details = query_data[f'experiment_{experiment_number}']
    #query_workload_name = f"{config_file.split('/')[-1].split('.')[0]}"
    #gt_folder = f"{query_details['gt_results_folder']}/{query_workload_name}"
    gt_folder = f"{query_details['gt_results_folder']}/{query_details['expt_name']}"

    # Compute the ground truth probability distribution
    for suffix in ["-gamma", "-hypoexp"]:

        logger.info(f"Processing GT file with suffix {suffix}")

        gt_dict = compute_gt_pd(
            query_details,
            experiment_number,
            gt_folder=gt_folder,
            suffix=suffix
        )

        gt_pd_filepath = f"{gt_folder}/gt-exp-{experiment_number}{suffix}.pkl"

        with open(gt_pd_filepath, 'wb') as f:
            pickle.dump(gt_dict, f)
