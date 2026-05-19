# pylint: disable=pointless-string-statement, logging-fstring-interpolation
"""Script to compare probability distributions (Beta version)."""

import argparse
import os
import logging
import pickle
import yaml

import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
from utils import get_plot_title

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """Plot and save figure.
    
    Function call: 
    python utils/compare_queries.py  
    --config_file configs/queries.yaml
    --experiment_number 1 -v
    """
    parser = argparse.ArgumentParser(
        description="Compare ground truth to inferred DBN")
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        help="Path to the configuration file (e.g. configs/queries.yaml)",
        default="config/query_workload_exp-1.json")
    parser.add_argument("--experiment_number",
                        "-e",
                        type=int,
                        help="Experiment number (e.g. 1)",
                        default=1)
    parser.add_argument('--verbose',
                        '-v',
                        help='Increase output verbosity',
                        action='store_true')
    parser.add_argument('--sim_config',
                        '-s',
                        type=str,
                        help='Path to the simulation configuration file',
                        default='config/beta_simulator.yaml',
                        required=False)
    parser.add_argument(
        '--time_disc_config',
        '-t',
        type=str,
        help='Path to the time discretization configuration file',
        default='config/beta_hypoexp_time_discretization.yaml',
        required=False)
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

    with open(args.time_disc_config, 'r', encoding='utf-8') as tfile:
        time_data = yaml.safe_load(tfile)
        time_discrete_exp = time_data[
            f"experiment_{query_details['time_discretization_experiment']}"]
        sampling_interval = time_discrete_exp['sampling_interval']
    logger.info(f'Sampling interval: {sampling_interval}')

    with open(args.sim_config, 'r', encoding='utf-8') as sfile:
        sim_data = yaml.safe_load(sfile)
        sim_exp = sim_data[
            f"experiment_{time_discrete_exp['beta_hypoexp_m1_time_series_experiment']}"]
        max_iql = sim_exp['max_iql']

    # Get ground truth probability distribution
    query_workload_name = f"{config_file.split('/')[-1].split('.')[0]}"

    # ---- Load Beta GT ----
    with open(f"data/queries_gt/{query_details['expt_name']}/gt-exp-{experiment_number}-beta.pkl", 'rb') as f:
        gt_beta_dict = pickle.load(f)

    gt_beta = gt_beta_dict['query_dist']
    ci_beta = gt_beta_dict['half_width']

    # ---- Load HypoExp GT ----
    with open(f"data/queries_gt/{query_details['expt_name']}/gt-exp-{experiment_number}-hypoexp.pkl", 'rb') as f:
        gt_hypo_dict = pickle.load(f)

    gt_hypo = gt_hypo_dict['query_dist']
    ci_hypo = gt_hypo_dict['half_width']

    logger.info(f'Beta GT distribution: {gt_beta}')
    logger.info(f'HypoExp GT distribution: {gt_hypo}')
 
    results_folder = f"{query_details['results_folder']}/{query_workload_name}"
    dbn_output_filename = f"{results_folder}/beta-posterior-exp-{experiment_number}.pkl"
    logger.debug(f'DBN output filename: {dbn_output_filename}')

    # Extract the inferred probability distribution
    inferred_pd = pd.read_pickle(dbn_output_filename)['Posterior']
    logger.info(f'Inferred probability distribution: {inferred_pd}')

    # Print the time taken to run inference
    inference_time = pd.read_pickle(dbn_output_filename)['InferenceTime']
    num_slices = pd.read_pickle(dbn_output_filename)['TotalSlices']
    logger.info(f'Inference time: {inference_time} seconds')
    logger.info(f'Number of slices: {num_slices}')

    # Compare the ground truth and inferred probability distributions using JS distance
    jsd_dbn_beta = distance.jensenshannon(
        list(gt_beta.values()),
        list(inferred_pd.values())
    )

    jsd_dbn_hypo = distance.jensenshannon(
        list(gt_hypo.values()),
        list(inferred_pd.values())
    )

    jsd_beta_hypo = distance.jensenshannon(
        list(gt_beta.values()),
        list(gt_hypo.values())
    )

    logger.info(f'JSD(DBN , Beta) = {jsd_dbn_beta}')
    logger.info(f'JSD(DBN , HypoExp) = {jsd_dbn_hypo}')
    logger.info(f'JSD(Beta , HypoExp) = {jsd_beta_hypo}')

    # Plot the ground truth and inferred probability distributions

    plot_title = get_plot_title(query_details,
                                sampling_interval,
                                max_iql=max_iql)
    logger.info(f'Plot title: {plot_title}')

    plt.scatter(list(gt_beta.keys()), list(gt_beta.values()),
                s=10, color='r', alpha=0.5)

    plt.scatter(list(gt_hypo.keys()), list(gt_hypo.values()),
                s=10, color='g', alpha=0.5)

    plt.scatter(list(inferred_pd.keys()), list(inferred_pd.values()),
                s=10, color='b', alpha=0.5)

    y1 = list(map(np.subtract, list(gt_beta.values()), list(ci_beta.values())))
    y2 = list(map(np.add, list(gt_beta.values()), list(ci_beta.values())))
    plt.fill_between(list(gt_beta.keys()), y1, y2, color='pink', alpha=0.5)

    plt.plot(list(gt_beta.keys()),
             list(gt_beta.values()),
             label='Beta GT',
             color='r',
             alpha=0.7)

    plt.plot(list(gt_hypo.keys()),
             list(gt_hypo.values()),
             label='HypoExp GT',
             color='g',
             alpha=0.7)

    plt.plot(list(inferred_pd.keys()),
             list(inferred_pd.values()),
             label='DBN',
             color='b',
             alpha=0.7)

    plt.legend()
    plt.xlabel('State')
    plt.ylabel('Probability')

    plt.title(
        f'{plot_title}\n'
        f'JSD(DBN,Beta)={jsd_dbn_beta:.3f}  '
        f'JSD(DBN,Hypo)={jsd_dbn_hypo:.3f}  '
        f'JSD(Beta,Hypo)={jsd_beta_hypo:.3f}'
    )
    
    query_workload_filename = f"{config_file.split('/')[-1].split('.')[0]}"
    figures_folder = f"{query_details['figures_folder']}/{query_workload_filename}"
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
    plt.savefig(f'{figures_folder}/3dist-beta-fig-exp-{experiment_number}.png',
                bbox_inches='tight',
                dpi=100)

    plt.figure(figsize=(10, 6))
    ecdf = gt_beta_dict['empirical_cdf']
    pointwise_bounds = gt_beta_dict['cdf_pointwise_bounds']

    pointwise_bounds = {k: (0 if np.isnan(v[0]) else v[0], 0 if np.isnan(v[1]) else v[1])
                        for k, v in pointwise_bounds.items()}
    logger.info(f'Pointwise bounds: {pointwise_bounds}')

    plt.plot(ecdf.keys(), ecdf.values(), label='Empirical CDF', color='r')
    plt.scatter(list(ecdf.keys()), list(ecdf.values()), s=10, color='r', alpha=0.5)

    pointwise_lower = {k: v - pointwise_bounds[k][0] for k, v in ecdf.items()}
    pointwise_upper = {k: v + pointwise_bounds[k][1] for k, v in ecdf.items()}

    plt.fill_between(list(ecdf.keys()),
                     list(pointwise_lower.values()),
                     list(pointwise_upper.values()),
                     color='pink',
                     alpha=0.8)

    inferred_cdf = {k: sum([inferred_pd[i] for i in range(k + 1)])
                    for k in range(len(inferred_pd))}
    
    jsd_cdf = distance.jensenshannon(list(ecdf.values()), list(inferred_cdf.values()))
    logger.info(f'Jensen-Shannon distance between inferred CDF and empirical CDF: {jsd_cdf}')

    plt.plot(inferred_cdf.keys(), inferred_cdf.values(), label='Inferred CDF', color='b')
    plt.scatter(list(inferred_cdf.keys()),
                list(inferred_cdf.values()),
                s=10,
                color='b',
                alpha=0.5)

    plt.legend()
    plt.xlabel('State')
    plt.ylabel('Empirical CDF')
    plt.title(f'{plot_title}\nEmpirical CDF with Clopper-Pearson CI \n JSD = {jsd_cdf:.3f}')
    plt.savefig(f'{figures_folder}/3dist-cdf-beta-fig-exp-{experiment_number}.png',
                bbox_inches='tight',
                dpi=100)