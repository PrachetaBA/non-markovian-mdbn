# pylint: disable=pointless-string-statement, logging-fstring-interpolation
"""Script to compare probability distributions."""

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
                        default='config/gamma_simulator.yaml',
                        required=False)
    parser.add_argument(
        '--time_disc_config',
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
            f"experiment_{time_discrete_exp['hypoexp_m1_time_series_experiment']}"]
        max_iql = sim_exp['max_iql']

    # Get ground truth probability distribution
    query_workload_name = f"{config_file.split('/')[-1].split('.')[0]}"
    with open(f"data/queries_gt/{query_workload_name}/gt-exp-{experiment_number}.pkl", 'rb') as f:
        gt_dict = pickle.load(f)
    gt = gt_dict['query_dist']
    ci = gt_dict['half_width']
    logger.info(f'Ground truth probability distribution: {gt}')
    logger.info(f'Confidence interval: {ci}')
 
    results_folder = f"{query_details['results_folder']}/{query_workload_name}"
    dbn_output_filename = f"{results_folder}/posterior-exp-{experiment_number}.pkl"
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
    jsd_value = distance.jensenshannon(list(gt.values()),
                                       list(inferred_pd.values()))
    logger.info(f'Jensen-Shannon distance: {jsd_value}')

    # Plot the ground truth and inferred probability distributions

    # Get title for the plot from the query details
    plot_title = get_plot_title(query_details,
                                sampling_interval,
                                max_iql=max_iql)
    logger.info(f'Plot title: {plot_title}')
    # Put scatter points on the plot
    plt.scatter(list(gt.keys()), list(gt.values()), s=10, color='r', alpha=0.5)
    plt.scatter(list(inferred_pd.keys()),
                list(inferred_pd.values()),
                s=10,
                color='b',
                alpha=0.5)

    # Create a shaded area for the confidence interval
    y1 = list(map(np.subtract, list(gt.values()), list(ci.values())))
    y2 = list(map(np.add, list(gt.values()), list(ci.values())))
    plt.fill_between(list(gt.keys()), y1, y2, color='pink', alpha=0.5)

    # Add error bars to the plot
    # plt.errorbar(list(gt.keys()),
    #              list(gt.values()),
    #              yerr=list(ci.values()),
    #              fmt='o',
    #              label='Ground truth CI',
    #              color='r', alpha=0.5, capsize=5,
    #              markersize=3)  # Change the markersize to adjust the size of the scatter points
    # Add plot lines
    plt.plot(list(gt.keys()),
             list(gt.values()),
             label='Ground truth',
             color='r',
             alpha=0.5,
             markersize=5)
    plt.plot(list(inferred_pd.keys()),
             list(inferred_pd.values()),
             label='Inferred',
             color='b',
             alpha=0.5,
             markersize=5)

    plt.legend()
    plt.xlabel('State')
    plt.ylabel('Probability')
    # Print the JSD in the title rounded to 3 decimal places
    plt.title(f'{plot_title}\nJSD = {jsd_value:.3f}')
    
    # Create the figures folder if it does not exist
    query_workload_filename = f'{config_file.split('/')[-1].split('.')[0]}'
    figures_folder = f"{query_details['figures_folder']}/{query_workload_filename}"
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
    plt.savefig(f'{figures_folder}/fig-exp-{experiment_number}.png',
                bbox_inches='tight',
                dpi=100)

    # Create a plot of the empirical CDF and the Clopper-Pearson confidence interval
    plt.figure(figsize=(10, 6))
    ecdf = gt_dict['empirical_cdf']
    pointwise_bounds = gt_dict['cdf_pointwise_bounds']
    # If any of the values in the pointwise bounds tuple are nan, set them to 0
    pointwise_bounds = {k: (0 if np.isnan(v[0]) else v[0], 0 if np.isnan(v[1]) else v[1])
                        for k, v in pointwise_bounds.items()}
    logger.info(f'Pointwise bounds: {pointwise_bounds}')
    plt.plot(ecdf.keys(), ecdf.values(), label='Empirical CDF', color='r')
    plt.scatter(list(ecdf.keys()), list(ecdf.values()), s=10, color='r', alpha=0.5)
    # Create the lower bound line, by subtracting the lower bound from the ECDF
    pointwise_lower = {k: v - pointwise_bounds[k][0] for k, v in ecdf.items()}
    # Create the upper bound line, by subtracting the upper bound from the ECDF
    pointwise_upper = {k: v + pointwise_bounds[k][1] for k, v in ecdf.items()}
    plt.fill_between(list(ecdf.keys()),
                     list(pointwise_lower.values()),
                     list(pointwise_upper.values()),
                     color='pink',
                     alpha=0.8)
    # Create the inferred CDF from the inferred_pd distribution
    inferred_cdf = {k: sum([inferred_pd[i] for i in range(k + 1)])
                    for k in range(len(inferred_pd))}
    
    # Compute the JSD between the inferred CDF and the empirical CDF
    jsd_cdf = distance.jensenshannon(list(ecdf.values()), list(inferred_cdf.values()))
    logger.info(f'Jensen-Shannon distance between inferred CDF and empirical CDF: {jsd_cdf}')
    # Plot the inferred CDF
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
    plt.savefig(f'{figures_folder}/cdf-fig-exp-{experiment_number}.png',
                bbox_inches='tight',
                dpi=100)

   
