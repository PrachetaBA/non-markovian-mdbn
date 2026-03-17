# pylint: disable=line-too-long, logging-fstring-interpolation
"""Utility functions for the workspace."""
import logging
import pickle
import pandas as pd
from scipy.spatial import distance
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

logger = logging.getLogger('utils')


def max_ql_subsampled_data(file_2tbn):
    """Computes the maximum queue length from sub-sampled data."""
    df = pd.read_csv(file_2tbn)
    # Find the maximum value from the following columns: L_qa_tprev, L_qa, L_qb_tprev, L_qb, L_qc_tprev, L_qc
    max_ql = df[[
        'L_qa_tprev', 'L_qa', 'L_qb_tprev', 'L_qb', 'L_qc_tprev', 'L_qc'
    ]].max().max()
    print(f'Maximum QL value: {max_ql}')


def get_latex_string(key):
    if key == 'Lambdaqa':
        prkey = r'$\lambda^{A}$'
    elif key == 'Lambdaqb':
        prkey = r'$\lambda^{B}$'
    elif key == 'Lambdaqc':
        prkey = r'$\lambda^{C}$'
    elif key == 'Muqa':
        prkey = r'$\mu^{A}$'
    elif key == 'Muqb':
        prkey = r'$\mu^{B}$'
    elif key == 'Muqc':
        prkey = r'$\mu^{C}$'
    elif key == 'Rbc':
        prkey = r'$r^{BC}$'
    elif key == 'Rca':
        prkey = r'$r^{CA}$'
    elif key == 'Lqa':
        prkey = r'$L^{A}_{0.0}$'
    elif key == 'Lqb':
        prkey = r'$L^{B}_{0.0}$'
    elif key == 'Lqc':
        prkey = r'$L^{C}_{0.0}$'
    return prkey


def get_plot_title(query, delta, max_iql):
    """Constructs the title of the plot from the query."""
    title = 'Start params: '
    for key, value in query['start_parameters'].items():
        # prkey = get_latex_string(key)
        title += f'{key} = {value}, '
    title += '\nEvidence: '
    if query['interventions']:
        for iv in query['interventions']:
            iv_var = iv['intervention_variable']
            if iv['intervention_type'] == 'conditional':
                title += f"{iv_var}_{iv['intervention_start']} = {iv['intervention_value']}, "
            elif iv['intervention_type'] == 'interventional' or iv[
                    'intervention_type'] == 'parameter_intervention':
                title += f"{iv_var}_{iv['intervention_start']} "
                title += r'$\leftarrow$'
                title += f"{iv['intervention_value']}, "
            elif iv['intervention_type'] == 'additive':
                title += f"{iv_var}_{iv['intervention_start']} "
                title += r'$\leftarrow$'
                title += f"{iv['intervention_variable']}_{iv['intervention_start']} + {iv['intervention_value']}, "
            elif iv['intervention_type'] == 'subtractive':
                title += f"{iv_var}_{iv['intervention_start']} "
                title += r'$\leftarrow$'
                title += f"{iv_var}_{iv['intervention_start']} - {iv['intervention_value']}, "
    title += f"\nQuery: P({query['query_variable']}_{query['query_time']})"
    title += f"\nDelta: {delta}, IQL_Max: {max_iql}, GT reps: {query['gt_replications']}"
    return title


def extract_best_structure(config_file):
    """Computes the average JSD for each type of structure."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    jsds = {}

    for exp_num in range(1, 71):
        query_details = config['experiment_' + str(exp_num)]
        structure = query_details['dbn_edges']

        # Get ground truth probability distribution
        with open(f"{query_details['gt_results_folder']}/gt-exp-{exp_num}.pkl",
                  'rb') as f:
            gt, _ = pickle.load(f)
        logger.debug(f'Ground truth probability distribution: {gt}')

        dbn_output_filename = f"{query_details['results_folder']}/posterior-exp-{exp_num}.pkl"
        logger.debug(f'DBN output filename: {dbn_output_filename}')

        # Extract the inferred probability distribution
        inferred_pd = pd.read_pickle(dbn_output_filename)
        logger.debug(f'Inferred probability distribution: {inferred_pd}')

        # Compute the Jensen-Shannon divergence
        jsd = distance.jensenshannon(list(gt.values()),
                                     list(inferred_pd.values()))
        jsds[exp_num] = (jsd, ':'.join(structure))

    # Print the JSD for each experiment, after every 6 experiments
    # print a ###### separator
    for exp_num, (jsd, structure) in jsds.items():
        logger.info(
            f'Experiment {exp_num}: JSD = {jsd:.3f}, Structure = {structure}')
        if exp_num % 6 == 0 and exp_num < 61:
            logger.info('#' * 50)

    # Compute the average JSD for each type of structure
    avg_jsds = {}
    for exp_num, (jsd, structure) in jsds.items():
        if structure not in avg_jsds:
            avg_jsds[structure] = []
        avg_jsds[structure].append(jsd)

    for structure, jsds in avg_jsds.items():
        avg_jsds[structure] = round(sum(jsds) / len(jsds), 4)

    logger.info(f'Average JSD for each type of structure: {avg_jsds}')


def compute_avg_jsd(config_file, start_exp_num, end_exp_num, verbose=False):
    """Computes the average JSD for different interventions."""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    jsds = {}

    for exp_num in range(start_exp_num, end_exp_num + 1):
        query_details = config['experiment_' + str(exp_num)]

        # Get ground truth probability distribution
        with open(f"{query_details['gt_results_folder']}/gt-exp-{exp_num}.pkl",
                  'rb') as f:
            gt, _ = pickle.load(f)
        logger.debug(f'Ground truth probability distribution: {gt}')

        dbn_output_filename = f"{query_details['results_folder']}/posterior-exp-{exp_num}.pkl"
        logger.debug(f'DBN output filename: {dbn_output_filename}')

        # Extract the inferred probability distribution
        inferred_pd = pd.read_pickle(dbn_output_filename)
        logger.debug(f'Inferred probability distribution: {inferred_pd}')

        # Compute the Jensen-Shannon divergence
        jsd = distance.jensenshannon(list(gt.values()),
                                     list(inferred_pd.values()))
        jsds[exp_num] = jsd

    # Print the JSD for each experiment, after every 4 experiments compute the average
    # and print a ###### separator
    avg_jsd = 0
    for exp_num, jsd in jsds.items():
        if verbose:
            logger.info(f'Experiment {exp_num}: JSD = {jsd:.3f}')
        avg_jsd += jsd
        if exp_num % 4 == 0:
            avg_jsd = round(avg_jsd / 4, 4)
            logger.info(
                f'Average JSD for experiments {exp_num-3} to {exp_num}: {avg_jsd}'
            )
            avg_jsd = 0
            if verbose:
                logger.info('#' * 50)


def compute_jsd_detailed(config_file, gt_nreps=30):
    """Detailed analysis of the JSD for different interventions."""

    # Load config file (JSON or YAML)
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Attempt to infer the workload name (used for locating the posterior files)
    workload_name = os.path.basename(config_file).split('.')[0]

    jsd_analysis = pd.DataFrame(columns=[
        'exp_num', 'expt_name', 'intv_type', 'intv_var', 'intv_time',
        'intv_val', 'query_var', 'query_time', 'queue_setting', 'jsd',
        'num_conditional_events', 'inference_time', 'inference_compute_time'
    ])

    # Counters for reporting skipped experiments and reasons
    skipped_counters = Counter()
    total_count = 0

    for experiment in config:
        total_count += 1
        query_details = config[experiment]
        exp_num = int(experiment.split('_')[-1])
        logger.info(f'Experiment {exp_num}')

        # possible ground truth locations (prefer gt_results_folder/expt_name/gt-exp-{exp_num}.pkl)
        primary_gt_path = os.path.join(query_details['gt_results_folder'], query_details['expt_name'], f'gt-exp-{exp_num}.pkl')
        pooled_gt_path = os.path.join(query_details['gt_results_folder'], f'queries_nreps-{gt_nreps}_pooled', f'gt-exp-{exp_num}.pkl')

        # Try to load GT from the primary location, then fallback to pooled naming
        gt_dict = None
        if os.path.exists(primary_gt_path):
            gt_path_used = primary_gt_path
        elif os.path.exists(pooled_gt_path):
            gt_path_used = pooled_gt_path
        else:
            logger.warning(f'Experiment {exp_num} has no ground truth file at {primary_gt_path} or {pooled_gt_path}. Skipping...')
            skipped_counters['missing_gt'] += 1
            continue

        try:
            with open(gt_path_used, 'rb') as f:
                gt_dict = pickle.load(f)
            logger.debug(f'Ground truth probability distribution: {gt_dict.get("query_dist", None)}')
        except Exception as e:
            logger.warning(f'Could not load GT for experiment {exp_num} from {gt_path_used}: {e}. Skipping...')
            skipped_counters['bad_gt'] += 1
            continue

        # For small exp numbers, try to compute number of conditional events from CSV if available
        if exp_num <= 50:
            try:
                gt_csv = pd.read_csv(os.path.join(os.path.dirname(gt_path_used), f'gt-exp-{exp_num}.csv'))
                num_cond_events = len(gt_csv[gt_csv['Event'] == 'Conditional'])
            except FileNotFoundError:
                num_cond_events = None
                logger.warning(f'Experiment {exp_num} has no ground truth CSV file. Continuing without num_cond_events.')
        else:
            num_cond_events = None

        # Compose posterior filename path:
        # results_folder / workload_name / posterior-exp-{exp_num}.pkl
        results_folder_for_query = query_details['results_folder']
        posterior_path = os.path.join(results_folder_for_query, workload_name, f'posterior-exp-{exp_num}.pkl')

        if not os.path.exists(posterior_path):
            logger.warning(f'Experiment {exp_num} has no posterior file at {posterior_path}. Skipping...')
            skipped_counters['missing_posterior'] += 1
            continue

        # Load posterior
        try:
            inferred_data = pd.read_pickle(posterior_path)
            # expected structure based on compare_queries.py
            inferred_pd = inferred_data.get('Posterior') if isinstance(inferred_data, dict) else inferred_data
            inference_time = inferred_data.get('InferenceTime') if isinstance(inferred_data, dict) else None
            inference_compute_time = inferred_data.get('FullInferenceTime') if isinstance(inferred_data, dict) else None
        except Exception as e:
            logger.warning(f'Could not load posterior for experiment {exp_num} from {posterior_path}: {e}. Skipping...')
            skipped_counters['bad_posterior'] += 1
            continue

        # Validate GT and posterior dictionaries
        gt = gt_dict.get('query_dist') if isinstance(gt_dict, dict) else None
        if gt is None:
            logger.warning(f'Experiment {exp_num} GT does not contain "query_dist". Skipping...')
            skipped_counters['bad_gt_format'] += 1
            continue

        # If GT sums to zero, skip but count it
        if sum(gt.values()) == 0:
            logger.warning(f'Experiment {exp_num} has zero probability in ground truth! Skipping...')
            skipped_counters['zero_gt'] += 1
            continue

        # Ensure inferred_pd has same support ordering as gt: create lists aligned by state keys
        try:
            # If inferred_pd is a pandas Series or dict-like:
            if isinstance(inferred_pd, pd.Series):
                inferred_series = inferred_pd
            else:
                inferred_series = pd.Series(inferred_pd)

            # Align keys by sorted integer state (fallback to list order if keys are not numeric)
            try:
                states = sorted(map(int, gt.keys()))
            except Exception:
                states = list(gt.keys())

            posterior_probs = [float(inferred_series.get(s, 0.0)) for s in states]
            gt_probs = [float(gt.get(s, 0.0)) for s in states]
        except Exception as e:
            logger.warning(f'Error aligning distributions for experiment {exp_num}: {e}. Skipping...')
            skipped_counters['alignment_error'] += 1
            continue

        # Compute JSD and handle NaN
        jsd = None
        try:
            jsd = distance.jensenshannon(gt_probs, posterior_probs)
            if np.isnan(jsd):
                logger.warning(f'Experiment {exp_num} JSD is NaN. Skipping...')
                skipped_counters['nan_jsd'] += 1
                continue
        except Exception as e:
            logger.warning(f'Error computing JSD for experiment {exp_num}: {e}. Skipping...')
            skipped_counters['jsd_error'] += 1
            continue

        # Extract intervention metadata
        try:
            intv = query_details.get('interventions', [{}])[0]
            intv_type = intv.get('intervention_type', 'unknown')
            intv_var = intv.get('intervention_variable')
            intv_time = intv.get('intervention_start')
            intv_val = intv.get('intervention_value')
        except Exception:
            intv_type = 'unknown'
            intv_var = None
            intv_time = None
            intv_val = None

        # Populate the DataFrame (same columns as before)
        jsd_analysis.loc[len(jsd_analysis)] = [
            exp_num, query_details.get('expt_name'),
            intv_type, intv_var, intv_time,
            intv_val, query_details.get('query_variable'), query_details.get('query_time'),
            query_details.get('expt_name').split('_')[-1] if query_details.get('expt_name') else None,
            jsd, num_cond_events, inference_time, inference_compute_time
        ]

    # Merge additive and subtractive interventions
    jsd_analysis['intv_type'] = jsd_analysis['intv_type'].replace({
        'additive': 'add_sub',
        'subtractive': 'add_sub'
    })

    # Summary stats
    summary = {}
    for intv_type in jsd_analysis['intv_type'].unique():
        vals = jsd_analysis[jsd_analysis['intv_type'] == intv_type]['jsd'].dropna()
        summary[intv_type] = {
            'count': int(vals.shape[0]),
            'mean': float(vals.mean()) if not vals.empty else None,
            'median': float(vals.median()) if not vals.empty else None,
            'std': float(vals.std()) if not vals.empty else None,
        }

    logger.info(f'JSD summary by intervention type: {summary}')
    logger.info(f'Skipped counts: {dict(skipped_counters)}')
    logger.info(f'Total experiments processed: {total_count}, valid: {len(jsd_analysis)}, skipped: {sum(skipped_counters.values())}')

    # Compute the average JSD for each type of intervention (as before)
    avg_jsds = {}
    for intv_type in jsd_analysis['intv_type'].unique():
        avg_jsds[intv_type] = jsd_analysis[jsd_analysis['intv_type'] == intv_type]['jsd'].mean()

    print(avg_jsds)

    # Save the DataFrame to a CSV file (same naming convention as before)
    results_csv_dir = 'results'
    os.makedirs(results_csv_dir, exist_ok=True)
    csv_fname = f"{results_csv_dir}/{workload_name}_results.csv"
    jsd_analysis.to_csv(csv_fname, index=False)
    logger.info(f'Saved detailed JSD CSV to {csv_fname}')

    print(jsd_analysis)

    # ---- Produce a box plot of JSD grouped by intervention type ----
    if not jsd_analysis.empty:
        plt.figure(figsize=(8, 6))
        # Prepare data for boxplot in a consistent order
        intv_types = sorted(jsd_analysis['intv_type'].unique())
        data_to_plot = [jsd_analysis[jsd_analysis['intv_type'] == t]['jsd'].dropna().values for t in intv_types]

        # Create the boxplot
        plt.boxplot(data_to_plot, labels=intv_types, showfliers=True)
        plt.xlabel('Intervention Type')
        plt.ylabel('Jensen-Shannon Distance (PMF)')
        plt.title(f'JSD distribution by intervention type ({workload_name})')
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        # Save plot to results folder
        boxplot_fname = os.path.join(results_csv_dir, f"{workload_name}_jsd_boxplot.png")
        plt.savefig(boxplot_fname, bbox_inches='tight', dpi=150)
        logger.info(f'Saved JSD boxplot to {boxplot_fname}')

        # Also try to save to the figures folder of the first query if available
        try:
            first_query = next(iter(config.values()))
            figures_folder = first_query.get('figures_folder')
            if figures_folder:
                os.makedirs(figures_folder, exist_ok=True)
                fig_save_path = os.path.join(figures_folder, workload_name + '_jsd_boxplot.png')
                plt.savefig(fig_save_path, bbox_inches='tight', dpi=150)
                logger.info(f'Saved JSD boxplot to {fig_save_path}')
        except Exception:
            pass

    else:
        logger.warning('No valid JSDs to plot; boxplot skipped.')


def determine_stability(config_file, experiment_number):
    """For all combination of arrival rates, service rates and routing probabilities,
    determine the overall arrival rate for each of the three queues. This should be
    less than the service rates that have been specified by the user. If not, raise
    an error for the combination that does not guarantee stability."""

    # Read the configuration file and extract the parameters
    with open(config_file, 'r', encoding='utf-8') as file:
        all_configs = yaml.safe_load(file)
    config = all_configs[f'experiment_{experiment_number}']

    mean_interarrival_rates_queue_a = config['queue_a_arrival_rates']
    mean_service_rates_queue_a = config['queue_a_service_rates']
    mean_interarrival_rates_queue_b = config['queue_b_arrival_rates']
    mean_service_rates_queue_b = config['queue_b_service_rates']
    mean_interarrival_rates_queue_c = config['queue_c_arrival_rates']
    mean_service_rates_queue_c = config['queue_c_service_rates']
    rp_a_to_b = config['routing_probabilities']['a_to_b']
    rp_b_to_c = config['routing_probabilities']['b_to_c']
    rp_c_to_a = config['routing_probabilities']['c_to_a']

    min_ser_a = min(mean_service_rates_queue_a)
    min_ser_b = min(mean_service_rates_queue_b)
    min_ser_c = min(mean_service_rates_queue_c)

    for lambda_a in mean_interarrival_rates_queue_a:
        for lambda_b in mean_interarrival_rates_queue_b:
            for lambda_c in mean_interarrival_rates_queue_c:
                for r_ab in rp_a_to_b:
                    for r_bc in rp_b_to_c:
                        for r_ca in rp_c_to_a:
                            # Compute the overall arrival rate for each queue
                            overall_lambda_a = lambda_a + r_ca * lambda_c
                            overall_lambda_b = lambda_b + r_ab * lambda_a
                            overall_lambda_c = lambda_c + r_bc * lambda_b
                            # Check if the overall arrival rate is less than the service rate
                            if overall_lambda_a > min_ser_a or \
                                overall_lambda_b > min_ser_b or \
                                overall_lambda_c > min_ser_c:
                                print(
                                    f'Combination: {lambda_a}, {lambda_b}, {lambda_c}, {r_ab}, {r_bc}, {r_ca} is not stable!'
                                )
                                print(
                                    f'Overall arrival rates: {overall_lambda_a}, {overall_lambda_b}, {overall_lambda_c}'
                                )
                                print(
                                    f'Service rates: {min_ser_a}, {min_ser_b}, {min_ser_c}'
                                )
                                print('Combination not stable!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Compute the detailed JSD for each query in the config file
    INDICATOR = False
    INFERENCE_METHOD = 'exact-lazyprop'
    NREPS = 30
    compute_jsd_detailed(
        f'config/query_workload_exp-6.json',
        gt_nreps=30)
    #compute_jsd_detailed(
    #    f'configs/indicator-{INDICATOR}_inference-{INFERENCE_METHOD}_nreps-{NREPS}_pooled-False.json',
    #    gt_nreps=30)
    # compute_ci_points('configs/indicator-True_inference-approx-is_nreps-30.json')

    # Compute the maximum observed queue length in the subsampled data
    # exp_num = 12
    # max_ql_subsampled_data(f'data/discrete_time/discrete-time-2tbn-exp-{exp_num}.csv')

    # Determine the stability of the system
    # determine_stability('configs/simulator.yaml', 1)
