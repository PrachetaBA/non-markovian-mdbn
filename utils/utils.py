# pylint: disable=line-too-long, logging-fstring-interpolation
"""Utility functions for the workspace."""
import logging
import pickle
import pandas as pd
from scipy.spatial import distance
import yaml

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
    title += f"\nStructure: {','.join(query['dbn_edges'])}"
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
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    jsd_analysis = pd.DataFrame(columns=[
        'exp_num', 'expt_name', 'intv_type', 'intv_var', 'intv_time',
        'intv_val', 'query_var', 'query_time', 'queue_setting', 'jsd',
        'num_conditional_events', 'inference_time', 'inference_compute_time'
    ])

    for experiment in config:
        query_details = config[experiment]
        exp_num = int(experiment.split('_')[-1])
        logger.info(f'Experiment {exp_num}')

        # gt_results_folder = f"{query_details['gt_results_folder']}/{query_details['expt_name']}"
        # gt_results_folder = f"{query_details['gt_results_folder']}/queries_nreps-{gt_nreps}"
        gt_results_folder = f"{query_details['gt_results_folder']}/queries_nreps-{gt_nreps}_pooled"
        # Get ground truth probability distribution
        try:
            with open(f'{gt_results_folder}/gt-exp-{exp_num}.pkl', 'rb') as f:
                gt_dict = pickle.load(f)
            logger.debug(
                f'Ground truth probability distribution: {gt_dict["query_dist"]}'
            )
        except FileNotFoundError:
            logger.warning(
                f'Experiment {exp_num} has no ground truth file. Skipping...')
            continue

        # Compute the number of conditional events only for experiments < 50
        if exp_num <= 50:
            try:
                gt_csv = pd.read_csv(
                    f'{gt_results_folder}/gt-exp-{exp_num}.csv')
                # num_cond_events is the number of times "Conditional" appears in the "Event" column
                num_cond_events = len(gt_csv[gt_csv['Event'] == 'Conditional'])
            except FileNotFoundError:
                num_cond_events = None
                logger.warning(
                    f'Experiment {exp_num} has no ground truth CSV file. Skipping...'
                )
                continue
        else:
            num_cond_events = None

        results_folder = f"{query_details['results_folder']}/{query_details['expt_name']}"
        dbn_output_filename = f'{results_folder}/posterior-exp-{exp_num}.pkl'
        logger.debug(f'DBN output filename: {dbn_output_filename}')

        # If the posterior file does not exist, raise a warning and skip the experiment
        try:
            inferred_data = pd.read_pickle(dbn_output_filename)
            inferred_pd = inferred_data['Posterior']
            inference_time = inferred_data['InferenceTime']
            inference_compute_time = inferred_data['FullInferenceTime']
        except FileNotFoundError:
            # logger.warning(
            #     f'Experiment {exp_num} has no posterior file. Assuming zero probability...'
            # )
            # # Instead of skipping we can have 0 probabilities for all states
            # inferred_pd = {state: 0 for state in gt.keys()}
            # For now we skip the experiment
            logger.warning(
                f'Experiment {exp_num} has no posterior file. Skipping...')
            continue
        logger.debug(f'Inferred probability distribution: {inferred_pd}')

        # Check if sum of gt.values() = 0
        gt = gt_dict['query_dist']  # Ground truth probability distribution
        if sum(gt.values()) == 0:
            logger.warning(
                f'Experiment {exp_num} has zero probability in ground truth!')
        # Compute the Jensen-Shannon divergence
        jsd = distance.jensenshannon(list(inferred_pd.values()),
                                     list(gt.values()))

        # Populate the DataFrame
        jsd_analysis.loc[len(jsd_analysis)] = [
            exp_num, query_details['expt_name'],
            query_details['interventions'][0]['intervention_type'],
            query_details['interventions'][0]['intervention_variable'],
            query_details['interventions'][0]['intervention_start'],
            query_details['interventions'][0]['intervention_value'],
            query_details['query_variable'], query_details['query_time'],
            query_details['expt_name'].split('_')[-1], jsd, num_cond_events,
            inference_time, inference_compute_time
        ]

    # Compute the average JSD for each type of intervention
    avg_jsds = {}
    for intv_type in jsd_analysis['intv_type'].unique():
        avg_jsds[intv_type] = jsd_analysis[jsd_analysis['intv_type'] ==
                                           intv_type]['jsd'].mean()

    print(avg_jsds)

    # Save the DataFrame to a CSV file
    jsd_analysis.to_csv(
        f"results/{config_file.split('/')[1].split('.')[0]}_results.csv",
        index=False)

    print(jsd_analysis)


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
        f'configs/indicator-{INDICATOR}_inference-{INFERENCE_METHOD}_nreps-{NREPS}_pooled-False.json',
        gt_nreps=30)
    # compute_ci_points('configs/indicator-True_inference-approx-is_nreps-30.json')

    # Compute the maximum observed queue length in the subsampled data
    # exp_num = 12
    # max_ql_subsampled_data(f'data/discrete_time/discrete-time-2tbn-exp-{exp_num}.csv')

    # Determine the stability of the system
    # determine_stability('configs/simulator.yaml', 1)
