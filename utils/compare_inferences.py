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
import seaborn as sns

logger = logging.getLogger('utils')


def compute_detailed_weibull_inference_results(config_file, gt_nreps=30):
    """
    Compute detailed moment-based inference diagnostics for the Weibull workload.

    For each experiment in the config, this function:
      1. Loads the Weibull ground-truth distribution and the HypoExp simulator GT.
      2. Loads the MDBN posterior distribution.
      3. Aligns the supports and normalizes all distributions.
      4. Computes mean and variance for:
           - MDBN posterior
           - Weibull GT (analytical/model distribution)
           - HypoExp GT (simulator approximation)
      5. Computes absolute and percentage errors.
      6. Saves a CSV with all per-query statistics.
      7. Generates 8 diagnostic plots and saves them under:
           results/distribution_mean_var_plots/

    Notes:
      - Experiments are skipped if GT/posterior files are missing, distributions are malformed,
        or distributions have zero total probability mass.
      - Percentage errors are set to NaN when the denominator is too small.
      - The logger is expected to be configured by the caller (as in your main block).
    """
    eps = 1e-12
    logger = logging.getLogger('utils')

    logger.info(f'Starting detailed Weibull inference moment analysis using config: {config_file}')

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    workload_name = os.path.basename(config_file).split('.')[0]
    logger.info(f'Inferred workload name: {workload_name}')

    results_df = pd.DataFrame(columns=[
        'exp_num', 'expt_name', 'intv_type', 'intv_var', 'intv_time', 'intv_val',
        'query_var', 'query_time', 'queue_setting',
        'mean_mdbn', 'mean_model', 'mean_sim',
        'var_mdbn', 'var_model', 'var_sim',
        'mean_err_model_abs', 'mean_err_sim_abs',
        'var_err_model_abs', 'var_err_sim_abs',
        'mean_err_model_pct', 'mean_err_sim_pct',
        'var_err_model_pct', 'var_err_sim_pct',
        'num_conditional_events', 'inference_time', 'inference_compute_time'
    ])

    skipped = Counter()
    total = 0

    for experiment in config:
        total += 1
        q = config[experiment]
        exp_num = int(experiment.split('_')[-1])

        logger.info(f'Processing experiment {exp_num}')

        # ---------- Ground truth paths ----------
        primary_gt_weibull = os.path.join(
            q['gt_results_folder'], q['expt_name'],
            f'gt-exp-{exp_num}-weibull.pkl'
        )
        pooled_gt_weibull = os.path.join(
            q['gt_results_folder'],
            f'queries_nreps-{gt_nreps}_pooled',
            f'gt-exp-{exp_num}-weibull.pkl'
        )

        primary_gt_hypo = os.path.join(
            q['gt_results_folder'], q['expt_name'],
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )
        pooled_gt_hypo = os.path.join(
            q['gt_results_folder'],
            f'queries_nreps-{gt_nreps}_pooled',
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )

        weibull_gt_used = primary_gt_weibull if os.path.exists(primary_gt_weibull) else (
            pooled_gt_weibull if os.path.exists(pooled_gt_weibull) else None
        )
        hypo_gt_used = primary_gt_hypo if os.path.exists(primary_gt_hypo) else (
            pooled_gt_hypo if os.path.exists(pooled_gt_hypo) else None
        )

        if weibull_gt_used is None:
            logger.warning(f'Experiment {exp_num}: missing Weibull GT. Skipping.')
            skipped['missing_weibull_gt'] += 1
            continue

        if hypo_gt_used is None:
            logger.warning(f'Experiment {exp_num}: missing HypoExp GT. Skipping.')
            skipped['missing_hypo_gt'] += 1
            continue

        # ---------- Load GT ----------
        try:
            with open(weibull_gt_used, 'rb') as f:
                weibull_gt_dict = pickle.load(f)
            with open(hypo_gt_used, 'rb') as f:
                hypo_gt_dict = pickle.load(f)
        except Exception as e:
            logger.warning(f'Experiment {exp_num}: GT load failed ({e}). Skipping.')
            skipped['bad_gt'] += 1
            continue

        gt_weibull = weibull_gt_dict.get('query_dist') if isinstance(weibull_gt_dict, dict) else None
        gt_hypo = hypo_gt_dict.get('query_dist') if isinstance(hypo_gt_dict, dict) else None

        if gt_weibull is None or gt_hypo is None:
            logger.warning(f'Experiment {exp_num}: GT missing query_dist. Skipping.')
            skipped['bad_gt_format'] += 1
            continue

        if sum(gt_weibull.values()) == 0 or sum(gt_hypo.values()) == 0:
            logger.warning(f'Experiment {exp_num}: GT has zero probability mass. Skipping.')
            skipped['zero_gt'] += 1
            continue

        # ---------- Posterior ----------
        posterior_path = os.path.join(
            q['results_folder'], workload_name,
            f'weibull-posterior-exp-{exp_num}.pkl'
        )

        if not os.path.exists(posterior_path):
            logger.warning(f'Experiment {exp_num}: missing posterior at {posterior_path}. Skipping.')
            skipped['missing_posterior'] += 1
            continue

        try:
            inferred_data = pd.read_pickle(posterior_path)
            inferred_pd = inferred_data.get('Posterior') if isinstance(inferred_data, dict) else inferred_data
            inference_time = inferred_data.get('InferenceTime') if isinstance(inferred_data, dict) else None
            inference_compute_time = inferred_data.get('FullInferenceTime') if isinstance(inferred_data, dict) else None
        except Exception as e:
            logger.warning(f'Experiment {exp_num}: posterior load failed ({e}). Skipping.')
            skipped['bad_posterior'] += 1
            continue

        # ---------- Align distributions ----------
        try:
            inferred_series = inferred_pd if isinstance(inferred_pd, pd.Series) else pd.Series(inferred_pd)

            states = sorted(set(list(gt_weibull.keys()) + list(gt_hypo.keys()) + list(inferred_series.index)))
            states_arr = np.array(states, dtype=float)

            posterior_probs = np.array([float(inferred_series.get(s, 0.0)) for s in states], dtype=float)
            weibull_probs = np.array([float(gt_weibull.get(s, 0.0)) for s in states], dtype=float)
            hypo_probs = np.array([float(gt_hypo.get(s, 0.0)) for s in states], dtype=float)

            if posterior_probs.sum() <= eps:
                logger.warning(f'Experiment {exp_num}: posterior has zero probability mass. Skipping.')
                skipped['zero_posterior'] += 1
                continue

            posterior_probs /= posterior_probs.sum()
            weibull_probs /= weibull_probs.sum()
            hypo_probs /= hypo_probs.sum()

        except Exception as e:
            logger.warning(f'Experiment {exp_num}: alignment error ({e}). Skipping.')
            skipped['alignment_error'] += 1
            continue

        # ---------- Moments ----------
        try:
            mean_mdbn = float(np.sum(states_arr * posterior_probs))
            mean_model = float(np.sum(states_arr * weibull_probs))
            mean_sim = float(np.sum(states_arr * hypo_probs))

            var_mdbn = float(np.sum((states_arr - mean_mdbn) ** 2 * posterior_probs))
            var_model = float(np.sum((states_arr - mean_model) ** 2 * weibull_probs))
            var_sim = float(np.sum((states_arr - mean_sim) ** 2 * hypo_probs))

            logger.debug(
                'Experiment %s moments: mean_mdbn=%.6f, mean_model=%.6f, mean_sim=%.6f, '
                'var_mdbn=%.6f, var_model=%.6f, var_sim=%.6f',
                exp_num, mean_mdbn, mean_model, mean_sim, var_mdbn, var_model, var_sim
            )

        except Exception as e:
            logger.warning(f'Experiment {exp_num}: moment computation error ({e}). Skipping.')
            skipped['moment_error'] += 1
            continue

        # ---------- Errors ----------
        mean_err_model_abs = abs(mean_mdbn - mean_model)
        mean_err_sim_abs = abs(mean_mdbn - mean_sim)
        var_err_model_abs = abs(var_mdbn - var_model)
        var_err_sim_abs = abs(var_mdbn - var_sim)

        mean_err_model_pct = (mean_err_model_abs / mean_model * 100.0) if abs(mean_model) > eps else np.nan
        mean_err_sim_pct = (mean_err_sim_abs / mean_sim * 100.0) if abs(mean_sim) > eps else np.nan
        var_err_model_pct = (var_err_model_abs / var_model * 100.0) if abs(var_model) > eps else np.nan
        var_err_sim_pct = (var_err_sim_abs / var_sim * 100.0) if abs(var_sim) > eps else np.nan

        # ---------- Metadata ----------
        try:
            intv = q.get('interventions', [{}])[0]
            intv_type = intv.get('intervention_type', 'unknown')
            intv_var = intv.get('intervention_variable')
            intv_time = intv.get('intervention_start')
            intv_val = intv.get('intervention_value')
        except Exception:
            intv_type = 'unknown'
            intv_var = None
            intv_time = None
            intv_val = None

        try:
            gt_csv_path = os.path.join(os.path.dirname(weibull_gt_used), f'gt-exp-{exp_num}-weibull.csv')
            gt_csv = pd.read_csv(gt_csv_path)
            num_cond_events = len(gt_csv[gt_csv['Event'] == 'Conditional'])
        except Exception:
            num_cond_events = None
            logger.debug(f'Experiment {exp_num}: could not read conditional event count.')

        queue_setting = q.get('expt_name').split('_')[-1] if q.get('expt_name') else None

        results_df.loc[len(results_df)] = [
            exp_num, q.get('expt_name'), intv_type, intv_var, intv_time, intv_val,
            q.get('query_variable'), q.get('query_time'), queue_setting,
            mean_mdbn, mean_model, mean_sim,
            var_mdbn, var_model, var_sim,
            mean_err_model_abs, mean_err_sim_abs,
            var_err_model_abs, var_err_sim_abs,
            mean_err_model_pct, mean_err_sim_pct,
            var_err_model_pct, var_err_sim_pct,
            num_cond_events, inference_time, inference_compute_time
        ]

        logger.info(
            'Experiment %s done | mean(model)=%.4f, mean(mdbn)=%.4f, mean(sim)=%.4f | '
            'var(model)=%.4f, var(mdbn)=%.4f, var(sim)=%.4f',
            exp_num, mean_model, mean_mdbn, mean_sim, var_model, var_mdbn, var_sim
        )

    # Merge additive/subtractive for grouping
    results_df['intv_type'] = results_df['intv_type'].replace({
        'additive': 'add_sub',
        'subtractive': 'add_sub'
    })

    # ---------- Save CSV ----------
    os.makedirs('results', exist_ok=True)
    csv_file = f'results/{workload_name}_moment_results.csv'
    results_df.to_csv(csv_file, index=False)

    logger.info(f'Saved moment results CSV to {csv_file}')
    logger.info(f'Total experiments processed: {total}')
    logger.info(f'Valid experiments: {len(results_df)}')
    logger.info(f'Skipped counts: {dict(skipped)}')

    # ---------- Plots ----------
    if results_df.empty:
        logger.warning('No valid experiments available; all plots skipped.')
        return results_df

    generate_weibull_16_plots(results_df, workload_name)

    logger.info('Completed detailed Weibull inference moment analysis.')
    return results_df


def generate_weibull_16_plots(results_df, workload_name):
    """
    Generate 16 diagnostic plots comparing MDBN to both Weibull GT and HypoExp GT.
    Saves plots under 'results/distribution_mean_var_plots/'.
    """

    if results_df.empty:
        print("No valid experiments available; plots skipped.")
        return

    plot_dir = 'results/distribution_mean_var_plots'
    os.makedirs(plot_dir, exist_ok=True)

    comparisons = [
        ('model', 'Weibull GT'),
        ('sim', 'HypoExp GT')
    ]

    for suffix, label in comparisons:
        # ---------- Scatter plots ----------
        try:
            # Mean scatter
            plt.figure(figsize=(7, 7))
            x = results_df[f'mean_{suffix}'].dropna()
            y = results_df.loc[x.index, 'mean_mdbn'].dropna()
            common_idx = x.index.intersection(y.index)
            x = results_df.loc[common_idx, f'mean_{suffix}']
            y = results_df.loc[common_idx, 'mean_mdbn']
            plt.scatter(x, y, alpha=0.6)
            mx = max(float(x.max()), float(y.max())) if len(x) else 1.0
            plt.plot([0, mx], [0, mx], 'r--')
            plt.xlabel(f'{label} Mean')
            plt.ylabel('MDBN Mean')
            plt.title(f'Mean Scatter ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_scatter_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Variance scatter
            plt.figure(figsize=(7, 7))
            x = results_df[f'var_{suffix}'].dropna()
            y = results_df.loc[x.index, 'var_mdbn'].dropna()
            common_idx = x.index.intersection(y.index)
            x = results_df.loc[common_idx, f'var_{suffix}']
            y = results_df.loc[common_idx, 'var_mdbn']
            plt.scatter(x, y, alpha=0.6)
            mx = max(float(x.max()), float(y.max())) if len(x) else 1.0
            plt.plot([0, mx], [0, mx], 'r--')
            plt.xlabel(f'{label} Variance')
            plt.ylabel('MDBN Variance')
            plt.title(f'Variance Scatter ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_scatter_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'Scatter plots failed for {label}: {e}')

        # ---------- Histograms ----------
        try:
            # Absolute error
            plt.figure(figsize=(8, 6))
            values = results_df[f'mean_err_{suffix}_abs'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Absolute Mean Error |MDBN - {label}|')
            plt.ylabel('Count')
            plt.title(f'Mean Absolute Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_abs_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            values = results_df[f'var_err_{suffix}_abs'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Absolute Variance Error |MDBN - {label}|')
            plt.ylabel('Count')
            plt.title(f'Variance Absolute Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_abs_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Percentage error
            plt.figure(figsize=(8, 6))
            values = results_df[f'mean_err_{suffix}_pct'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Mean % Error (MDBN vs {label})')
            plt.ylabel('Count')
            plt.title(f'Mean % Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_pct_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            values = results_df[f'var_err_{suffix}_pct'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Variance % Error (MDBN vs {label})')
            plt.ylabel('Count')
            plt.title(f'Variance % Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_pct_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'Histogram plots failed for {label}: {e}')

        # ---------- Boxplots by intervention ----------
        try:
            grouped_mean = []
            grouped_var = []
            labels_intv = []

            for t in sorted(results_df['intv_type'].dropna().unique()):
                mean_vals = results_df.loc[results_df['intv_type'] == t, f'mean_err_{suffix}_abs'].dropna().values
                var_vals = results_df.loc[results_df['intv_type'] == t, f'var_err_{suffix}_abs'].dropna().values
                if len(mean_vals) > 0 and len(var_vals) > 0:
                    grouped_mean.append(mean_vals)
                    grouped_var.append(var_vals)
                    labels_intv.append(t)

            if grouped_mean:
                plt.figure(figsize=(8, 6))
                plt.boxplot(grouped_mean, labels=labels_intv, showfliers=True)
                plt.xlabel('Intervention Type')
                plt.ylabel(f'Absolute Mean Error (MDBN vs {label})')
                plt.title(f'Mean Absolute Error vs Intervention ({workload_name} | MDBN vs {label})')
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_boxplot_{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()

            if grouped_var:
                plt.figure(figsize=(8, 6))
                plt.boxplot(grouped_var, labels=labels_intv, showfliers=True)
                plt.xlabel('Intervention Type')
                plt.ylabel(f'Absolute Variance Error (MDBN vs {label})')
                plt.title(f'Variance Absolute Error vs Intervention ({workload_name} | MDBN vs {label})')
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_boxplot_{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f'Boxplots failed for {label}: {e}')

    print(f'All 16 plots generated in {plot_dir}')



def compute_detailed_beta_inference_results(config_file, gt_nreps=30):
    """
    Compute detailed moment-based inference diagnostics for the Beta workload.

    For each experiment in the config, this function:
      1. Loads the Beta ground-truth distribution and the HypoExp simulator GT.
      2. Loads the MDBN posterior distribution.
      3. Aligns the supports and normalizes all distributions.
      4. Computes mean and variance for:
           - MDBN posterior
           - Beta GT (analytical/model distribution)
           - HypoExp GT (simulator approximation)
      5. Computes absolute and percentage errors.
      6. Saves a CSV with all per-query statistics.
      7. Generates 8 diagnostic plots and saves them under:
           results/distribution_mean_var_plots/

    Notes:
      - Experiments are skipped if GT/posterior files are missing, distributions are malformed,
        or distributions have zero total probability mass.
      - Percentage errors are set to NaN when the denominator is too small.
      - The logger is expected to be configured by the caller.
    """
    import os, yaml, pickle, logging
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter

    eps = 1e-12
    logger = logging.getLogger('utils')

    logger.info(f'Starting detailed Beta inference moment analysis using config: {config_file}')

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    workload_name = os.path.basename(config_file).split('.')[0]
    logger.info(f'Inferred workload name: {workload_name}')

    results_df = pd.DataFrame(columns=[
        'exp_num', 'expt_name', 'intv_type', 'intv_var', 'intv_time', 'intv_val',
        'query_var', 'query_time', 'queue_setting',
        'mean_mdbn', 'mean_model', 'mean_sim',
        'var_mdbn', 'var_model', 'var_sim',
        'mean_err_model_abs', 'mean_err_sim_abs',
        'var_err_model_abs', 'var_err_sim_abs',
        'mean_err_model_pct', 'mean_err_sim_pct',
        'var_err_model_pct', 'var_err_sim_pct',
        'num_conditional_events', 'inference_time', 'inference_compute_time'
    ])

    skipped = Counter()
    total = 0

    for experiment in config:
        total += 1
        q = config[experiment]
        exp_num = int(experiment.split('_')[-1])

        logger.info(f'Processing experiment {exp_num}')

        # ---------- Ground truth paths ----------
        primary_gt_beta = os.path.join(
            q['gt_results_folder'], q['expt_name'],
            f'gt-exp-{exp_num}-beta.pkl'
        )
        pooled_gt_beta = os.path.join(
            q['gt_results_folder'],
            f'queries_nreps-{gt_nreps}_pooled',
            f'gt-exp-{exp_num}-beta.pkl'
        )

        primary_gt_hypo = os.path.join(
            q['gt_results_folder'], q['expt_name'],
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )
        pooled_gt_hypo = os.path.join(
            q['gt_results_folder'],
            f'queries_nreps-{gt_nreps}_pooled',
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )

        beta_gt_used = primary_gt_beta if os.path.exists(primary_gt_beta) else (
            pooled_gt_beta if os.path.exists(pooled_gt_beta) else None
        )
        hypo_gt_used = primary_gt_hypo if os.path.exists(primary_gt_hypo) else (
            pooled_gt_hypo if os.path.exists(pooled_gt_hypo) else None
        )

        if beta_gt_used is None:
            logger.warning(f'Experiment {exp_num}: missing Beta GT. Skipping.')
            skipped['missing_beta_gt'] += 1
            continue

        if hypo_gt_used is None:
            logger.warning(f'Experiment {exp_num}: missing HypoExp GT. Skipping.')
            skipped['missing_hypo_gt'] += 1
            continue

        # ---------- Load GT ----------
        try:
            with open(beta_gt_used, 'rb') as f:
                beta_gt_dict = pickle.load(f)
            with open(hypo_gt_used, 'rb') as f:
                hypo_gt_dict = pickle.load(f)
        except Exception as e:
            logger.warning(f'Experiment {exp_num}: GT load failed ({e}). Skipping.')
            skipped['bad_gt'] += 1
            continue

        gt_beta = beta_gt_dict.get('query_dist') if isinstance(beta_gt_dict, dict) else None
        gt_hypo = hypo_gt_dict.get('query_dist') if isinstance(hypo_gt_dict, dict) else None

        if gt_beta is None or gt_hypo is None:
            logger.warning(f'Experiment {exp_num}: GT missing query_dist. Skipping.')
            skipped['bad_gt_format'] += 1
            continue

        if sum(gt_beta.values()) == 0 or sum(gt_hypo.values()) == 0:
            logger.warning(f'Experiment {exp_num}: GT has zero probability mass. Skipping.')
            skipped['zero_gt'] += 1
            continue

        # ---------- Posterior ----------
        posterior_path = os.path.join(
            q['results_folder'], workload_name,
            f'beta-posterior-exp-{exp_num}.pkl'
        )

        if not os.path.exists(posterior_path):
            logger.warning(f'Experiment {exp_num}: missing posterior at {posterior_path}. Skipping.')
            skipped['missing_posterior'] += 1
            continue

        try:
            inferred_data = pd.read_pickle(posterior_path)
            inferred_pd = inferred_data.get('Posterior') if isinstance(inferred_data, dict) else inferred_data
            inference_time = inferred_data.get('InferenceTime') if isinstance(inferred_data, dict) else None
            inference_compute_time = inferred_data.get('FullInferenceTime') if isinstance(inferred_data, dict) else None
        except Exception as e:
            logger.warning(f'Experiment {exp_num}: posterior load failed ({e}). Skipping.')
            skipped['bad_posterior'] += 1
            continue

        # ---------- Align distributions ----------
        try:
            inferred_series = inferred_pd if isinstance(inferred_pd, pd.Series) else pd.Series(inferred_pd)

            states = sorted(set(list(gt_beta.keys()) + list(gt_hypo.keys()) + list(inferred_series.index)))
            states_arr = np.array(states, dtype=float)

            posterior_probs = np.array([float(inferred_series.get(s, 0.0)) for s in states], dtype=float)
            beta_probs = np.array([float(gt_beta.get(s, 0.0)) for s in states], dtype=float)
            hypo_probs = np.array([float(gt_hypo.get(s, 0.0)) for s in states], dtype=float)

            if posterior_probs.sum() <= eps:
                logger.warning(f'Experiment {exp_num}: posterior has zero probability mass. Skipping.')
                skipped['zero_posterior'] += 1
                continue

            posterior_probs /= posterior_probs.sum()
            beta_probs /= beta_probs.sum()
            hypo_probs /= hypo_probs.sum()

        except Exception as e:
            logger.warning(f'Experiment {exp_num}: alignment error ({e}). Skipping.')
            skipped['alignment_error'] += 1
            continue

        # ---------- Moments ----------
        try:
            mean_mdbn = float(np.sum(states_arr * posterior_probs))
            mean_model = float(np.sum(states_arr * beta_probs))
            mean_sim = float(np.sum(states_arr * hypo_probs))

            var_mdbn = float(np.sum((states_arr - mean_mdbn) ** 2 * posterior_probs))
            var_model = float(np.sum((states_arr - mean_model) ** 2 * beta_probs))
            var_sim = float(np.sum((states_arr - mean_sim) ** 2 * hypo_probs))

            logger.debug(
                'Experiment %s moments: mean_mdbn=%.6f, mean_model=%.6f, mean_sim=%.6f, '
                'var_mdbn=%.6f, var_model=%.6f, var_sim=%.6f',
                exp_num, mean_mdbn, mean_model, mean_sim, var_mdbn, var_model, var_sim
            )

        except Exception as e:
            logger.warning(f'Experiment {exp_num}: moment computation error ({e}). Skipping.')
            skipped['moment_error'] += 1
            continue

        # ---------- Errors ----------
        mean_err_model_abs = abs(mean_mdbn - mean_model)
        mean_err_sim_abs = abs(mean_mdbn - mean_sim)
        var_err_model_abs = abs(var_mdbn - var_model)
        var_err_sim_abs = abs(var_mdbn - var_sim)

        mean_err_model_pct = (mean_err_model_abs / mean_model * 100.0) if abs(mean_model) > eps else np.nan
        mean_err_sim_pct = (mean_err_sim_abs / mean_sim * 100.0) if abs(mean_sim) > eps else np.nan
        var_err_model_pct = (var_err_model_abs / var_model * 100.0) if abs(var_model) > eps else np.nan
        var_err_sim_pct = (var_err_sim_abs / var_sim * 100.0) if abs(var_sim) > eps else np.nan

        # ---------- Metadata ----------
        try:
            intv = q.get('interventions', [{}])[0]
            intv_type = intv.get('intervention_type', 'unknown')
            intv_var = intv.get('intervention_variable')
            intv_time = intv.get('intervention_start')
            intv_val = intv.get('intervention_value')
        except Exception:
            intv_type = 'unknown'
            intv_var = None
            intv_time = None
            intv_val = None

        try:
            gt_csv_path = os.path.join(os.path.dirname(beta_gt_used), f'gt-exp-{exp_num}-beta.csv')
            gt_csv = pd.read_csv(gt_csv_path)
            num_cond_events = len(gt_csv[gt_csv['Event'] == 'Conditional'])
        except Exception:
            num_cond_events = None
            logger.debug(f'Experiment {exp_num}: could not read conditional event count.')

        queue_setting = q.get('expt_name').split('_')[-1] if q.get('expt_name') else None

        results_df.loc[len(results_df)] = [
            exp_num, q.get('expt_name'), intv_type, intv_var, intv_time, intv_val,
            q.get('query_variable'), q.get('query_time'), queue_setting,
            mean_mdbn, mean_model, mean_sim,
            var_mdbn, var_model, var_sim,
            mean_err_model_abs, mean_err_sim_abs,
            var_err_model_abs, var_err_sim_abs,
            mean_err_model_pct, mean_err_sim_pct,
            var_err_model_pct, var_err_sim_pct,
            num_cond_events, inference_time, inference_compute_time
        ]

        logger.info(
            'Experiment %s done | mean(model)=%.4f, mean(mdbn)=%.4f, mean(sim)=%.4f | '
            'var(model)=%.4f, var(mdbn)=%.4f, var(sim)=%.4f',
            exp_num, mean_model, mean_mdbn, mean_sim, var_model, var_mdbn, var_sim
        )

    # Merge additive/subtractive for grouping
    results_df['intv_type'] = results_df['intv_type'].replace({
        'additive': 'add_sub',
        'subtractive': 'add_sub'
    })

    # ---------- Save CSV ----------
    os.makedirs('results', exist_ok=True)
    csv_file = f'results/{workload_name}_moment_results.csv'
    results_df.to_csv(csv_file, index=False)
    logger.info(f'Saved Beta moment results CSV to {csv_file}')
    logger.info(f'Total experiments processed: {total}')
    logger.info(f'Valid experiments: {len(results_df)}')
    logger.info(f'Skipped counts: {dict(skipped)}')

    generate_beta_16_plots(results_df, workload_name)

    logger.info('Completed detailed Beta inference moment analysis.')
    return results_df



def generate_beta_16_plots(results_df, workload_name):
    """
    Generate 16 diagnostic plots comparing MDBN to both Beta GT and HypoExp GT.
    Saves plots under 'results/distribution_mean_var_plots/'.
    """

    if results_df.empty:
        print("No valid experiments available; plots skipped.")
        return

    plot_dir = 'results/distribution_mean_var_plots'
    os.makedirs(plot_dir, exist_ok=True)

    comparisons = [
        ('model', 'Beta GT'),
        ('sim', 'HypoExp GT')
    ]

    for suffix, label in comparisons:
        # ---------- Scatter plots ----------
        try:
            # Mean scatter
            plt.figure(figsize=(7, 7))
            x = results_df[f'mean_{suffix}'].dropna()
            y = results_df.loc[x.index, 'mean_mdbn'].dropna()
            common_idx = x.index.intersection(y.index)
            x = results_df.loc[common_idx, f'mean_{suffix}']
            y = results_df.loc[common_idx, 'mean_mdbn']
            plt.scatter(x, y, alpha=0.6)
            mx = max(float(x.max()), float(y.max())) if len(x) else 1.0
            plt.plot([0, mx], [0, mx], 'r--')
            plt.xlabel(f'{label} Mean')
            plt.ylabel('MDBN Mean')
            plt.title(f'Mean Scatter ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_scatter_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Variance scatter
            plt.figure(figsize=(7, 7))
            x = results_df[f'var_{suffix}'].dropna()
            y = results_df.loc[x.index, 'var_mdbn'].dropna()
            common_idx = x.index.intersection(y.index)
            x = results_df.loc[common_idx, f'var_{suffix}']
            y = results_df.loc[common_idx, 'var_mdbn']
            plt.scatter(x, y, alpha=0.6)
            mx = max(float(x.max()), float(y.max())) if len(x) else 1.0
            plt.plot([0, mx], [0, mx], 'r--')
            plt.xlabel(f'{label} Variance')
            plt.ylabel('MDBN Variance')
            plt.title(f'Variance Scatter ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_scatter_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'Scatter plots failed for {label}: {e}')

        # ---------- Histograms ----------
        try:
            # Absolute error
            plt.figure(figsize=(8, 6))
            values = results_df[f'mean_err_{suffix}_abs'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Absolute Mean Error |MDBN - {label}|')
            plt.ylabel('Count')
            plt.title(f'Mean Absolute Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_abs_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            values = results_df[f'var_err_{suffix}_abs'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Absolute Variance Error |MDBN - {label}|')
            plt.ylabel('Count')
            plt.title(f'Variance Absolute Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_abs_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Percentage error
            plt.figure(figsize=(8, 6))
            values = results_df[f'mean_err_{suffix}_pct'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Mean % Error (MDBN vs {label})')
            plt.ylabel('Count')
            plt.title(f'Mean % Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_pct_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            values = results_df[f'var_err_{suffix}_pct'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Variance % Error (MDBN vs {label})')
            plt.ylabel('Count')
            plt.title(f'Variance % Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_pct_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'Histogram plots failed for {label}: {e}')

        # ---------- Boxplots by intervention ----------
        try:
            grouped_mean = []
            grouped_var = []
            labels_intv = []

            for t in sorted(results_df['intv_type'].dropna().unique()):
                mean_vals = results_df.loc[results_df['intv_type'] == t, f'mean_err_{suffix}_abs'].dropna().values
                var_vals = results_df.loc[results_df['intv_type'] == t, f'var_err_{suffix}_abs'].dropna().values
                if len(mean_vals) > 0 and len(var_vals) > 0:
                    grouped_mean.append(mean_vals)
                    grouped_var.append(var_vals)
                    labels_intv.append(t)

            if grouped_mean:
                plt.figure(figsize=(8, 6))
                plt.boxplot(grouped_mean, labels=labels_intv, showfliers=True)
                plt.xlabel('Intervention Type')
                plt.ylabel(f'Absolute Mean Error (MDBN vs {label})')
                plt.title(f'Mean Absolute Error vs Intervention ({workload_name} | MDBN vs {label})')
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_boxplot_{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()

            if grouped_var:
                plt.figure(figsize=(8, 6))
                plt.boxplot(grouped_var, labels=labels_intv, showfliers=True)
                plt.xlabel('Intervention Type')
                plt.ylabel(f'Absolute Variance Error (MDBN vs {label})')
                plt.title(f'Variance Absolute Error vs Intervention ({workload_name} | MDBN vs {label})')
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_boxplot_{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f'Boxplots failed for {label}: {e}')

    print(f'All 16 plots generated in {plot_dir}')



def compute_detailed_gamma_inference_results(config_file, gt_nreps=30):
    """
    Compute detailed moment-based inference diagnostics for the Gamma workload.

    For each experiment in the config, this function:
      1. Loads the Gamma ground-truth distribution and the HypoExp simulator GT.
      2. Loads the MDBN posterior distribution.
      3. Aligns the supports and normalizes all distributions.
      4. Computes mean and variance for:
           - MDBN posterior
           - Gamma GT (analytical/model distribution)
           - HypoExp GT (simulator approximation)
      5. Computes absolute and percentage errors.
      6. Saves a CSV with all per-query statistics.
      7. Generates 8 diagnostic plots and saves them under:
           results/distribution_mean_var_plots/

    Notes:
      - Experiments are skipped if GT/posterior files are missing, distributions are malformed,
        or distributions have zero total probability mass.
      - Percentage errors are set to NaN when the denominator is too small.
      - The logger is expected to be configured by the caller.
    """

    eps = 1e-12
    logger = logging.getLogger('utils')

    logger.info(f'Starting detailed Gamma inference moment analysis using config: {config_file}')

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    workload_name = os.path.basename(config_file).split('.')[0]
    logger.info(f'Inferred workload name: {workload_name}')

    results_df = pd.DataFrame(columns=[
        'exp_num', 'expt_name', 'intv_type', 'intv_var', 'intv_time', 'intv_val',
        'query_var', 'query_time', 'queue_setting',
        'mean_mdbn', 'mean_model', 'mean_sim',
        'var_mdbn', 'var_model', 'var_sim',
        'mean_err_model_abs', 'mean_err_sim_abs',
        'var_err_model_abs', 'var_err_sim_abs',
        'mean_err_model_pct', 'mean_err_sim_pct',
        'var_err_model_pct', 'var_err_sim_pct',
        'num_conditional_events', 'inference_time', 'inference_compute_time'
    ])

    skipped = Counter()
    total = 0

    for experiment in config:
        total += 1
        q = config[experiment]
        exp_num = int(experiment.split('_')[-1])

        logger.info(f'Processing experiment {exp_num}')

        # ---------- Ground truth paths ----------
        primary_gt_gamma = os.path.join(
            q['gt_results_folder'], q['expt_name'],
            f'gt-exp-{exp_num}-gamma.pkl'
        )
        pooled_gt_gamma = os.path.join(
            q['gt_results_folder'],
            f'queries_nreps-{gt_nreps}_pooled',
            f'gt-exp-{exp_num}-gamma.pkl'
        )

        primary_gt_hypo = os.path.join(
            q['gt_results_folder'], q['expt_name'],
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )
        pooled_gt_hypo = os.path.join(
            q['gt_results_folder'],
            f'queries_nreps-{gt_nreps}_pooled',
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )

        gamma_gt_used = primary_gt_gamma if os.path.exists(primary_gt_gamma) else (
            pooled_gt_gamma if os.path.exists(pooled_gt_gamma) else None
        )
        hypo_gt_used = primary_gt_hypo if os.path.exists(primary_gt_hypo) else (
            pooled_gt_hypo if os.path.exists(pooled_gt_hypo) else None
        )

        if gamma_gt_used is None:
            logger.warning(f'Experiment {exp_num}: missing Gamma GT. Skipping.')
            skipped['missing_gamma_gt'] += 1
            continue

        if hypo_gt_used is None:
            logger.warning(f'Experiment {exp_num}: missing HypoExp GT. Skipping.')
            skipped['missing_hypo_gt'] += 1
            continue

        # ---------- Load GT ----------
        try:
            with open(gamma_gt_used, 'rb') as f:
                gamma_gt_dict = pickle.load(f)
            with open(hypo_gt_used, 'rb') as f:
                hypo_gt_dict = pickle.load(f)
        except Exception as e:
            logger.warning(f'Experiment {exp_num}: GT load failed ({e}). Skipping.')
            skipped['bad_gt'] += 1
            continue

        gt_gamma = gamma_gt_dict.get('query_dist') if isinstance(gamma_gt_dict, dict) else None
        gt_hypo = hypo_gt_dict.get('query_dist') if isinstance(hypo_gt_dict, dict) else None

        if gt_gamma is None or gt_hypo is None:
            logger.warning(f'Experiment {exp_num}: GT missing query_dist. Skipping.')
            skipped['bad_gt_format'] += 1
            continue

        if sum(gt_gamma.values()) == 0 or sum(gt_hypo.values()) == 0:
            logger.warning(f'Experiment {exp_num}: GT has zero probability mass. Skipping.')
            skipped['zero_gt'] += 1
            continue

        # ---------- Posterior ----------
        posterior_path = os.path.join(
            q['results_folder'], workload_name,
            f'posterior-exp-{exp_num}.pkl'
        )

        if not os.path.exists(posterior_path):
            logger.warning(f'Experiment {exp_num}: missing posterior at {posterior_path}. Skipping.')
            skipped['missing_posterior'] += 1
            continue

        try:
            inferred_data = pd.read_pickle(posterior_path)
            inferred_pd = inferred_data.get('Posterior') if isinstance(inferred_data, dict) else inferred_data
            inference_time = inferred_data.get('InferenceTime') if isinstance(inferred_data, dict) else None
            inference_compute_time = inferred_data.get('FullInferenceTime') if isinstance(inferred_data, dict) else None
        except Exception as e:
            logger.warning(f'Experiment {exp_num}: posterior load failed ({e}). Skipping.')
            skipped['bad_posterior'] += 1
            continue

        # ---------- Align distributions ----------
        try:
            inferred_series = inferred_pd if isinstance(inferred_pd, pd.Series) else pd.Series(inferred_pd)

            states = sorted(set(list(gt_gamma.keys()) + list(gt_hypo.keys()) + list(inferred_series.index)))
            states_arr = np.array(states, dtype=float)

            posterior_probs = np.array([float(inferred_series.get(s, 0.0)) for s in states], dtype=float)
            gamma_probs = np.array([float(gt_gamma.get(s, 0.0)) for s in states], dtype=float)
            hypo_probs = np.array([float(gt_hypo.get(s, 0.0)) for s in states], dtype=float)

            if posterior_probs.sum() <= eps:
                logger.warning(f'Experiment {exp_num}: posterior has zero probability mass. Skipping.')
                skipped['zero_posterior'] += 1
                continue

            posterior_probs /= posterior_probs.sum()
            gamma_probs /= gamma_probs.sum()
            hypo_probs /= hypo_probs.sum()

        except Exception as e:
            logger.warning(f'Experiment {exp_num}: alignment error ({e}). Skipping.')
            skipped['alignment_error'] += 1
            continue

        # ---------- Moments ----------
        try:
            mean_mdbn = float(np.sum(states_arr * posterior_probs))
            mean_model = float(np.sum(states_arr * gamma_probs))
            mean_sim = float(np.sum(states_arr * hypo_probs))

            var_mdbn = float(np.sum((states_arr - mean_mdbn) ** 2 * posterior_probs))
            var_model = float(np.sum((states_arr - mean_model) ** 2 * gamma_probs))
            var_sim = float(np.sum((states_arr - mean_sim) ** 2 * hypo_probs))

            logger.debug(
                'Experiment %s moments: mean_mdbn=%.6f, mean_model=%.6f, mean_sim=%.6f, '
                'var_mdbn=%.6f, var_model=%.6f, var_sim=%.6f',
                exp_num, mean_mdbn, mean_model, mean_sim, var_mdbn, var_model, var_sim
            )

        except Exception as e:
            logger.warning(f'Experiment {exp_num}: moment computation error ({e}). Skipping.')
            skipped['moment_error'] += 1
            continue

        # ---------- Errors ----------
        #mean_err_model_abs = abs(mean_mdbn - mean_model)
        mean_err_model_abs = mean_mdbn - mean_model
        #mean_err_sim_abs = abs(mean_mdbn - mean_sim)
        mean_err_sim_abs = mean_mdbn - mean_sim
        var_err_model_abs = var_mdbn - var_model
        var_err_sim_abs = var_mdbn - var_sim

        mean_err_model_pct = (mean_err_model_abs / mean_model * 100.0) if abs(mean_model) > eps else np.nan
        mean_err_sim_pct = (mean_err_sim_abs / mean_sim * 100.0) if abs(mean_sim) > eps else np.nan
        var_err_model_pct = (var_err_model_abs / var_model * 100.0) if abs(var_model) > eps else np.nan
        var_err_sim_pct = (var_err_sim_abs / var_sim * 100.0) if abs(var_sim) > eps else np.nan

        # ---------- Metadata ----------
        try:
            intv = q.get('interventions', [{}])[0]
            intv_type = intv.get('intervention_type', 'unknown')
            intv_var = intv.get('intervention_variable')
            intv_time = intv.get('intervention_start')
            intv_val = intv.get('intervention_value')
        except Exception:
            intv_type = 'unknown'
            intv_var = None
            intv_time = None
            intv_val = None

        try:
            gt_csv_path = os.path.join(os.path.dirname(gamma_gt_used), f'gt-exp-{exp_num}-gamma.csv')
            gt_csv = pd.read_csv(gt_csv_path)
            num_cond_events = len(gt_csv[gt_csv['Event'] == 'Conditional'])
        except Exception:
            num_cond_events = None
            logger.debug(f'Experiment {exp_num}: could not read conditional event count.')

        queue_setting = q.get('expt_name').split('_')[-1] if q.get('expt_name') else None

        results_df.loc[len(results_df)] = [
            exp_num, q.get('expt_name'), intv_type, intv_var, intv_time, intv_val,
            q.get('query_variable'), q.get('query_time'), queue_setting,
            mean_mdbn, mean_model, mean_sim,
            var_mdbn, var_model, var_sim,
            mean_err_model_abs, mean_err_sim_abs,
            var_err_model_abs, var_err_sim_abs,
            mean_err_model_pct, mean_err_sim_pct,
            var_err_model_pct, var_err_sim_pct,
            num_cond_events, inference_time, inference_compute_time
        ]

        logger.info(
            'Experiment %s done | mean(model)=%.4f, mean(mdbn)=%.4f, mean(sim)=%.4f | '
            'var(model)=%.4f, var(mdbn)=%.4f, var(sim)=%.4f',
            exp_num, mean_model, mean_mdbn, mean_sim, var_model, var_mdbn, var_sim
        )

    # Merge additive/subtractive for grouping
    results_df['intv_type'] = results_df['intv_type'].replace({
        'additive': 'add_sub',
        'subtractive': 'add_sub'
    })

    # ---------- Save CSV ----------
    os.makedirs('results', exist_ok=True)
    csv_file = f'results/{workload_name}_moment_results.csv'
    results_df.to_csv(csv_file, index=False)
    logger.info(f'Saved Gamma moment results CSV to {csv_file}')
    logger.info(f'Total experiments processed: {total}')
    logger.info(f'Valid experiments: {len(results_df)}')
    logger.info(f'Skipped counts: {dict(skipped)}')

    generate_gamma_16_plots(results_df, workload_name)

    # Note: You can call generate_gamma_16_plots(results_df, workload_name) here later
    logger.info('Completed detailed Gamma inference moment analysis.')
    return results_df


def generate_gamma_16_plots(results_df, workload_name):
    """
    Generate 16 diagnostic plots comparing MDBN to both Gamma GT and HypoExp GT.
    Saves plots under 'results/distribution_mean_var_plots/'.
    """

    if results_df.empty:
        print("No valid experiments available; plots skipped.")
        return

    plot_dir = 'results/distribution_mean_var_plots'
    os.makedirs(plot_dir, exist_ok=True)

    comparisons = [
        ('model', 'Gamma GT'),
        ('sim', 'HypoExp GT')
    ]

    for suffix, label in comparisons:
        # ---------- Scatter plots ----------
        try:
            # Mean scatter
            plt.figure(figsize=(7, 7))
            x = results_df[f'mean_{suffix}'].dropna()
            y = results_df.loc[x.index, 'mean_mdbn'].dropna()
            common_idx = x.index.intersection(y.index)
            x = results_df.loc[common_idx, f'mean_{suffix}']
            y = results_df.loc[common_idx, 'mean_mdbn']
            plt.scatter(x, y, alpha=0.6)
            mx = max(float(x.max()), float(y.max())) if len(x) else 1.0
            plt.plot([0, mx], [0, mx], 'r--')
            plt.xlabel(f'{label} Mean')
            plt.ylabel('MDBN Mean')
            plt.title(f'Mean Scatter ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_scatter_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Variance scatter
            plt.figure(figsize=(7, 7))
            x = results_df[f'var_{suffix}'].dropna()
            y = results_df.loc[x.index, 'var_mdbn'].dropna()
            common_idx = x.index.intersection(y.index)
            x = results_df.loc[common_idx, f'var_{suffix}']
            y = results_df.loc[common_idx, 'var_mdbn']
            plt.scatter(x, y, alpha=0.6)
            mx = max(float(x.max()), float(y.max())) if len(x) else 1.0
            plt.plot([0, mx], [0, mx], 'r--')
            plt.xlabel(f'{label} Variance')
            plt.ylabel('MDBN Variance')
            plt.title(f'Variance Scatter ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_scatter_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'Scatter plots failed for {label}: {e}')

        # ---------- Histograms ----------
        try:
            # Absolute error
            plt.figure(figsize=(8, 6))
            values = results_df[f'mean_err_{suffix}_abs'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Absolute Mean Error |MDBN - {label}|')
            plt.ylabel('Count')
            plt.title(f'Mean Absolute Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_abs_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            values = results_df[f'var_err_{suffix}_abs'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Absolute Variance Error |MDBN - {label}|')
            plt.ylabel('Count')
            plt.title(f'Variance Absolute Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_abs_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Percentage error
            plt.figure(figsize=(8, 6))
            values = results_df[f'mean_err_{suffix}_pct'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Mean % Error (MDBN vs {label})')
            plt.ylabel('Count')
            plt.title(f'Mean % Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_pct_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(8, 6))
            values = results_df[f'var_err_{suffix}_pct'].dropna()
            plt.hist(values, bins=40)
            plt.xlabel(f'Variance % Error (MDBN vs {label})')
            plt.ylabel('Count')
            plt.title(f'Variance % Error ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_pct_hist_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f'Histogram plots failed for {label}: {e}')

        # ---------- Probability density plots ----------
        try:
            # Mean absolute error density
            plt.figure(figsize=(8, 6))
            values = results_df[f'mean_err_{suffix}_abs'].dropna()
            if len(values) > 0:
                sns.kdeplot(values, fill=True, alpha=0.5)
            plt.xlabel(f'Absolute Mean Error |MDBN - {label}|')
            plt.ylabel('Density')
            plt.title(f'Mean Absolute Error Density ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_abs_density_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Variance absolute error density
            plt.figure(figsize=(8, 6))
            values = results_df[f'var_err_{suffix}_abs'].dropna()
            if len(values) > 0:
                sns.kdeplot(values, fill=True, alpha=0.5)
            plt.xlabel(f'Absolute Variance Error |MDBN - {label}|')
            plt.ylabel('Density')
            plt.title(f'Variance Absolute Error Density ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_abs_density_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Mean percentage error density
            plt.figure(figsize=(8, 6))
            values = results_df[f'mean_err_{suffix}_pct'].dropna()
            if len(values) > 0:
                sns.kdeplot(values, fill=True, alpha=0.5)
            plt.xlabel(f'Mean % Error (MDBN vs {label})')
            plt.ylabel('Density')
            plt.title(f'Mean % Error Density ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_pct_density_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Variance percentage error density
            plt.figure(figsize=(8, 6))
            values = results_df[f'var_err_{suffix}_pct'].dropna()
            if len(values) > 0:
                sns.kdeplot(values, fill=True, alpha=0.5)
            plt.xlabel(f'Variance % Error (MDBN vs {label})')
            plt.ylabel('Density')
            plt.title(f'Variance % Error Density ({workload_name} | MDBN vs {label})')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_pct_density_{suffix}.png'), dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f'Probability density plots failed for {label}: {e}')

        # ---------- Boxplots by intervention ----------
        try:
            grouped_mean = []
            grouped_var = []
            labels_intv = []

            for t in sorted(results_df['intv_type'].dropna().unique()):
                mean_vals = results_df.loc[results_df['intv_type'] == t, f'mean_err_{suffix}_abs'].dropna().values
                var_vals = results_df.loc[results_df['intv_type'] == t, f'var_err_{suffix}_abs'].dropna().values
                if len(mean_vals) > 0 and len(var_vals) > 0:
                    grouped_mean.append(mean_vals)
                    grouped_var.append(var_vals)
                    labels_intv.append(t)

            if grouped_mean:
                plt.figure(figsize=(8, 6))
                plt.boxplot(grouped_mean, labels=labels_intv, showfliers=True)
                plt.xlabel('Intervention Type')
                plt.ylabel(f'Absolute Mean Error (MDBN vs {label})')
                plt.title(f'Mean Absolute Error vs Intervention ({workload_name} | MDBN vs {label})')
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_boxplot_{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()

            if grouped_var:
                plt.figure(figsize=(8, 6))
                plt.boxplot(grouped_var, labels=labels_intv, showfliers=True)
                plt.xlabel('Intervention Type')
                plt.ylabel(f'Absolute Variance Error (MDBN vs {label})')
                plt.title(f'Variance Absolute Error vs Intervention ({workload_name} | MDBN vs {label})')
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_boxplot_{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f'Boxplots failed for {label}: {e}')

        # ---------- Boxplots by query class ----------
        try:
            grouped_mean = []
            grouped_var = []
            labels_query = []

            for qv in sorted(results_df['query_var'].dropna().unique()):
                mean_vals = results_df.loc[results_df['query_var'] == qv, f'mean_err_{suffix}_abs'].dropna().values
                var_vals = results_df.loc[results_df['query_var'] == qv, f'var_err_{suffix}_abs'].dropna().values
                if len(mean_vals) > 0 and len(var_vals) > 0:
                    grouped_mean.append(mean_vals)
                    grouped_var.append(var_vals)
                    labels_query.append(qv)

            if grouped_mean:
                plt.figure(figsize=(8, 6))
                plt.boxplot(grouped_mean, labels=labels_query, showfliers=True)
                plt.xlabel('Query Class')
                plt.ylabel(f'Absolute Mean Error (MDBN vs {label})')
                plt.title(f'Mean Absolute Error vs Query Class ({workload_name} | MDBN vs {label})')
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                plt.savefig(os.path.join(plot_dir, f'{workload_name}_mean_boxplot_query_{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()

            if grouped_var:
                plt.figure(figsize=(8, 6))
                plt.boxplot(grouped_var, labels=labels_query, showfliers=True)
                plt.xlabel('Query Class')
                plt.ylabel(f'Absolute Variance Error (MDBN vs {label})')
                plt.title(f'Variance Absolute Error vs Query Class ({workload_name} | MDBN vs {label})')
                plt.grid(axis='y', linestyle='--', alpha=0.4)
                plt.savefig(os.path.join(plot_dir, f'{workload_name}_var_boxplot_query_{suffix}.png'), dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f'Boxplots by query class failed for {label}: {e}')

    print(f'All 16 plots generated in {plot_dir}')



#######################################################################
# New functions to generate csvs and plots
#######################################################################
import os
import pickle
import yaml
import pandas as pd
import numpy as np
import logging
from collections import Counter


import os
import pickle
import yaml
import pandas as pd
import numpy as np
import logging
from collections import Counter

def generate_gamma_csv(config_file):
    """
    Generate a CSV with MDBN, Gamma GT, and HypoExp statistics for all experiments.
    
    CSV columns:
        exp_num
        query_type
        mean_mdbn
        var_mdbn
        iqr_mdbn
        mean_gt
        var_gt
        iqr_gt
        mean_hypo
        var_hypo
        iqr_hypo
        diff_mean_mdbn_gt
        diff_var_mdbn_gt
        diff_iqr_mdbn_gt
        diff_mean_mdbn_hypo
        diff_var_mdbn_hypo
        diff_iqr_mdbn_hypo
    """
    eps = 1e-12
    logger = logging.getLogger('gamma_csv')
    logging.basicConfig(level=logging.INFO)
    
    logger.info(f'Starting Gamma CSV generation from config: {config_file}')
    
    # Load config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    workload_name = os.path.basename(config_file).split('.')[0]
    
    # Weighted quantile function
    def weighted_quantile(values, probs, quantile):
        sorter = np.argsort(values)
        values_sorted = values[sorter]
        probs_sorted = probs[sorter]
        cdf = np.cumsum(probs_sorted)
        return np.interp(quantile, cdf, values_sorted)
    
    # Prepare results dataframe
    columns = [
        'exp_num', 'query_type',
        'mean_mdbn', 'var_mdbn', 'iqr_mdbn',
        'mean_gt', 'var_gt', 'iqr_gt',
        'mean_hypo', 'var_hypo', 'iqr_hypo',
        'diff_mean_mdbn_gt', 'diff_var_mdbn_gt', 'diff_iqr_mdbn_gt',
        'diff_mean_mdbn_hypo', 'diff_var_mdbn_hypo', 'diff_iqr_mdbn_hypo'
    ]
    results_df = pd.DataFrame(columns=columns)
    
    skipped = Counter()
    total = 0
    
    for experiment_key in config:
        total += 1
        exp_cfg = config[experiment_key]
        exp_num = int(experiment_key.split('_')[-1])
        
        # ---------- Ground truth paths ----------
        gamma_gt_path = os.path.join(
            exp_cfg['gt_results_folder'], exp_cfg['expt_name'],
            f'gt-exp-{exp_num}-gamma.pkl'
        )
        hypo_gt_path = os.path.join(
            exp_cfg['gt_results_folder'], exp_cfg['expt_name'],
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )

        if not os.path.exists(gamma_gt_path) or not os.path.exists(hypo_gt_path):
            skipped['missing_gt'] += 1
            continue
        
        # ---------- Load GT ----------
        try:
            with open(gamma_gt_path, 'rb') as f:
                gamma_gt_dict = pickle.load(f)
            with open(hypo_gt_path, 'rb') as f:
                hypo_gt_dict = pickle.load(f)
        except Exception:
            skipped['bad_gt'] += 1
            continue
        
        gt_gamma = gamma_gt_dict.get('query_dist')
        gt_hypo = hypo_gt_dict.get('query_dist')
        if gt_gamma is None or gt_hypo is None or sum(gt_gamma.values())==0 or sum(gt_hypo.values())==0:
            skipped['invalid_gt'] += 1
            continue
        
        # ---------- Loop over queries ----------
        for intervention in exp_cfg['interventions']:
            query_type = intervention['intervention_type']  # Only query_type
            
            # Posterior path (note: no "gamma" in filename)
            posterior_path = os.path.join(
                exp_cfg['results_folder'], workload_name,
                f'posterior-exp-{exp_num}.pkl'
            )
            if not os.path.exists(posterior_path):
                skipped['missing_posterior'] += 1
                continue
            
            # ---------- Load posterior ----------
            try:
                posterior_data = pd.read_pickle(posterior_path)
                posterior_series = posterior_data.get('Posterior') if isinstance(posterior_data, dict) else posterior_data
                posterior_series = posterior_series if isinstance(posterior_series, pd.Series) else pd.Series(posterior_series)
            except Exception:
                skipped['bad_posterior'] += 1
                continue
            
            # ---------- Align distributions ----------
            states = sorted(set(list(gt_gamma.keys()) + list(gt_hypo.keys()) + list(posterior_series.index)))
            states_arr = np.array(states, dtype=float)
            
            posterior_probs = np.array([float(posterior_series.get(s,0.0)) for s in states])
            gamma_probs = np.array([float(gt_gamma.get(s,0.0)) for s in states])
            hypo_probs = np.array([float(gt_hypo.get(s,0.0)) for s in states])
            
            if posterior_probs.sum() <= eps:
                skipped['zero_posterior'] += 1
                continue
            
            posterior_probs /= posterior_probs.sum()
            gamma_probs /= gamma_probs.sum()
            hypo_probs /= hypo_probs.sum()
            
            # ---------- Compute moments ----------
            mean_mdbn = float(np.sum(states_arr * posterior_probs))
            var_mdbn = float(np.sum((states_arr - mean_mdbn)**2 * posterior_probs))
            q1, q3 = weighted_quantile(states_arr, posterior_probs, 0.25), weighted_quantile(states_arr, posterior_probs, 0.75)
            iqr_mdbn = float(q3 - q1)
            
            mean_gt = float(np.sum(states_arr * gamma_probs))
            var_gt = float(np.sum((states_arr - mean_gt)**2 * gamma_probs))
            q1, q3 = weighted_quantile(states_arr, gamma_probs, 0.25), weighted_quantile(states_arr, gamma_probs, 0.75)
            iqr_gt = float(q3 - q1)
            
            mean_hypo = float(np.sum(states_arr * hypo_probs))
            var_hypo = float(np.sum((states_arr - mean_hypo)**2 * hypo_probs))
            q1, q3 = weighted_quantile(states_arr, hypo_probs, 0.25), weighted_quantile(states_arr, hypo_probs, 0.75)
            iqr_hypo = float(q3 - q1)
            
            # ---------- Differences ----------
            diff_mean_mdbn_gt = mean_mdbn - mean_gt
            diff_var_mdbn_gt = var_mdbn - var_gt
            diff_iqr_mdbn_gt = iqr_mdbn - iqr_gt
            diff_mean_mdbn_hypo = mean_mdbn - mean_hypo
            diff_var_mdbn_hypo = var_mdbn - var_hypo
            diff_iqr_mdbn_hypo = iqr_mdbn - iqr_hypo
            
            # ---------- Save row ----------
            results_df.loc[len(results_df)] = [
                exp_num, query_type,
                mean_mdbn, var_mdbn, iqr_mdbn,
                mean_gt, var_gt, iqr_gt,
                mean_hypo, var_hypo, iqr_hypo,
                diff_mean_mdbn_gt, diff_var_mdbn_gt, diff_iqr_mdbn_gt,
                diff_mean_mdbn_hypo, diff_var_mdbn_hypo, diff_iqr_mdbn_hypo
            ]
    
    # ---------- Save CSV ----------
    os.makedirs('results', exist_ok=True)
    csv_file = f'results/{workload_name}_gamma_moments.csv'
    results_df.to_csv(csv_file, index=False)
    
    logger.info(f'Gamma CSV saved to {csv_file}')
    logger.info(f'Total experiments processed: {total}')
    logger.info(f'Skipped counts: {dict(skipped)}')
    
    return results_df


def generate_weibull_csv(config_file):
    """
    Generate a CSV with MDBN, Weibull GT, and HypoExp statistics for all experiments.
    
    CSV columns:
        exp_num
        query_type
        mean_mdbn
        var_mdbn
        iqr_mdbn
        mean_gt
        var_gt
        iqr_gt
        mean_hypo
        var_hypo
        iqr_hypo
        diff_mean_mdbn_gt
        diff_var_mdbn_gt
        diff_iqr_mdbn_gt
        diff_mean_mdbn_hypo
        diff_var_mdbn_hypo
        diff_iqr_mdbn_hypo
    """
    eps = 1e-12
    logger = logging.getLogger('weibull_csv')
    logging.basicConfig(level=logging.INFO)
    
    logger.info(f'Starting Weibull CSV generation from config: {config_file}')
    
    # Load config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    workload_name = os.path.basename(config_file).split('.')[0]
    
    # Weighted quantile function
    def weighted_quantile(values, probs, quantile):
        sorter = np.argsort(values)
        values_sorted = values[sorter]
        probs_sorted = probs[sorter]
        cdf = np.cumsum(probs_sorted)
        return np.interp(quantile, cdf, values_sorted)
    
    # Prepare results dataframe
    columns = [
        'exp_num', 'query_type',
        'mean_mdbn', 'var_mdbn', 'iqr_mdbn',
        'mean_gt', 'var_gt', 'iqr_gt',
        'mean_hypo', 'var_hypo', 'iqr_hypo',
        'diff_mean_mdbn_gt', 'diff_var_mdbn_gt', 'diff_iqr_mdbn_gt',
        'diff_mean_mdbn_hypo', 'diff_var_mdbn_hypo', 'diff_iqr_mdbn_hypo'
    ]
    results_df = pd.DataFrame(columns=columns)
    
    skipped = Counter()
    total = 0
    
    for experiment_key in config:
        total += 1
        exp_cfg = config[experiment_key]
        exp_num = int(experiment_key.split('_')[-1])
        
        # ---------- Ground truth paths ----------
        weibull_gt_path = os.path.join(
            exp_cfg['gt_results_folder'], exp_cfg['expt_name'],
            f'gt-exp-{exp_num}-weibull.pkl'
        )
        hypo_gt_path = os.path.join(
            exp_cfg['gt_results_folder'], exp_cfg['expt_name'],
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )

        if not os.path.exists(weibull_gt_path) or not os.path.exists(hypo_gt_path):
            skipped['missing_gt'] += 1
            continue
        
        # ---------- Load GT ----------
        try:
            with open(weibull_gt_path, 'rb') as f:
                weibull_gt_dict = pickle.load(f)
            with open(hypo_gt_path, 'rb') as f:
                hypo_gt_dict = pickle.load(f)
        except Exception:
            skipped['bad_gt'] += 1
            continue
        
        gt_weibull = weibull_gt_dict.get('query_dist')
        gt_hypo = hypo_gt_dict.get('query_dist')
        if gt_weibull is None or gt_hypo is None or sum(gt_weibull.values())==0 or sum(gt_hypo.values())==0:
            skipped['invalid_gt'] += 1
            continue
        
        # ---------- Loop over queries ----------
        for intervention in exp_cfg['interventions']:
            query_type = intervention['intervention_type']  # This is the only query_type
            
            # Posterior path
            posterior_path = os.path.join(
                exp_cfg['results_folder'], workload_name,
                f'weibull-posterior-exp-{exp_num}.pkl'
            )
            if not os.path.exists(posterior_path):
                skipped['missing_posterior'] += 1
                continue
            
            # ---------- Load posterior ----------
            try:
                posterior_data = pd.read_pickle(posterior_path)
                posterior_series = posterior_data.get('Posterior') if isinstance(posterior_data, dict) else posterior_data
                posterior_series = posterior_series if isinstance(posterior_series, pd.Series) else pd.Series(posterior_series)
            except Exception:
                skipped['bad_posterior'] += 1
                continue
            
            # ---------- Align distributions ----------
            states = sorted(set(list(gt_weibull.keys()) + list(gt_hypo.keys()) + list(posterior_series.index)))
            states_arr = np.array(states, dtype=float)
            
            posterior_probs = np.array([float(posterior_series.get(s,0.0)) for s in states])
            weibull_probs = np.array([float(gt_weibull.get(s,0.0)) for s in states])
            hypo_probs = np.array([float(gt_hypo.get(s,0.0)) for s in states])
            
            if posterior_probs.sum() <= eps:
                skipped['zero_posterior'] += 1
                continue
            
            posterior_probs /= posterior_probs.sum()
            weibull_probs /= weibull_probs.sum()
            hypo_probs /= hypo_probs.sum()
            
            # ---------- Compute moments ----------
            mean_mdbn = float(np.sum(states_arr * posterior_probs))
            var_mdbn = float(np.sum((states_arr - mean_mdbn)**2 * posterior_probs))
            q1, q3 = weighted_quantile(states_arr, posterior_probs, 0.25), weighted_quantile(states_arr, posterior_probs, 0.75)
            iqr_mdbn = float(q3 - q1)
            
            mean_gt = float(np.sum(states_arr * weibull_probs))
            var_gt = float(np.sum((states_arr - mean_gt)**2 * weibull_probs))
            q1, q3 = weighted_quantile(states_arr, weibull_probs, 0.25), weighted_quantile(states_arr, weibull_probs, 0.75)
            iqr_gt = float(q3 - q1)
            
            mean_hypo = float(np.sum(states_arr * hypo_probs))
            var_hypo = float(np.sum((states_arr - mean_hypo)**2 * hypo_probs))
            q1, q3 = weighted_quantile(states_arr, hypo_probs, 0.25), weighted_quantile(states_arr, hypo_probs, 0.75)
            iqr_hypo = float(q3 - q1)
            
            # ---------- Differences ----------
            diff_mean_mdbn_gt = mean_mdbn - mean_gt
            diff_var_mdbn_gt = var_mdbn - var_gt
            diff_iqr_mdbn_gt = iqr_mdbn - iqr_gt
            diff_mean_mdbn_hypo = mean_mdbn - mean_hypo
            diff_var_mdbn_hypo = var_mdbn - var_hypo
            diff_iqr_mdbn_hypo = iqr_mdbn - iqr_hypo
            
            # ---------- Save row ----------
            results_df.loc[len(results_df)] = [
                exp_num, query_type,
                mean_mdbn, var_mdbn, iqr_mdbn,
                mean_gt, var_gt, iqr_gt,
                mean_hypo, var_hypo, iqr_hypo,
                diff_mean_mdbn_gt, diff_var_mdbn_gt, diff_iqr_mdbn_gt,
                diff_mean_mdbn_hypo, diff_var_mdbn_hypo, diff_iqr_mdbn_hypo
            ]
    
    # ---------- Save CSV ----------
    os.makedirs('results', exist_ok=True)
    csv_file = f'results/{workload_name}_weibull_moments.csv'
    results_df.to_csv(csv_file, index=False)
    
    logger.info(f'Weibull CSV saved to {csv_file}')
    logger.info(f'Total experiments processed: {total}')
    logger.info(f'Skipped counts: {dict(skipped)}')
    
    return results_df


def generate_beta_csv(config_file):
    """
    Generate a CSV with MDBN, Beta GT, and HypoExp statistics for all experiments.
    
    CSV columns:
        exp_num
        query_type
        mean_mdbn
        var_mdbn
        iqr_mdbn
        mean_gt
        var_gt
        iqr_gt
        mean_hypo
        var_hypo
        iqr_hypo
        diff_mean_mdbn_gt
        diff_var_mdbn_gt
        diff_iqr_mdbn_gt
        diff_mean_mdbn_hypo
        diff_var_mdbn_hypo
        diff_iqr_mdbn_hypo
    """
    eps = 1e-12
    logger = logging.getLogger('beta_csv')
    logging.basicConfig(level=logging.INFO)
    
    logger.info(f'Starting Beta CSV generation from config: {config_file}')
    
    # Load config
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    workload_name = os.path.basename(config_file).split('.')[0]
    
    # Weighted quantile function
    def weighted_quantile(values, probs, quantile):
        sorter = np.argsort(values)
        values_sorted = values[sorter]
        probs_sorted = probs[sorter]
        cdf = np.cumsum(probs_sorted)
        return np.interp(quantile, cdf, values_sorted)
    
    # Prepare results dataframe
    columns = [
        'exp_num', 'query_type',
        'mean_mdbn', 'var_mdbn', 'iqr_mdbn',
        'mean_gt', 'var_gt', 'iqr_gt',
        'mean_hypo', 'var_hypo', 'iqr_hypo',
        'diff_mean_mdbn_gt', 'diff_var_mdbn_gt', 'diff_iqr_mdbn_gt',
        'diff_mean_mdbn_hypo', 'diff_var_mdbn_hypo', 'diff_iqr_mdbn_hypo'
    ]
    results_df = pd.DataFrame(columns=columns)
    
    skipped = Counter()
    total = 0
    
    for experiment_key in config:
        total += 1
        exp_cfg = config[experiment_key]
        exp_num = int(experiment_key.split('_')[-1])
        
        # ---------- Ground truth paths ----------
        beta_gt_path = os.path.join(
            exp_cfg['gt_results_folder'], exp_cfg['expt_name'],
            f'gt-exp-{exp_num}-beta.pkl'
        )
        hypo_gt_path = os.path.join(
            exp_cfg['gt_results_folder'], exp_cfg['expt_name'],
            f'gt-exp-{exp_num}-hypoexp.pkl'
        )

        if not os.path.exists(beta_gt_path) or not os.path.exists(hypo_gt_path):
            skipped['missing_gt'] += 1
            continue
        
        # ---------- Load GT ----------
        try:
            with open(beta_gt_path, 'rb') as f:
                beta_gt_dict = pickle.load(f)
            with open(hypo_gt_path, 'rb') as f:
                hypo_gt_dict = pickle.load(f)
        except Exception:
            skipped['bad_gt'] += 1
            continue
        
        gt_beta = beta_gt_dict.get('query_dist')
        gt_hypo = hypo_gt_dict.get('query_dist')
        if gt_beta is None or gt_hypo is None or sum(gt_beta.values())==0 or sum(gt_hypo.values())==0:
            skipped['invalid_gt'] += 1
            continue
        
        # ---------- Loop over queries ----------
        for intervention in exp_cfg['interventions']:
            query_type = intervention['intervention_type']  # This is the only query_type
            
            # Posterior path
            posterior_path = os.path.join(
                exp_cfg['results_folder'], workload_name,
                f'beta-posterior-exp-{exp_num}.pkl'
            )
            if not os.path.exists(posterior_path):
                skipped['missing_posterior'] += 1
                continue
            
            # ---------- Load posterior ----------
            try:
                posterior_data = pd.read_pickle(posterior_path)
                posterior_series = posterior_data.get('Posterior') if isinstance(posterior_data, dict) else posterior_data
                posterior_series = posterior_series if isinstance(posterior_series, pd.Series) else pd.Series(posterior_series)
            except Exception:
                skipped['bad_posterior'] += 1
                continue
            
            # ---------- Align distributions ----------
            states = sorted(set(list(gt_beta.keys()) + list(gt_hypo.keys()) + list(posterior_series.index)))
            states_arr = np.array(states, dtype=float)
            
            posterior_probs = np.array([float(posterior_series.get(s,0.0)) for s in states])
            beta_probs = np.array([float(gt_beta.get(s,0.0)) for s in states])
            hypo_probs = np.array([float(gt_hypo.get(s,0.0)) for s in states])
            
            if posterior_probs.sum() <= eps:
                skipped['zero_posterior'] += 1
                continue
            
            posterior_probs /= posterior_probs.sum()
            beta_probs /= beta_probs.sum()
            hypo_probs /= hypo_probs.sum()
            
            # ---------- Compute moments ----------
            mean_mdbn = float(np.sum(states_arr * posterior_probs))
            var_mdbn = float(np.sum((states_arr - mean_mdbn)**2 * posterior_probs))
            q1, q3 = weighted_quantile(states_arr, posterior_probs, 0.25), weighted_quantile(states_arr, posterior_probs, 0.75)
            iqr_mdbn = float(q3 - q1)
            
            mean_gt = float(np.sum(states_arr * beta_probs))
            var_gt = float(np.sum((states_arr - mean_gt)**2 * beta_probs))
            q1, q3 = weighted_quantile(states_arr, beta_probs, 0.25), weighted_quantile(states_arr, beta_probs, 0.75)
            iqr_gt = float(q3 - q1)
            
            mean_hypo = float(np.sum(states_arr * hypo_probs))
            var_hypo = float(np.sum((states_arr - mean_hypo)**2 * hypo_probs))
            q1, q3 = weighted_quantile(states_arr, hypo_probs, 0.25), weighted_quantile(states_arr, hypo_probs, 0.75)
            iqr_hypo = float(q3 - q1)
            
            # ---------- Differences ----------
            diff_mean_mdbn_gt = mean_mdbn - mean_gt
            diff_var_mdbn_gt = var_mdbn - var_gt
            diff_iqr_mdbn_gt = iqr_mdbn - iqr_gt
            diff_mean_mdbn_hypo = mean_mdbn - mean_hypo
            diff_var_mdbn_hypo = var_mdbn - var_hypo
            diff_iqr_mdbn_hypo = iqr_mdbn - iqr_hypo
            
            # ---------- Save row ----------
            results_df.loc[len(results_df)] = [
                exp_num, query_type,
                mean_mdbn, var_mdbn, iqr_mdbn,
                mean_gt, var_gt, iqr_gt,
                mean_hypo, var_hypo, iqr_hypo,
                diff_mean_mdbn_gt, diff_var_mdbn_gt, diff_iqr_mdbn_gt,
                diff_mean_mdbn_hypo, diff_var_mdbn_hypo, diff_iqr_mdbn_hypo
            ]
    
    # ---------- Save CSV ----------
    os.makedirs('results', exist_ok=True)
    csv_file = f'results/{workload_name}_beta_moments.csv'
    results_df.to_csv(csv_file, index=False)
    
    logger.info(f'Beta CSV saved to {csv_file}')
    logger.info(f'Total experiments processed: {total}')
    logger.info(f'Skipped counts: {dict(skipped)}')
    
    return results_df



import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

def generate_density_scatter(csv_file):
    """
    Generate density (KDE) and enhanced scatter plots from a CSV of MDBN, GT, and HypoExp statistics.

    Features:
        - Density plots for mean/variance differences
        - Scatter plots with y=x line
        - R^2, Pearson correlation, Spearman correlation, SMAPE (%) displayed on scatter plots
        - Separate plots for Overall, Conditional, Interventional, Param intervention, Add/Sub query types
        - Outputs saved under results/<csv_name>/density/ and results/<csv_name>/scatter/

    Requirements:
        CSV must contain columns:
            - query_type
            - mean_mdbn, var_mdbn
            - mean_gt, var_gt
            - mean_hypo, var_hypo
            - diff_mean_mdbn_gt, diff_var_mdbn_gt
            - diff_mean_mdbn_hypo, diff_var_mdbn_hypo
    """

    # -----------------------------
    # Helper functions
    # -----------------------------
    def normalize_query_type(q):
        """Map CSV query_type to one of 4 groups."""
        if pd.isna(q):
            return None
        qn = str(q).strip().lower().replace("-", "_").replace(" ", "_")
        if qn in {"conditional"}:
            return "Conditional"
        if qn in {"interventional"}:
            return "Interventional"
        if qn in {"param_intervention", "parameter_intervention", "paramintervention"}:
            return "Param intervention"
        if qn in {"additive", "subtractive", "add_sub", "add/sub", "addsub"}:
            return "Add/Sub"
        return None

    def safe_name(s):
        """Make string safe for filenames."""
        s = str(s).strip().lower()
        s = re.sub(r"[\s\-]+", "_", s)
        s = re.sub(r"[^a-z0-9_]+", "", s)
        return s

    def compute_stats(y_true, y_pred):
        """Compute R2, Pearson, Spearman, SMAPE for given arrays."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        use_log = False  # add this line if you want log transform
        if use_log:
            y_true = np.log1p(y_true)
            y_pred = np.log1p(y_pred)

        if len(y_true) < 2:
            return np.nan, np.nan, np.nan, np.nan

        mae = np.mean(np.abs(y_true - y_pred))
        return mae


    def plot_stats_box(ax, mae):
        stats_text = (rf"$\mathrm{{MAE}} = {mae:.3f}$")
        ax.text(
            0.05, 0.95, stats_text,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=25,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.9)
        )

    def add_y_eq_x(ax, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            return

        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        pad = 0.05 * (max_val - min_val) if max_val != min_val else 1

        min_val -= pad
        max_val += pad

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            linewidth=2,
            color="red",
            #label=r"$\hat{y} = y$"
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    # -----------------------------
    # Load CSV and setup
    # -----------------------------
    df = pd.read_csv(csv_file)
    if "query_type" not in df.columns:
        raise ValueError("CSV must contain a 'query_type' column.")
    df["query_group"] = df["query_type"].apply(normalize_query_type)

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    density_dir = os.path.join("results", base_name, "density")
    scatter_dir = os.path.join("results", base_name, "scatter")
    os.makedirs(density_dir, exist_ok=True)
    os.makedirs(scatter_dir, exist_ok=True)

    sns.set_style("ticks")
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
    })
    query_colors = {
        "Conditional": "#1f77b4",
        "Interventional": "#ff7f0e",
        "Param intervention": "#2ca02c",
        "Add/Sub": "#d62728",
    }

    # -----------------------------
    # Density Plots
    # -----------------------------
    density_specs = [
        ("diff_mean_mdbn_gt", r"$\mathbb{E}[\widehat{L_{\tau_j}}] - \mathbb{E}[L_{\tau_j}]$ (MDBN - GT)"),
        ("diff_var_mdbn_gt", r"$\mathrm{Var}(\widehat{L_{\tau_j}}) - \mathrm{Var}(L_{\tau_j})$ (MDBN - GT)"),
        ("diff_mean_mdbn_hypo", r"$\mathbb{E}[\widehat{L_{\tau_j}}] - \mathbb{E}[L_{\tau_j}]$ (MDBN - HypoExp)"),
        ("diff_var_mdbn_hypo", r"$\mathrm{Var}(\widehat{L_{\tau_j}}) - \mathrm{Var}(L_{\tau_j})$ (MDBN - HypoExp)"),
        ("diff_iqr_mdbn_gt", r"$\mathrm{IQR}(\widehat{L_{\tau_j}}) - \mathrm{IQR}(L_{\tau_j})$ (MDBN - GT)"),
        ("diff_iqr_mdbn_hypo", r"$\mathrm{IQR}(\widehat{L_{\tau_j}}) - \mathrm{IQR}(L_{\tau_j})$ (MDBN - HypoExp)")
    ]

    for col, xlabel in density_specs:
        # Overall
        fig, ax = plt.subplots(figsize=(8,6))

        ax.tick_params(axis='both', which='both', direction='out', length=6, width=1.5, labelsize=12)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        for spine in ["left", "bottom"]:
                ax.spines[spine].set_visible(True)

        vals = df[col].dropna()
        if len(vals) >= 2 and np.std(vals) > 0:
            sns.kdeplot(vals, fill=True, ax=ax)
        else:
            ax.axvline(vals.iloc[0] if len(vals) else 0, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        #ax.set_title(f"{xlabel} - Overall")
        plt.tight_layout()
        fig.savefig(os.path.join(density_dir, f"{safe_name(col)}_overall.png"), dpi=300)
        plt.close(fig)

        # By query groups
        for group in ["Conditional", "Interventional", "Param intervention", "Add/Sub"]:
            subset = df[df["query_group"] == group]
            fig, ax = plt.subplots(figsize=(8,6))

            ax.tick_params(axis='both', which='both', direction='out', length=6, width=1.5, labelsize=12)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_visible(True)

            vals = subset[col].dropna()
            if len(vals) >= 2 and np.std(vals) > 0:
                sns.kdeplot(vals, fill=True, ax=ax, color=query_colors[group])
            elif len(vals) > 0:
                ax.axvline(vals.iloc[0], linestyle="--", color=query_colors[group])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            #ax.set_title(f"{xlabel} - {group}")
            plt.tight_layout()
            fig.savefig(os.path.join(density_dir, f"{safe_name(col)}_{safe_name(group)}.png"), dpi=300)
            plt.close(fig)

    # -----------------------------
    # Scatter Plots
    # -----------------------------
    # add SD scatter plots
    df["sd_mdbn"] = np.sqrt(df["var_mdbn"])
    df["sd_gt"] = np.sqrt(df["var_gt"])
    df["sd_hypo"] = np.sqrt(df["var_hypo"])

    scatter_specs = [
        ("mean_mdbn", "mean_gt", "Mean (MDBN vs General/GT)"),
        ("var_mdbn", "var_gt", "Variance (MDBN vs General/GT)"),
        ("sd_mdbn", "sd_gt", "SD (MDBN vs General/GT)"),
        ("mean_mdbn", "mean_hypo", "Mean (MDBN vs HypoExp)"),
        ("var_mdbn", "var_hypo", "Variance (MDBN vs HypoExp)"),
        ("sd_mdbn", "sd_hypo", "SD (MDBN vs HypoExp)"),
        ("iqr_mdbn", "iqr_gt", "IQR (MDBN vs General/GT)"),
        ("iqr_mdbn", "iqr_hypo", "IQR (MDBN vs HypoExp)")
    ]

    for x_col, y_col, title_prefix in scatter_specs:
        plot_groups = [
            ("Overall", df),
            ("Conditional", df[df["query_group"] == "Conditional"]),
            ("Interventional", df[df["query_group"] == "Interventional"]),
            ("Param intervention", df[df["query_group"] == "Param intervention"]),
            ("Add/Sub", df[df["query_group"] == "Add/Sub"]),
        ]

        for group_name, subset in plot_groups:
            # Ground truth on x, MDBN on y
            x = subset[y_col].to_numpy(dtype=float)
            y = subset[x_col].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            fig, ax = plt.subplots(figsize=(8,6))

            ax.tick_params(axis='both', which='both', direction='out', length=6, width=1.5, labelsize=25)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_visible(True)

            if group_name == "Overall":
                ax.scatter(x, y, alpha=0.75, s=35)
            else:
                ax.scatter(x, y, alpha=0.8, s=40, color=query_colors[group_name])

            add_y_eq_x(ax, x, y)
            ax.locator_params(axis='both', nbins=6)
            mae = compute_stats(y_true=x, y_pred=y)
            #plot_stats_box(ax, r2, pearson, spearman, smape, mae, mape, medae)
            plot_stats_box(ax, mae)

            if "mean" in x_col:
                ax.set_xlabel(r"Ground truth: $\mathbb{E}[L_{\tau_j}]$", fontsize=25)
                ax.set_ylabel(r"MDBN: $\mathbb{E}[\widehat{L_{\tau_j}}]$", fontsize=25)
            elif "var" in x_col:
                ax.set_xlabel(r"Ground truth: $\mathrm{Var}(L_{\tau_j})$", fontsize=25)
                ax.set_ylabel(r"MDBN: $\mathrm{Var}(\widehat{L_{\tau_j}})$", fontsize=25)
            elif "sd" in x_col:
                ax.set_xlabel(r"Ground truth: $\mathrm{SD}(L_{\tau_j})$", fontsize=25)
                ax.set_ylabel(r"MDBN: $\mathrm{SD}(\widehat{L_{\tau_j}})$", fontsize=25)
            elif "iqr" in x_col:
                ax.set_xlabel(r"Ground truth: $\mathrm{IQR}(L_{\tau_j})$", fontsize=25)
                ax.set_ylabel(r"MDBN: $\mathrm{IQR}(\widehat{L_{\tau_j}})$", fontsize=25)

            ax.grid(False)
            plt.tight_layout()
            out_name = f"{safe_name(x_col)}_vs_{safe_name(y_col)}_{safe_name(group_name)}.pdf"
            fig.savefig(os.path.join(scatter_dir, out_name), bbox_inches="tight")


    def plot_ecdf(ax, values, color=None, label=None):
        """
        Plot the empirical CDF (ECDF) of a 1D array of values.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return

        sorted_vals = np.sort(values)
        yvals = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        ax.step(sorted_vals, yvals, where="post", color=color, label=label)

    # -----------------------------
    # CDF Plots (absolute error)
    # -----------------------------
    df["diff_sd_mdbn_gt"] = df["sd_mdbn"] - df["sd_gt"]
    df["diff_sd_mdbn_hypo"] = df["sd_mdbn"] - df["sd_hypo"]

    cdf_dir = os.path.join("results", base_name, "cdf")
    os.makedirs(cdf_dir, exist_ok=True)

    cdf_specs = [
        ("diff_mean_mdbn_gt", "Absolute mean difference |MDBN - General/GT|"),
        ("diff_var_mdbn_gt", "Absolute variance difference |MDBN - General/GT|"),
        ("diff_sd_mdbn_gt", "Absolute SD difference |MDBN - General/GT|"),

        ("diff_mean_mdbn_hypo", "Absolute mean difference |MDBN - HypoExp|"),
        ("diff_var_mdbn_hypo", "Absolute variance difference |MDBN - HypoExp|"),
        ("diff_sd_mdbn_hypo", "Absolute SD difference |MDBN - HypoExp|"),

        ("diff_iqr_mdbn_gt", "Absolute IQR difference |MDBN - General/GT|"),
        ("diff_iqr_mdbn_hypo", "Absolute IQR difference |MDBN - HypoExp|")
    ]

    for col, xlabel in cdf_specs:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.tick_params(axis='both', which='both', direction='out', length=6, width=1.5, labelsize=12)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        for spine in ["left", "bottom"]:
            ax.spines[spine].set_visible(True)

        abs_vals = np.abs(df[col].dropna().to_numpy(dtype=float))
        abs_vals = abs_vals[np.isfinite(abs_vals)]

        if len(abs_vals) == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        else:
            plot_ecdf(ax, abs_vals, color="#1f77b4", label="ECDF of absolute error")

        if "mean" in col:
            ax.set_xlabel(
                r"$\left|\mathbb{E}[\widehat{L_{\tau_j}}] - \mathbb{E}[L_{\tau_j}]\right|$",
                fontsize=14
            )
        elif "var" in col:
            ax.set_xlabel(
                r"$\left|\mathrm{Var}(\widehat{L_{\tau_j}}) - \mathrm{Var}(L_{\tau_j})\right|$",
                fontsize=14
            )
        elif "sd" in col:
            ax.set_xlabel(
                r"$\left|\mathrm{SD}(\widehat{L_{\tau_j}}) - \mathrm{SD}(L_{\tau_j})\right|$",
                fontsize=14
            )
        elif "iqr" in col:
            ax.set_xlabel(
                r"$\left|\mathrm{IQR}(\widehat{L_{\tau_j}}) - \mathrm{IQR}(L_{\tau_j})\right|$",
                fontsize=14
            )
        ax.set_ylabel(r"$\mathrm{ECDF}$", fontsize=14)
        ax.grid(False)
        plt.tight_layout()

        fig.savefig(
            os.path.join(cdf_dir, f"{safe_name(col)}_overall.png"),
            dpi=300
        )
        plt.close(fig)

    print(f"Density plots saved to: {density_dir}")
    print(f"Scatter plots saved to: {scatter_dir}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Compute the detailed JSD for each query in the config file
    #compute_jsd_detailed(
    #    f'config/query_workload_exp-6.json',
    #    gt_nreps=30)

    #compute_jsd_detailed_weibull(
    #    f'config/weibull_query_workload_exp-5.json',
    #    gt_nreps=30)

    #compute_jsd_detailed_beta(
    #    f'config/beta_query_workload_exp-1.json',
    #    gt_nreps=30)

    #compute_detailed_weibull_inference_results(f'config/weibull_query_workload_exp-5.json', gt_nreps=30)

    #compute_detailed_beta_inference_results(f'config/beta_query_workload_exp-1.json', gt_nreps=30)

    #compute_detailed_gamma_inference_results(f'config/query_workload_exp-6.json', gt_nreps=30)

    #generate_gamma_csv(f'config/query_workload_exp-6.json')

    #generate_weibull_csv(f'config/weibull_query_workload_exp-5.json')

    #generate_beta_csv(f'config/beta_query_workload_exp-3.json')

    generate_density_scatter(f'results/query_workload_exp-6_gamma_moments.csv')

    generate_density_scatter(f'results/weibull_query_workload_exp-5_weibull_moments.csv')

    generate_density_scatter(f'results/beta_query_workload_exp-3_beta_moments.csv')