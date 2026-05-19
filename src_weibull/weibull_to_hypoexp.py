"""
Script to simulate a G/M/1 distribution using Weibull as the arrival process.
"""
import argparse
import json
import os
from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from pathlib import Path
import yaml
from approx_hypoexp_NLP import sweep_k_and_select, hypoexp_pdf, hypoexp_moments_np


def weibull_hypoexponential_approximation(wb_shape, wb_scale,
                                            k_min=2,
                                            k_max=8,
                                            n_samples=50000,
                                            seed=42):
    """
    Returns an approximation of the list of hypoexponential phase rates
    given the Weibull distribution parameters.

    Uses moment-based NLP phase-type fitting with k-sweep selection.

    Parameters
    ----------
    wb_shape : float
        Shape parameter of Weibull distribution.
    wb_scale : float
        Scale parameter of Weibull distribution.

    Returns
    -------
    phase_rates : list[float]
        List of hypoexponential phase rates (lambdas).
    """

    assert(wb_shape > 0), "Shape must be positive for a valid Weibull distribution"
    assert(wb_scale > 0), "Scale must be positive for a valid Weibull distribution"

    phase_rates = []

    try:
        # Map to scipy weibull parameters
        params = {
            "c": wb_shape,
            "scale": wb_scale
        }

        info = sweep_k_and_select(
            distribution="weibull",
            params=params,
            k_min=k_min,
            k_max=k_max,
            n_samples=n_samples,
            seed=seed,
            lambda_bounds=(0.1, 10),
            verbose=False
        )

        best = info.get("best", None)

        best = info["best"]
        print(f"-> best k = {best['k']}, JSD = {best['jsd']:.6e}, CE ≈ {best['ce']:.6e}")
        print(f"-> lambdas = {np.round(best['lambdas'],6)}")

        # Compute hypoexp moments for the selected lambdas
        h_mom = hypoexp_moments_np(best['lambdas'])
        print(f"-> Hypoexp moments: mean={h_mom[0]:.4f}, var={h_mom[1]:.4f}, 3rd-proxy={h_mom[2]:.4f}")

        # Compare to target moments
        tm = info["target_moments"]
        print(f"-> Target moments: mean={tm[0]:.4f}, var={tm[1]:.4f}, 3rd-proxy={tm[2]:.4f}")

        if best is None:
            print(f"Weibull hypoexp fit failed for wb_shape={wb_shape}, wb_scale={wb_scale}")
            return phase_rates

        lambdas = best.get("lambdas", None)

        if lambdas is None or len(lambdas) == 0:
            print(f"Invalid hypoexp parameters for wb_shape={wb_shape}, wb_scale={wb_scale}")
            return phase_rates

        phase_rates = list(lambdas)

    except Exception as e:
        print(f"Weibull hypoexp approximation failed for wb_shape={wb_shape}, wb_scale={wb_scale}: {e}")

    return phase_rates


def plot_weibull_vs_hypoexp(wb_shape, wb_scale, phase_rates, experiment_no, num_samples=100000):
    """
    Plot the Weibull distribution and the Hypoexponential approximation on the same graph
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import weibull_min
    import os

    # sample from Weibull
    samples = wb_scale * np.random.weibull(wb_shape, size=num_samples)

    # plotting window (important for heavy tail)
    x_max = np.percentile(samples, 99.5)
    x = np.linspace(0, x_max, 1000)

    # Weibull pdf
    weibull_pdf = weibull_min.pdf(x, c=wb_shape, scale=wb_scale)

    # exact hypoexp pdf
    hypo_pdf = hypoexp_pdf(x, np.array(phase_rates))

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(x, weibull_pdf, label="Weibull", color="#0072B2", lw=2)       # Blue
    plt.plot(x, hypo_pdf, label="GED", color="#D62E00", lw=2)               # Red

    # --- Labels in LaTeX with bigger fonts ---
    plt.xlabel(r"$X$", fontsize=25)
    plt.ylabel(r"$P(X)$", fontsize=25)

    # --- Ticks with bigger fonts ---
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    # --- Legend ---
    plt.legend(fontsize=24)

    # --- Remove grid ---
    plt.grid(False)

    # --- Save figure as PDF ---
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figures/weibull_vs_hypoexp_exp{experiment_no}_shape_{wb_shape}_scale_{wb_scale}.pdf')
    plt.close()


def compute_cross_entropy_weibull(wb_shape, wb_scale, phase_rates, num_samples=100000, epsilon=1e-12):
    """
    Compute approximate cross-entropy H(Weibull || Hypoexp)
    """
    # sample from weibull
    samples = wb_scale * np.random.weibull(wb_shape, size=num_samples)

    # evaluate hypoexp pdf
    hypo_pdf_vals = hypoexp_pdf(samples, np.array(phase_rates))

    # avoid log(0)
    hypo_pdf_vals = np.maximum(hypo_pdf_vals, epsilon)

    cross_entropy = -np.mean(np.log(hypo_pdf_vals))

    return cross_entropy
    

def compute_kl_divergence_weibull(wb_shape, wb_scale, phase_rates, num_samples=100000, epsilon=1e-12):
    """
    Compute approximate KL divergence KL(Weibull || Hypoexp)

    KL(P || Q) = E_P[ log P(x) - log Q(x) ]
    where P is the Weibull distribution and Q is the Hypoexponential approximation.
    """

    # sample from weibull
    samples = wb_scale * np.random.weibull(wb_shape, size=num_samples)

    # evaluate true weibull pdf
    p_vals = weibull_min.pdf(samples, c=wb_shape, scale=wb_scale)

    # evaluate hypoexp pdf
    q_vals = hypoexp_pdf(samples, np.array(phase_rates))

    # avoid log(0)
    p_vals = np.maximum(p_vals, epsilon)
    q_vals = np.maximum(q_vals, epsilon)

    kl_divergence = np.mean(np.log(p_vals) - np.log(q_vals))

    return kl_divergence


def create_hypoexp_sim_config(query_workload): 
    """
    Function creates the HypoExp/M/1 simulation config for a particular query workload. 
    Weibull-based version.
    """
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    input_config_file = project_root / query_workload.get('config_file', None)
    input_experiment_number  = query_workload.get('experiment_number', None)
    output_path = config_dir / query_workload.get('output_file', f'hypoexp_weibull_simulator.yaml')

    with open(input_config_file, "r") as f:
        input_config_data = yaml.safe_load(f)
    input_config = input_config_data.get(f"experiment_{input_experiment_number}", None)

    # Step 2: Approx weibull with hypoexp, plot both distributions
    arrival_distributions = []
    for wb_shape in input_config['WB_SHAPES']:
        for wb_scale in input_config['WB_SCALES']:
            phase_rates = weibull_hypoexponential_approximation(wb_shape, wb_scale, k_min=2, k_max=5)
            plot_weibull_vs_hypoexp(wb_shape, wb_scale, phase_rates, input_experiment_number)
            cross_entropy = compute_cross_entropy_weibull(wb_shape, wb_scale, phase_rates)
            print(f"Cross-entropy for shape={wb_shape}, scale={wb_scale}: {cross_entropy}")

            KL = compute_kl_divergence_weibull(wb_shape, wb_scale, phase_rates)
            print(f"KL divergence for shape={wb_shape}, scale={wb_scale}: {KL}")

            arrival_distributions.append({
                "wb_shape": wb_shape,
                "wb_scale": wb_scale,
                "phase_rates": [float(l) for l in phase_rates]
            })

    # Step 3 onward identical
    try: 
        with open(output_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}
        
    if config:
        exp_nums = [int(k.split("_")[1]) for k in config.keys() if k.startswith("experiment_")]
        experiment_number = max(exp_nums) + 1
    else:
        experiment_number = 1

    config[f"experiment_{experiment_number}"] = {
        "arrival_distributions": arrival_distributions,
        "service_rates": input_config['service_rates'],
        "runs": input_config.get('runs', 100),
        "simulation_end": input_config.get('simulation_end', 1000),
        "varying_iql": input_config.get('varying_iql', False),
        "max_iql": input_config.get('max_iql', 0),
        "output_folder": input_config.get('output_folder', 'results/')
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert Weibull configs to Hypoexponential simulation configs."
    )
    
    parser.add_argument("--config_file",
                        "-c",
                        type=str,
                        required=True,
                        help="Relative path to the weibull configuration file for simulation")
    
    parser.add_argument("--experiment_number",
                        "-e",
                        type=int,
                        default=1,
                        help="Experiment number (e.g., 1)")

    args = parser.parse_args()

    query_workload = {
        "config_file": args.config_file,
        "experiment_number": args.experiment_number
    }

    # Always generate simulation config
    create_hypoexp_sim_config(query_workload)