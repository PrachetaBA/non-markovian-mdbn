"""
Script to simulate a G/M/1 distribution using Beta as the arrival process.
"""
import argparse
import json
import os
from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from pathlib import Path
import yaml
from approx_hypoexp_NLP import sweep_k_and_select, hypoexp_pdf, hypoexp_moments_np


def beta_hypoexponential_approximation(alpha, beta_param,
                                       k_min=2,
                                       k_max=8,
                                       n_samples=50000,
                                       seed=42):
    """
    Returns an approximation of the list of hypoexponential phase rates
    given the Beta distribution parameters.

    Uses moment-based NLP phase-type fitting with k-sweep selection.

    Parameters
    ----------
    alpha : float
        Alpha (shape) parameter of Beta distribution.
    beta_param : float
        Beta (shape) parameter of Beta distribution.

    Returns
    -------
    phase_rates : list[float]
        List of hypoexponential phase rates (lambdas).
    """

    assert(alpha > 0), "Alpha must be positive for a valid Beta distribution"
    assert(beta_param > 0), "Beta must be positive for a valid Beta distribution"

    phase_rates = []

    try:
        # Map to scipy beta parameters
        params = {
            "a": alpha,
            "b": beta_param,
            "scale": 1.0  # default support [0,1]
        }

        info = sweep_k_and_select(
            distribution="beta",
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
            print(f"Beta hypoexp fit failed for alpha={alpha}, beta={beta_param}")
            return phase_rates

        lambdas = best.get("lambdas", None)

        if lambdas is None or len(lambdas) == 0:
            print(f"Invalid hypoexp parameters for alpha={alpha}, beta={beta_param}")
            return phase_rates

        phase_rates = list(lambdas)

    except Exception as e:
        print(f"Beta hypoexp approximation failed for alpha={alpha}, beta={beta_param}: {e}")

    return phase_rates


def plot_beta_vs_hypoexp(alpha, beta_param, phase_rates, experiment_no, num_samples=100000):
    """
    Plot the Beta distribution and the Hypoexponential approximation on the same graph
    """
    # sample from Beta
    samples = np.random.beta(alpha, beta_param, size=num_samples)

    # plotting window (important for tail)
    x_max = np.percentile(samples, 99.5)
    x = np.linspace(0, x_max, 1000)

    # Beta pdf
    beta_pdf = beta_dist.pdf(x, a=alpha, b=beta_param, scale=1.0)

    # exact hypoexp pdf
    hypo_pdf = hypoexp_pdf(x, np.array(phase_rates))

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(x, beta_pdf, label="Beta", color="#0072B2", lw=2)       # Blue
    plt.plot(x, hypo_pdf, label="GED", color="#D62E00", lw=2)         # Red

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
    plt.savefig(f'figures/beta_vs_hypoexp_exp{experiment_no}_alpha_{alpha}_beta_{beta_param}.pdf')
    plt.close()


def compute_cross_entropy_beta(alpha, beta_param, phase_rates, num_samples=100000, epsilon=1e-12):
    """
    Compute approximate cross-entropy H(Beta || Hypoexp)
    """
    # sample from beta
    samples = np.random.beta(alpha, beta_param, size=num_samples)

    # evaluate hypoexp pdf
    hypo_pdf_vals = hypoexp_pdf(samples, np.array(phase_rates))

    # avoid log(0)
    hypo_pdf_vals = np.maximum(hypo_pdf_vals, epsilon)

    cross_entropy = -np.mean(np.log(hypo_pdf_vals))

    return cross_entropy
    

def compute_kl_divergence_beta(alpha, beta_param, phase_rates, num_samples=100000, epsilon=1e-12):
    """
    Compute approximate KL divergence KL(Beta || Hypoexp)

    KL(P || Q) = E_P[ log P(x) - log Q(x) ]
    where P is the Beta distribution and Q is the Hypoexponential approximation.
    """

    # sample from beta
    samples = np.random.beta(alpha, beta_param, size=num_samples)

    # evaluate true beta pdf
    p_vals = beta_dist.pdf(samples, a=alpha, b=beta_param, scale=1.0)

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
    Beta-based version.
    """
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    input_config_file = project_root / query_workload.get('config_file', None)
    input_experiment_number  = query_workload.get('experiment_number', None)
    output_path = config_dir / query_workload.get('output_file', f'hypoexp_beta_simulator.yaml')

    with open(input_config_file, "r") as f:
        input_config_data = yaml.safe_load(f)
    input_config = input_config_data.get(f"experiment_{input_experiment_number}", None)

    # Step 2: Approx beta with hypoexp, plot both distributions
    arrival_distributions = []
    for alpha in input_config['B_ALPHAS']:
        for beta_param in input_config['B_BETAS']:
            phase_rates = beta_hypoexponential_approximation(alpha, beta_param, k_min=2, k_max=5)
            plot_beta_vs_hypoexp(alpha, beta_param, phase_rates, input_experiment_number)
            cross_entropy = compute_cross_entropy_beta(alpha, beta_param, phase_rates)
            print(f"Cross-entropy for alpha={alpha}, beta={beta_param}: {cross_entropy}")

            KL = compute_kl_divergence_beta(alpha, beta_param, phase_rates)
            print(f"KL divergence for alpha={alpha}, beta={beta_param}: {KL}")

            arrival_distributions.append({
                "alpha": alpha,
                "beta": beta_param,
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
        description="Convert Beta configs to Hypoexponential simulation configs."
    )
    
    parser.add_argument("--config_file",
                        "-c",
                        type=str,
                        required=True,
                        help="Relative path to the Beta configuration file for simulation")
    
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