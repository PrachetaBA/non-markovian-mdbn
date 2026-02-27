"""
Script to simulate a G/M/1 distribution using Gamma as the arrival process.
"""
import os
from math import ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from pathlib import Path
import yaml


def gamma_hypoexponential_approximation(alpha, theta): 
    """
    Returns an approximation of the list of hypoexponential parameters given the gamma distribution parameters.
    reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC9850366/#sec3
    """    
    assert(alpha > 0 and theta > 0), "Alpha and Theta must be positive for a valid Gamma distribution"
    phase_rates = []
    tau = alpha * theta  # mean of gamma distribution
    eps = 1e-12

    # get total phases and store the list of phase rates
    # if alpha is very small handle case separately
    if abs(alpha - 1.0) < eps:
        phase_rates = [1.0 / tau]

    else:
        n = max(ceil(alpha), 2)

        # rate for first n-2 phases
        lambda_first = n / tau

        # calculate final two rates lambda_second and lambda_third using paper formulas
        inside = (n / (2.0 * alpha)) * (n - alpha)
        if inside < 0: # make sure it is not negative
            inside = 0.0
        s = sqrt(inside)
        inv_lambda_second = (tau / n) * (1.0 + s)
        inv_lambda_third = (tau / n) * (1.0 - s)

        # handle numeric unstability
        if inv_lambda_third <= eps or inv_lambda_second <= eps:
            print(f"Invalid hypoexp parameters for alpha={alpha}, tau={tau}, got non-positive phase inverse(s)")
            return phase_rates

        lambda_second = 1.0 / inv_lambda_second
        lambda_third = 1.0 / inv_lambda_third

        # phase rates list: first n-2 are lambda_first, last two are lambda_second and lambda_third
        if n > 2:
            phase_rates = [lambda_first] * (n - 2) + [lambda_second, lambda_third]
        else:
            phase_rates = [lambda_second, lambda_third]

    return phase_rates


def plot_gamma_vs_hypoexp(alpha, theta, phase_rates, experiment_no, num_samples=100000, bins=100):
    """
    Plot the Gamma distribution and the Hypoexponential approximation on the same graph
    """
    # gamma dist pdf
    shape = alpha
    scale = theta
    x = np.linspace(0, alpha * theta * 5, 1000)
    gamma_pdf = gamma.pdf(x, a=shape, scale=scale)

    # hypoexp samples and approximate pdf
    hypo_samples = np.sum([np.random.exponential(1.0 / rate, size=num_samples) for rate in phase_rates], axis=0)
    hist_vals, bin_edges = np.histogram(hypo_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # plot the distributions together and save the figure
    plt.figure(figsize=(8, 5))
    plt.plot(x, gamma_pdf, label="Gamma PDF", color="blue", lw=2)
    plt.plot(bin_centers, hist_vals, label="Hypoexp Approx (samples)", color="orange", lw=2, alpha=0.7)
    plt.title(f"Gamma vs Hypoexponential Approximation (alpha={alpha}, theta={theta})")
    plt.xlabel("Time")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f'figures/gamma_vs_hypoexp_exp{experiment_no}_alpha_{alpha}_theta_{theta}.png')


def compute_cross_entropy(alpha, theta, phase_rates, num_gamma_samples=100000, bins=100, epsilon=1e-12):
    """
    Compute approximate cross-entropy H(Gamma || Hypoexp)
    """
    # get samples from gamma
    shape = alpha
    scale = theta
    gamma_samples = np.random.gamma(shape, scale, num_gamma_samples)

    # hypoexp samples and estimate pdf
    hypo_samples = np.sum([np.random.exponential(1.0 / rate, size=num_gamma_samples) for rate in phase_rates], axis=0)
    hist_vals, bin_edges = np.histogram(hypo_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # hypoexp PDF at Gamma samples, avoid log(0) and get CE
    hypo_pdf_at_gamma = np.interp(gamma_samples, bin_centers, hist_vals, left=epsilon, right=epsilon)
    hypo_pdf_at_gamma = np.maximum(hypo_pdf_at_gamma, epsilon)
    cross_entropy = -np.mean(np.log(hypo_pdf_at_gamma))

    return cross_entropy


def create_hypoexp_sim_config(query_workload): 
    """Function creates the HypoExp/M/1 simulation config for a particular query workload. 
    
    Query workload: 
        Gamma parameters: 
            alphas = []
            thetas = []
        Service rates: 
            mus = []
    """
    # Step 1: Resolve paths and extract the config from the corresponding experiment_number in the query_workload 
    project_root = Path(__file__).resolve().parents[1]
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    input_config_file = project_root / query_workload.get('simulator_config_file', None)
    input_experiment_number  = query_workload.get('experiment_number', None)
    output_path = config_dir / query_workload.get('output_file', f'hypoexp_simulator.yaml')
    with open(input_config_file, "r") as f:
        input_config_data = yaml.safe_load(f)
    input_config = input_config_data.get(f"experiment_{input_experiment_number}", None)
    
    # Step 2: Approx gamma with hypoexp, plot both distributions and compute CE
    phase_rates_list = []
    for alpha in input_config['alphas']:
        for theta in input_config['thetas']:
            phase_rates = gamma_hypoexponential_approximation(alpha, theta)
            plot_gamma_vs_hypoexp(alpha, theta, phase_rates, input_experiment_number)
            cross_entropy = compute_cross_entropy(alpha, theta, phase_rates)
            print(f"Cross-entropy for alpha={alpha}, theta={theta}: {cross_entropy}")
            phase_rates_list.append(phase_rates) 
     
    # Step 3: Load existing config
    try: 
        with open(output_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}
        
    # Step 4: Get next experiment number
    if config:
        exp_nums = [int(k.split("_")[1]) for k in config.keys() if k.startswith("experiment_")]
        experiment_number = max(exp_nums) + 1
    else:
        experiment_number = 1

    # Step 5: Append top the yaml file
    config[f"experiment_{experiment_number}"] = {
        "phase_rates": phase_rates_list,
        "service_rates": input_config['service_rates'],
        "runs": input_config.get('runs', 100),
        "simulation_end": input_config.get('simulation_end', 1000),
        "varying_iql": input_config.get('varying_iql', False),
        "max_iql": input_config.get('max_iql', 0),
        "output_folder": input_config.get('output_folder', 'results/')
    }

    # Step 6: Write back to the config file
    with open(output_path, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":
    query_workload = {
        'simulator_config_file': 'config/gamma_simulator.yaml',
        'experiment_number': 1, 
    }
    create_hypoexp_sim_config(query_workload)