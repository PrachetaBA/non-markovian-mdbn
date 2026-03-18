import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma as sp_gamma, beta as sp_beta, weibull_min as sp_weibull, gaussian_kde, entropy
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# -------------------------
# Core utilities
# -------------------------
def hypoexp_moments_np(lambdas: np.ndarray):
    """Return mean, variance, and 3rd-proxy (2 * sum 1/lambda^3)."""
    m1 = np.sum(1.0 / lambdas)
    m2 = np.sum(1.0 / lambdas**2)
    m3 = 2.0 * np.sum(1.0 / lambdas**3)
    return float(m1), float(m2), float(m3)

def hypoexp_pdf(x: np.ndarray, lambdas: np.ndarray, eps: float = 1e-12):
    """
    Stabilized hypoexponential PDF. If all lambdas are nearly equal -> use Erlang (gamma) formula.
    x can be scalar or 1-d array.
    """
    x = np.asarray(x)
    k = len(lambdas)
    # EDk fallback when lambdas are nearly equal
    if np.max(lambdas) - np.min(lambdas) < 1e-8:
        lam = float(np.mean(lambdas))
        # Erlang(k, rate=lam): pdf = lam^k * x^(k-1) * exp(-lam*x) / (k-1)!
        # Use scipy's gamma pdf with shape=k and scale=1/lam
        return sp_gamma.pdf(x, a=k, scale=1.0/lam)
    # General unequal-rates formula (stable)
    pdf = np.zeros_like(x, dtype=float)
    for i in range(k):
        coef = 1.0
        lam_i = lambdas[i]
        for j in range(k):
            if j == i:
                continue
            diff = lambdas[j] - lam_i
            # avoid tiny denominators
            if np.abs(diff) < eps:
                diff = np.sign(diff) * eps if np.sign(diff) != 0 else eps
            coef *= lambdas[j] / diff
        pdf += coef * lam_i * np.exp(-lam_i * x)
    # numerical safeguards
    pdf = np.maximum(pdf, 0.0)
    return pdf

# -------------------------
# Moment-based NLP fit (Elmagrhraby)
# -------------------------
def fit_hypoexp_moments_nlp(target_moments, k=4, h=(10.0,5.0,1.0),
                            lambda_bounds=(1e-3, 1e3), verbose=False):
    """
    Fit lambdas (rates) by solving the Elmagrhraby NLP with slack/excess variables.
    Returns sorted lambdas (1D numpy array).
    """
    mu, var, mu3 = target_moments
    h1, h2, h3 = h

    # initial guess: equal rates -> lambda = k / mu
    lambda_init = np.full(k, k / max(mu, 1e-9))
    theta_init = np.log(lambda_init)
    s_init = np.zeros(3)
    e_init = np.zeros(3)
    x0 = np.concatenate([theta_init, s_init, e_init])

    lb_theta = np.log(lambda_bounds[0])
    ub_theta = np.log(lambda_bounds[1])
    bnds = [(lb_theta, ub_theta)] * k + [(0.0, None)]*3 + [(0.0, None)]*3

    def unpack(x):
        theta = x[:k]
        s = x[k:k+3]
        e = x[k+3:k+6]
        return np.exp(theta), s, e

    def objective(x):
        s = x[k:k+3]
        e = x[k+3:k+6]
        return float(h1*(s[0]+e[0]) + h2*(s[1]+e[1]) + h3*(s[2]+e[2]))

    def c_mean(x):
        lambdas, s, e = unpack(x)
        m1, _, _ = hypoexp_moments_np(lambdas)
        return float(m1 + s[0] - e[0] - mu)

    def c_var(x):
        lambdas, s, e = unpack(x)
        _, m2, _ = hypoexp_moments_np(lambdas)
        return float(m2 + s[1] - e[1] - var)

    def c_m3(x):
        lambdas, s, e = unpack(x)
        _, _, m3 = hypoexp_moments_np(lambdas)
        return float(m3 + s[2] - e[2] - mu3)

    cons = [{'type':'eq', 'fun': c_mean},
            {'type':'eq', 'fun': c_var},
            {'type':'eq', 'fun': c_m3}]

    res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
                   options={'maxiter':2000, 'ftol':1e-10})

    if not res.success and verbose:
        print("NLP warning:", res.message)

    # Extract lambdas (on success or not)
    lambdas = np.exp(res.x[:k])
    # EDk detection: if nearly equal, collapse to mean
    if np.max(lambdas) - np.min(lambdas) < 1e-8:
        lambdas[:] = np.mean(lambdas)
    return np.sort(lambdas)

# -------------------------
# Metrics: cross-entropy (using samples) and JSD (using grid)
# -------------------------
def cross_entropy_samples(samples: np.ndarray, lambdas: np.ndarray, min_eps=1e-12):
    """
    Estimate cross-entropy H(P, Q) = -E_{x~P}[ log Q(x) ] using sample points from P.
    samples: array of observed samples from P
    lambdas: hypoexp rates (Q)
    """
    q_vals = hypoexp_pdf(samples, lambdas)
    q_vals = np.clip(q_vals, min_eps, None)
    return -float(np.mean(np.log(q_vals)))

def jensen_shannon_grid(samples: np.ndarray, lambdas: np.ndarray, grid_points=1000, min_eps=1e-12):
    """
    Compute JS divergence between target P (estimated by KDE from samples) and hypoexp Q on a grid.
    Returns scalar JSD (nat units).
    """
    x = np.linspace(0.0, np.percentile(samples, 99.5), grid_points)
    dx = x[1] - x[0]
    # target PDF via KDE
    kde = gaussian_kde(samples)
    p = kde(x)
    q = hypoexp_pdf(x, lambdas)
    # clip and form discrete pmf via p * dx
    p = np.clip(p, min_eps, None)
    q = np.clip(q, min_eps, None)
    p_pmf = p * dx
    q_pmf = q * dx
    # renormalize to sum to 1
    p_pmf /= np.sum(p_pmf)
    q_pmf /= np.sum(q_pmf)
    m = 0.5 * (p_pmf + q_pmf)
    # entropy returns KL divergence when passed (p, m)
    jsd = 0.5 * (entropy(p_pmf, m) + entropy(q_pmf, m))
    return float(jsd)

# -------------------------
# k-sweep routine: fit for k in [k_min, k_max], pick best by JSD
# -------------------------
def sweep_k_and_select(distribution: str, params: dict, k_min=2, k_max=8, h=(10,5,1),
                       n_samples=100000, seed=0, lambda_bounds=(1e-3,1e3), verbose=False):
    rng = np.random.default_rng(seed)
    # sample generator mapping
    if distribution == "gamma":
        def sample_fn(n): return sp_gamma.rvs(params["shape"], scale=params["scale"], size=n, random_state=rng)
    elif distribution == "beta":
        def sample_fn(n): return sp_beta.rvs(params["a"], params["b"], size=n, random_state=rng)
    elif distribution == "weibull":
        def sample_fn(n): return sp_weibull.rvs(params["c"], scale=params["scale"], size=n, random_state=rng)
    elif distribution == "lognormal":
        # params should be {"s": sigma, "scale": exp(mu)}
        def sample_fn(n): return lognorm.rvs(s=params["s"], scale=params["scale"], size=n, random_state=rng)
    else:
        raise ValueError("Unsupported distribution")

    samples = sample_fn(n_samples)
    mu, var, mu3 = hypoexp_moments_np(np.array([1.0]))  # dummy to get signature; we'll compute moments properly
    # compute empirical moments based on samples
    mu = float(np.mean(samples))
    var = float(np.var(samples))
    mu3 = 2.0 * float(np.mean((samples - mu)**3))
    target_moments = (mu, var, mu3)
    if verbose:
        print(f"Target moments: mean={mu:.4f}, var={var:.4f}, 3rd-proxy={mu3:.4f}")

    results = []
    for k in range(k_min, k_max + 1):
        try:
            lambdas = fit_hypoexp_moments_nlp(target_moments, k=k, h=h, lambda_bounds=lambda_bounds, verbose=verbose)
            # evaluate metrics
            ce = cross_entropy_samples(samples, lambdas)
            jsd = jensen_shannon_grid(samples, lambdas)
            results.append((k, lambdas, ce, jsd))
            if verbose:
                print(f"k={k:2d}  jsd={jsd:.6e}  ce={ce:.6e}  lambdas={np.round(lambdas,6)}")
        except Exception as exc:
            if verbose:
                print(f"k={k} failed: {exc}")
            continue

    if not results:
        raise RuntimeError("All fits failed in sweep.")

    # choose best by minimal JSD (tie-breaker: minimal CE)
    results.sort(key=lambda t: (t[3], t[2]))
    best_k, best_lambdas, best_ce, best_jsd = results[0]
    return {
        "samples": samples,
        "target_moments": target_moments,
        "results": results,
        "best": {
            "k": best_k,
            "lambdas": best_lambdas,
            "ce": best_ce,
            "jsd": best_jsd
        }
    }

# -------------------------
# Plotting
# -------------------------
def plot_best_fit(info: dict, distribution_name: str):
    samples = info["samples"]
    best = info["best"]
    lambdas = best["lambdas"]
    k = best["k"]

    x = np.linspace(0.0, np.percentile(samples, 99.5), 1000)
    kde = gaussian_kde(samples)
    pdf_target = kde(x)
    pdf_hypo = hypoexp_pdf(x, lambdas)

    plt.figure(figsize=(8,5))
    plt.plot(x, pdf_target, label=f"Target PDF (KDE)", lw=2.4)
    plt.plot(x, pdf_hypo, linestyle="--", label=f"Hypoexp (k={k})", lw=2.4)
    plt.title(f"{distribution_name.capitalize()} — best hypoexp fit (k={k})", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(alpha=0.4, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------
# Simple CLI-style main
# -------------------------
if __name__ == "__main__":
    # stable example params (recommended)
    examples = {
        "gamma": {"shape": 3.0, "scale": 1.0},
        "beta": {"a": 2.5, "b": 5.0},
        "weibull": {"c": 1.5, "scale": 0.5},
        "lognormal": {"s": 0.5, "scale": np.exp(1.0)},  # s = sigma, scale = exp(mu)
    }

    for dist_name, params in examples.items():
        print("\n" + "="*70)
        print(f"Fitting distribution: {dist_name}  params={params}")
        info = sweep_k_and_select(dist_name, params, k_min=2, k_max=8, h=(10,5,1),
                                  n_samples=50000, seed=42, lambda_bounds=(1e-3, 1e3),
                                  verbose=True)
        best = info["best"]
        print(f"-> best k = {best['k']}, JSD = {best['jsd']:.6e}, CE ≈ {best['ce']:.6e}")
        print(f"-> lambdas = {np.round(best['lambdas'],6)}")

        # Compute hypoexp moments for the selected lambdas
        h_mom = hypoexp_moments_np(best['lambdas'])
        print(f"-> Hypoexp moments: mean={h_mom[0]:.4f}, var={h_mom[1]:.4f}, 3rd-proxy={h_mom[2]:.4f}")

        # Compare to target moments
        tm = info["target_moments"]
        print(f"-> Target moments: mean={tm[0]:.4f}, var={tm[1]:.4f}, 3rd-proxy={tm[2]:.4f}")

        plot_best_fit(info, dist_name)

  