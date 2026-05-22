# hypoexp_fit_pytorch.py
"""
Hypoexponential fitting (PyTorch)
- Train hypoexponential (k sequential exponential stages) to approximate a target distribution f
  by minimizing forward-KL (i.e. cross-entropy) estimated from samples from f,
  plus mean/variance regularization: Loss = -E_f[log g_lambda(X)] + alpha*mean_err^2 + beta*var_err^2

- Includes baseline: nonlinear moment programming (their paper method) using SLSQP.
- Evaluation metrics: estimated KL (forward), JSD (sample-based), 1-Wasserstein (scipy implementation).

Author: (adapted for your project)
"""

from curses import raw
from typing import Callable, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
import math

# -------------------------
# Utilities & numerics
# -------------------------
DTYPE = torch.float64  # high precision
EPS = 1e-12


def softplus(x: Tensor) -> Tensor:
    """Numerically stable softplus wrapper."""
    return F.softplus(x)


def build_T_matrix(lambdas: Tensor) -> Tensor:
    """
    Build k x k sub-generator matrix T for sequential phases.
    lambdas: (k,) positive rates tensor
    returns: (k,k) tensor
    """
    k = lambdas.shape[0]
    T = torch.zeros((k, k), dtype=DTYPE)
    # diagonal
    T.fill_diagonal_(-lambdas)
    # superdiagonal transitions (i -> i+1)
    for i in range(k - 1):
        T[i, i + 1] = lambdas[i]
    return T


def hypoexp_logpdf_torch(x: Tensor, lambdas: Tensor) -> Tensor:
    """
    Closed-form hypoexponential log-pdf (distinct rates).
    Vectorized + autograd friendly.
    """

    x = x.to(DTYPE)
    lambdas = lambdas.to(DTYPE)

    k = lambdas.shape[0]

    log_l = torch.log(lambdas)
    sum_log = torch.sum(log_l)

    diff = lambdas.unsqueeze(0) - lambdas.unsqueeze(1)
    diff = diff + torch.eye(k, dtype=DTYPE)  # avoid zero on diag temporarily

    log_alpha = []

    for i in range(k):

        num = sum_log - log_l[i]

        den = torch.sum(torch.log(torch.abs(diff[i]))) - torch.log(torch.abs(diff[i, i]))

        log_alpha.append(num - den)

    log_alpha = torch.stack(log_alpha)

    vals = log_alpha.unsqueeze(1) - lambdas.unsqueeze(1) * x

    logpdf = torch.logsumexp(vals, dim=0)

    return logpdf


def hypoexp_sample(rng: np.random.Generator, lambdas: np.ndarray, n: int) -> np.ndarray:
    """
    Sample n values from hypoexponential (sum of sequential exponentials).
    Implementation: sum of independent exponentials is equivalent to sequential phases.
    lambdas: numpy array (k,)
    returns: numpy array (n,)
    """
    k = lambdas.shape[0]
    # sample shape (n,k) where each column ~ Exp(rate=lambda_i)
    # Exp(rate=l) has scale = 1/l
    scales = 1.0 / (lambdas + 0.0)
    # Use numpy exponential parameterized by scale
    samp = rng.exponential(scale=scales[np.newaxis, :], size=(n, k))
    return samp.sum(axis=1)


def hypoexp_moments_np(lambdas: np.ndarray) -> Tuple[float, float, float]:
    """Return mean, variance, third central moment for sum of exponentials (np arrays)."""
    lambdas = np.asarray(lambdas, dtype=float)
    mean = np.sum(1.0 / lambdas)
    var = np.sum(1.0 / (lambdas ** 2))
    mu3 = 2.0 * np.sum(1.0 / (lambdas ** 3))
    return mean, var, mu3


def hypoexp_moments_torch(lambdas: Tensor) -> Tuple[Tensor, Tensor]:
    """Return mean and variance (torch Tensors) for use in regularizer."""
    mean = torch.sum(1.0 / lambdas)
    var = torch.sum(1.0 / (lambdas ** 2))
    return mean, var


# -------------------------
# Training objective: Forward KL (sample-based) + mean/variance regularizer
# -------------------------
def train_hypoexp_forwardKL(
    target_sampler: Callable[[int], np.ndarray],
    target_pdf: Optional[Callable[[np.ndarray], np.ndarray]],
    k: int = 4,
    steps: int = 3000,
    batch_size: int = 8000,
    alpha: float = 10.0,
    beta: float = 5.0,
    lr: float = 3e-4,
    rng_seed: int = 12345,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Fit hypoexponential to target by minimizing forward KL + moment regularizer.
    target_sampler: function(n) -> np.ndarray of n samples
    target_pdf: function(x_array) -> pdf values (np.ndarray). If None, we cannot compute KL analytically but training uses samples only.
    returns dict with 'lambdas', 'theta', 'history'
    """
    if device is None:
        device = torch.device('cpu')

    rng = np.random.default_rng(rng_seed)

    # -------------------------
    # sample once for initial moments
    # -------------------------
    X = target_sampler(batch_size)
    X_torch = torch.tensor(X, dtype=DTYPE, device=device)
    mu_f = float(np.mean(X))
    var_f = float(np.var(X))

    # parameterization: theta unconstrained -> lambda = softplus(theta) + tiny_min
    theta = torch.tensor(np.full((k,), math.log(max(1e-2, k / max(mu_f, 1e-6)))), 
                         dtype=DTYPE, requires_grad=True, device=device)
    opt = torch.optim.Adam([theta], lr=lr)

    history = {'loss': [], 'mean_err': [], 'var_err': [], 'lambdas': []}

    for step in range(steps):
        X = target_sampler(batch_size)
        X_torch = torch.tensor(X, dtype=DTYPE, device=device)
        
        # strictly increasing lambdas via cumulative softplus
        raw = softplus(theta)
        lambdas = torch.cumsum(raw, dim=0)

        logg = hypoexp_logpdf_torch(X_torch, lambdas)  # (batch_size,)
        loss_KL = -torch.mean(logg)  # forward KL up to const
        mu_lambda, var_lambda = hypoexp_moments_torch(lambdas)
        mean_err = (mu_f - mu_lambda)
        var_err = (var_f - var_lambda)
        loss_mom = alpha * (mean_err ** 2) + beta * (var_err ** 2)
        loss = loss_KL + loss_mom

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=10.0)
        opt.step()

        if step % 50 == 0 or step == steps - 1:
            cur_loss = float(loss.item())
            history['loss'].append(cur_loss)
            history['mean_err'].append(float(mean_err.item()))
            history['var_err'].append(float(var_err.item()))
            history['lambdas'].append(lambdas.detach().cpu().numpy())
            if verbose:
                print(f"[step {step:5d}] loss={cur_loss:.6g} KL={float(loss_KL.item()):.6g} mean_err={float(mean_err.item()):.6g} var_err={float(var_err.item()):.6g}")

    lambdas_final = softplus(theta).detach().cpu().numpy()
    return {'lambdas': np.sort(lambdas_final), 'theta': theta.detach().cpu().numpy(), 'history': history}


# -------------------------
# JSD estimator (sample-based)
# JSD(f,g) = 0.5 KL(f||m) + 0.5 KL(g||m), m = 0.5(f+g)
# We estimate using samples from f and g, and evaluating pdfs using target_pdf and hypoexp pdf.
# -------------------------
def estimate_jsd(
    target_pdf: Callable[[np.ndarray], np.ndarray],
    hypoexp_logpdf_fn: Callable[[np.ndarray], np.ndarray],
    target_sampler: Callable[[int], np.ndarray],
    lambdas: np.ndarray,
    n_samples: int = 5000,
    rng_seed: int = 12345,
) -> float:
    """
    Estimate JSD between target f and hypoexp g_lambda.
    target_pdf(xarr) returns numpy array pdfs
    hypoexp_logpdf_fn(xarr, lambdas) -> logpdf numpy
    """
    rng = np.random.default_rng(rng_seed)
    xf = target_sampler(n_samples)
    # sample from g: sum of exponentials
    xg = hypoexp_sample(rng, lambdas, n_samples)

    # evaluate densities
    pf_xf = np.clip(target_pdf(xf), a_min=EPS, a_max=None)
    # compute g and m for xf
    logg_xf = hypoexp_logpdf_fn(xf, lambdas)
    g_xf = np.exp(logg_xf)
    m_xf = 0.5 * (pf_xf + g_xf)
    # KL(f||m) estimate = mean_xf [ log pf_xf - log m_xf ]
    kl_f_m = np.mean(np.log(pf_xf) - np.log(np.clip(m_xf, a_min=EPS, a_max=None)))

    # for KL(g||m)
    pg_xg = np.clip(np.exp(hypoexp_logpdf_fn(xg, lambdas)), a_min=EPS, a_max=None)
    pf_xg = np.clip(target_pdf(xg), a_min=EPS, a_max=None)
    m_xg = 0.5 * (pf_xg + pg_xg)
    kl_g_m = np.mean(np.log(pg_xg) - np.log(np.clip(m_xg, a_min=EPS, a_max=None)))

    jsd = 0.5 * kl_f_m + 0.5 * kl_g_m
    return float(jsd)


# -------------------------
# Simple wrapper: hypoexp logpdf numpy (calls torch implementation)
# -------------------------
def hypoexp_logpdf_numpy(x: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """
    Evaluate hypoexponential pdf values using torch implementation for reliability.
    Returns logpdf numpy array.
    """
    x_t = torch.tensor(x, dtype=DTYPE)
    lamb_t = torch.tensor(lambdas, dtype=DTYPE)
    with torch.no_grad():
        logpdf = hypoexp_logpdf_torch(x_t, lamb_t)
    return logpdf.cpu().numpy()


# -------------------------
# Baseline: nonlinear moment programming (their method)
# We implement a stable version solving for lambda_i (positive) and slack/excess variables.
# We parameterize lambdas = exp(theta) for positivity in optimizer.
# -------------------------
def fit_hypoexp_moments_nlp(target_moments: Tuple[float, float, float], k: int = 4, h: Tuple[float, float, float] = (10.0, 5.0, 1.0)) -> np.ndarray:
    """
    Solve the nonlinear program from the paper:
    minimize sum_i h_i * (s_i + e_i)
    subject to:
      sum 1/lambda_i + s1 - e1 = mu
      sum 1/lambda_i^2 + s2 - e2 = var
      2 sum 1/lambda_i^3 + s3 - e3 = mu3
    lambdas > 0, s_i >= 0, e_i >= 0
    """
    mu, var, mu3 = target_moments
    h1, h2, h3 = h

    # initial guess: equal rates based on mean
    lambda_init = np.full(k, k / max(mu, 1e-9))
    theta_init = np.log(lambda_init)
    s_init = np.zeros(3)
    e_init = np.zeros(3)
    x0 = np.concatenate([theta_init, s_init, e_init])

    # bounds: theta unbounded in R, we can bound to prevent overflow
    bnds = []
    for _ in range(k):
        bnds.append((np.log(1e-9), np.log(1e9)))  # bounds on theta
    for _ in range(3):  # s
        bnds.append((0.0, None))
    for _ in range(3):  # e
        bnds.append((0.0, None))

    def unpack(x):
        theta = x[:k]
        s = x[k:k + 3]
        e = x[k + 3:k + 6]
        lambdas = np.exp(theta)
        return lambdas, s, e

    def objective(x):
        # penalize slack/excess with weights
        s = x[k:k + 3]
        e = x[k + 3:k + 6]
        return float(h1 * (s[0] + e[0]) + h2 * (s[1] + e[1]) + h3 * (s[2] + e[2]))

    # constraints return zero when satisfied
    def c_mean(x):
        lambdas, s, e = unpack(x)
        m1, m2, m3 = hypoexp_moments_np(lambdas)
        return float(m1 + s[0] - e[0] - mu)

    def c_var(x):
        lambdas, s, e = unpack(x)
        m1, m2, m3 = hypoexp_moments_np(lambdas)
        return float(m2 + s[1] - e[1] - var)

    def c_m3(x):
        lambdas, s, e = unpack(x)
        m1, m2, m3 = hypoexp_moments_np(lambdas)
        # note paper uses 2 sum 1/l^3 as mu3 central third moment
        return float(m3 + s[2] - e[2] - mu3)

    cons = [{'type': 'eq', 'fun': c_mean},
            {'type': 'eq', 'fun': c_var},
            {'type': 'eq', 'fun': c_m3}]

    res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
    if not res.success:
        print("Warning: moment NLP did not converge:", res.message)

    lambdas = np.exp(res.x[:k])
    # sort for presentation but note order may matter in sequential embedding
    return np.sort(lambdas)


# -------------------------
# Evaluation helper: compute forward KL (approx via samples)
# KL(f||g) = E_f[log f - log g] ; we can return the estimated value up to additive constant if f known
# If target_pdf available, compute full KL estimate; else only -E_f[log g] returned.
# -------------------------
def estimate_forward_KL(target_sampler: Callable[[int], np.ndarray],
                        target_pdf: Optional[Callable[[np.ndarray], np.ndarray]],
                        lambdas: np.ndarray,
                        n_samples: int = 10000) -> float:
    xs = target_sampler(n_samples)
    logg = hypoexp_logpdf_numpy(xs, lambdas)
    if target_pdf is not None:
        pf = np.clip(target_pdf(xs), a_min=EPS, a_max=None)
        kl_est = np.mean(np.log(pf) - logg)
        return float(kl_est)
    else:
        # return -E_f[log g] (not centered)
        return float(-np.mean(logg))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from scipy.stats import gamma

    rng = np.random.default_rng(42)

    shape = 4.3
    scale = 1.2
    k = 4

    def gamma_sampler(n):
        return rng.gamma(shape, scale, size=n)

    def gamma_pdf(x):
        return gamma.pdf(x, a=shape, scale=scale)

    print("\n===== TRAINING HYPOEXP FIT =====\n")

    res = train_hypoexp_forwardKL(
        target_sampler=gamma_sampler,
        target_pdf=gamma_pdf,
        k=k,
        steps=1500,
        batch_size=6000,
        alpha=200.0,
        beta=50.0,
        lr=1e-4,
        verbose=True,
    )

    lambdas = res['lambdas']

    print("\nFitted lambdas:", lambdas)

    # ===== Moments comparison =====

    X_big = gamma_sampler(200000)

    mu_f = np.mean(X_big)
    var_f = np.var(X_big)
    mu3_f = np.mean((X_big - mu_f) ** 3)

    mu_g, var_g, mu3_g = hypoexp_moments_np(lambdas)

    print("\n===== MOMENT COMPARISON =====")
    print("Target mean:", mu_f)
    print("Hypoexp mean:", mu_g)

    print("\nTarget var:", var_f)
    print("Hypoexp var:", var_g)

    print("\nTarget 3rd moment:", mu3_f)
    print("Hypoexp 3rd moment:", mu3_g)

    # ===== Metrics =====

    print("\n===== METRICS =====")

    KL_est = estimate_forward_KL(
        target_sampler=gamma_sampler,
        target_pdf=gamma_pdf,
        lambdas=lambdas,
        n_samples=12000,
    )

    print("Estimated Forward KL:", KL_est)

    jsd_val = estimate_jsd(
        target_pdf=gamma_pdf,
        hypoexp_logpdf_fn=hypoexp_logpdf_numpy,
        target_sampler=gamma_sampler,
        lambdas=lambdas,
        n_samples=6000,
    )

    print("Estimated JSD:", jsd_val)

    sample_f = gamma_sampler(12000)
    sample_g = hypoexp_sample(rng, lambdas, 12000)

    w1 = wasserstein_distance(sample_f, sample_g)

    print("Estimated Wasserstein:", w1)

    # ===== Plot PDF comparison =====

    xs = np.linspace(0, np.percentile(sample_f, 99.5), 400)

    pdf_f = gamma_pdf(xs)
    pdf_g = np.exp(hypoexp_logpdf_numpy(xs, lambdas))

    plt.figure(figsize=(8,5))
    plt.plot(xs, pdf_f, label="Gamma target", linewidth=3)
    plt.plot(xs, pdf_g, '--', label="Hypoexp fit", linewidth=3)
    plt.title("PDF comparison")
    plt.legend()
    plt.show()

    # ===== Histogram comparison =====

    plt.figure(figsize=(8,5))
    plt.hist(sample_f, bins=80, density=True, alpha=0.5, label="Gamma samples")
    plt.hist(sample_g, bins=80, density=True, alpha=0.5, label="Hypoexp samples")
    plt.legend()
    plt.title("Sample distribution comparison")
    plt.show()

    print("\n===== NONLINEAR MOMENT PROGRAMMING BASELINE =====")

    target_moments = (mu_f, var_f, mu3_f)

    lambdas_nlp = fit_hypoexp_moments_nlp(
        target_moments=target_moments,
        k=k,
        h=(10.0, 5.0, 1.0)
    )

    print("NLP lambdas:", lambdas_nlp)

    mu_g_nlp, var_g_nlp, mu3_g_nlp = hypoexp_moments_np(lambdas_nlp)

    print("\nNLP mean:", mu_g_nlp)
    print("NLP var:", var_g_nlp)
    print("NLP 3rd:", mu3_g_nlp)

    KL_nlp = estimate_forward_KL(
        target_sampler=gamma_sampler,
        target_pdf=gamma_pdf,
        lambdas=lambdas_nlp,
        n_samples=12000
    )

    print("NLP Forward KL:", KL_nlp)

    jsd_nlp = estimate_jsd(
        target_pdf=gamma_pdf,
        hypoexp_logpdf_fn=hypoexp_logpdf_numpy,
        target_sampler=gamma_sampler,
        lambdas=lambdas_nlp,
        n_samples=6000
    )

    print("NLP JSD:", jsd_nlp)

    sample_g_nlp = hypoexp_sample(rng, lambdas_nlp, 12000)
    w1_nlp = wasserstein_distance(sample_f, sample_g_nlp)

    print("NLP Wasserstein:", w1_nlp)

    pdf_g_nlp = np.exp(hypoexp_logpdf_numpy(xs, lambdas_nlp))

    plt.figure(figsize=(8,5))
    plt.plot(xs, pdf_f, label="Gamma", linewidth=3)
    plt.plot(xs, pdf_g, '--', label="Forward KL fit", linewidth=3)
    plt.plot(xs, pdf_g_nlp, ':', label="Moment NLP", linewidth=3)
    plt.legend()
    plt.title("PDF comparison (All methods)")
    plt.show()

    print("\n===== DONE =====")
