
# hypoexp_fitting.py

from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch


def hypoexp_moments(lambdas):
    """
    defines first three hypoexpnential moments
    """
    lambdas = np.asarray(lambdas)

    mean = np.sum(1 / lambdas)
    var = np.sum(1 / (lambdas ** 2))
    mu3 = 2 * np.sum(1 / (lambdas ** 3))

    return mean, var, mu3


def hypoexp_logpdf_numpy(x, lambdas):
    """
    defines log prob distribution for the hypoexp function
    """

    x = np.asarray(x)
    lambdas = np.asarray(lambdas)
    k = len(lambdas)
    log_alpha = np.zeros(k)

    for i in range(k):
        num = np.sum(np.log(lambdas)) - np.log(lambdas[i])
        diff = np.abs(lambdas - lambdas[i]) + 1e-12
        diff[i] = 1
        den = np.sum(np.log(diff))
        log_alpha[i] = num - den

    vals = []
    for i in range(k):
        vals.append(log_alpha[i] - lambdas[i] * x)

    vals = np.vstack(vals)
    m = np.max(vals, axis=0)
    logpdf = m + np.log(np.sum(np.exp(vals - m), axis=0))

    return logpdf


############################################################
# NLP moment fitting (paper method)
############################################################

def fit_hypoexp_moments_nlp(target_moments, k=4):

    mu, var, mu3 = target_moments

    lambda_init = np.ones(k) * k / mu
    theta_init = np.log(lambda_init)

    s_init = np.zeros(3)
    e_init = np.zeros(3)

    x0 = np.concatenate([theta_init, s_init, e_init])


    bounds = []

    for i in range(k):
        bounds.append((np.log(1e-9), np.log(1e9)))

    for i in range(6):
        bounds.append((0, None))


    def unpack(x):

        theta = x[:k]
        s = x[k:k+3]
        e = x[k+3:k+6]

        lambdas = np.exp(theta)

        return lambdas, s, e


    def objective(x):

        lambdas, s, e = unpack(x)

        return 10*(s[0]+e[0]) + 5*(s[1]+e[1]) + 1*(s[2]+e[2])


    def constraint_mean(x):

        lambdas, s, e = unpack(x)

        m1, m2, m3 = hypoexp_moments(lambdas)

        return m1 + s[0] - e[0] - mu


    def constraint_var(x):

        lambdas, s, e = unpack(x)

        m1, m2, m3 = hypoexp_moments(lambdas)

        return m2 + s[1] - e[1] - var


    def constraint_m3(x):

        lambdas, s, e = unpack(x)

        m1, m2, m3 = hypoexp_moments(lambdas)

        return m3 + s[2] - e[2] - mu3


    cons = [
        {'type':'eq','fun':constraint_mean},
        {'type':'eq','fun':constraint_var},
        {'type':'eq','fun':constraint_m3}
    ]


    res = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=cons,
        method="SLSQP"
    )


    lambdas = np.exp(res.x[:k])

    return np.sort(lambdas)


############################################################
# Hypoexp log pdf (torch)
############################################################

def hypoexp_logpdf_torch(x, lambdas):

    k = lambdas.shape[0]

    log_l = torch.log(lambdas)

    sum_log = torch.sum(log_l)

    diff = torch.abs(lambdas.unsqueeze(0) - lambdas.unsqueeze(1)) + 1e-12

    log_alpha = []

    for i in range(k):

        num = sum_log - log_l[i]

        den = torch.sum(torch.log(diff[i])) - torch.log(diff[i,i])

        log_alpha.append(num - den)

    log_alpha = torch.stack(log_alpha)

    vals = log_alpha.unsqueeze(1) - lambdas.unsqueeze(1)*x

    logpdf = torch.logsumexp(vals, dim=0)

    return logpdf


############################################################
# KL / Cross entropy fitting
############################################################

def fit_hypoexp_cross_entropy(target_sampler, k=4, steps=2000):

    X = target_sampler(5000)

    X = torch.tensor(X, dtype=torch.float64)

    theta = torch.zeros(k, dtype=torch.float64, requires_grad=True)

    opt = torch.optim.Adam([theta], lr=1e-3)


    for step in range(steps):

        lambdas = torch.exp(theta)

        logg = hypoexp_logpdf_torch(X, lambdas)

        loss = -torch.mean(logg)

        opt.zero_grad()
        loss.backward()
        opt.step()


    lambdas = torch.exp(theta).detach().numpy()

    return np.sort(lambdas)


############################################################
# Main API
############################################################

def fit_hypoexp(target, method="kl", k=4):

    sampler = target["sampler"]

    if method == "moments":

        if "moments" in target:
            m = target["moments"]()
        else:

            X = sampler(100000)

            mu = np.mean(X)
            var = np.var(X)
            mu3 = np.mean((X-mu)**3)

            m = (mu,var,mu3)

        return fit_hypoexp_moments_nlp(m,k)


    if method == "kl":

        return fit_hypoexp_cross_entropy(sampler,k)


def get_optimal_subsampling_interval(max_queue_length, arrival_phase_rates, service_rate):
    """
    return optimal subsampling interval delta
    reference: https://www.jstor.org/stable/24340803?seq=4
    """
    Q = generate_Q_matrix(max_queue_length, arrival_phase_rates, service_rate)
    diagonal_entries = np.diagonal(Q)
    max_diag_abs = np.max(np.abs(diagonal_entries))
    delta = 1.0 / max_diag_abs

    return delta


def generate_Q_matrix(max_queue_length, arrival_phase_rates, service_rate):
    """
    Generator matrix for HypoExp/M/1 queue.

    Arrival process = Hypoexponential with phase rates λ1..λk
    Service process = Exponential with rate μ

    State = (arrival_phase, queue_length)

    arrival_phase: 1..k
    queue_length:  0..max_queue_length
    """
    k = len(arrival_phase_rates)
    num_states = (max_queue_length + 1) * k
    Q = np.zeros((num_states, num_states))

    for queue_len in range(max_queue_length + 1):
        for phase in range(1, k + 1):

            state = queue_len * k + (phase - 1)

            # Arrival phase transitions
            if phase < k:
                next_state = queue_len * k + phase
                Q[state, next_state] += arrival_phase_rates[phase - 1]

            else:
                # final phase -> customer arrives
                if queue_len < max_queue_length:
                    next_state = (queue_len + 1) * k + 0
                    Q[state, next_state] += arrival_phase_rates[k - 1]

            # Service completion
            if queue_len > 0:
                next_state = (queue_len - 1) * k + (phase - 1)
                Q[state, next_state] += service_rate

            # Diagonal entry
            Q[state, state] = -np.sum(Q[state])

    return Q