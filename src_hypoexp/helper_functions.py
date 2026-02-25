from math import ceil, sqrt

def gamma_hypoexponential_approximation(alpha, theta): 
    """Function returns the approximation of the three
    hypoexponential parameters given the gamma distribution parameters."""
    """
    # get gamma parameters and check validity
        self.alpha = float(self.input_params.alpha)
        self.theta = float(self.input_params.theta)
        self.tau = self.alpha * self.theta  # mean of gamma distribution
        assert(self.alpha > 0 and self.theta > 0) # Check for validity of parameters
        
        # Returns a list of phase rates (length of this list <= n)
        self.phase_rates = helper_functions.gamma_hypoexponential_approximation(self.alpha, self.theta)
        self.num_phases = len(self.phase_rates) # TODO: Check why error
        
    """
    
    
    phase_rates = []
    tau = alpha * theta  # mean of gamma distribution
    # get total phases and store the list of phase rates
    # if alpha is very small handle case separately
    if abs(alpha - 1.0) < 1e-12:
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

        # make sure both are +ve
        if inv_lambda_third <= 0 or inv_lambda_second <= 0:
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
    
def get_optimal_subsampling_interval(self, max_queue_length: int):
    """
    return optimal subsampling interval delta
    reference: https://www.jstor.org/stable/24340803?seq=4
    """
    Q = self.generate_Q_matrix(max_queue_length)
    diagonal_entries = np.diagonal(Q)
    max_diag_abs = np.max(np.abs(diagonal_entries))
    delta = 1.0 / max_diag_abs

    return delta

def generate_Q_matrix(self, max_queue_length: int):
    """
    builds generator matrix Q for HypoExp/M/1 queue
    state = (phase, queue_length)
    """
    num_phases = self.num_phases
    service_rate = 1.0 / self.input_params.mean_service_time
    num_states = num_phases * (max_queue_length + 1)
    Q = np.zeros((num_states, num_states))

    for queue_len in range(max_queue_length + 1):
        for phase in range(1, num_phases + 1):

            state_idx = queue_len * num_phases + (phase - 1)

            # arrival transition: phase advance or arrival after final phase
            if phase < num_phases:
                # transition to next phase in same queue length
                next_state_idx = queue_len * num_phases + phase
                Q[state_idx, next_state_idx] = self.phase_rates[phase - 1]
            else:
                # customer arrives (after last phase)
                if queue_len < max_queue_length:
                    next_state_idx = (queue_len + 1) * num_phases
                    Q[state_idx, next_state_idx] = self.phase_rates[num_phases - 1]

            # service completed
            if queue_len > 0:
                next_state_idx = (queue_len - 1) * num_phases + (phase - 1)
                Q[state_idx, next_state_idx] = service_rate

            # set diagonal entries so rows add up to 0
            Q[state_idx, state_idx] = -np.sum(Q[state_idx, :])

    return Q

# TODO: Come back
def plot_distributions(self, num_samples=100000, bins=100):
    """
    Plot the Gamma distribution and the Hypoexponential approximation on the same graph
    """
    # gamma dist pdf
    shape = self.input_params.alpha
    scale = self.input_params.tau / shape  # mean = tau, so scale = tau / shape
    x = np.linspace(0, self.input_params.tau * 5, 1000)
    gamma_pdf = gamma.pdf(x, a=shape, scale=scale)

    # hypoexp samples
    hypo_samples = []
    for _ in range(num_samples):
        sample = sum(self.rng.exponential(1.0 / rate) for rate in self.phase_rates)
        hypo_samples.append(sample)
    hypo_samples = np.array(hypo_samples)

    # histogram to approximate PDF
    hist_vals, bin_edges = np.histogram(hypo_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # plot the distributions together
    plt.figure(figsize=(8, 5))
    plt.plot(x, gamma_pdf, label="Gamma PDF", color="blue", lw=2)
    plt.plot(bin_centers, hist_vals, label="Hypoexp Approx (samples)", color="orange", lw=2, alpha=0.7)
    plt.title(f"Gamma vs Hypoexponential Approximation (alpha={shape}, tau={self.input_params.tau})")
    plt.xlabel("Time")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_cross_entropy(self, num_gamma_samples=100000, bins=100, epsilon=1e-12):
    """
    Compute approximate cross-entropy H(Gamma || Hypoexp)
    """
    # get samples from gamma
    shape = self.input_params.alpha
    scale = self.input_params.tau / shape
    #gamma_samples = np.random.default_rng().gamma(shape, scale, num_gamma_samples)
    gamma_samples = self.rng.gamma(shape, scale, num_gamma_samples)

    # hypoexp samples and estimate pdf
    hypo_samples = np.array([sum(self.rng.exponential(1.0 / rate) for rate in self.phase_rates)
                            for _ in range(num_gamma_samples)])
    hist_vals, bin_edges = np.histogram(hypo_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # hypoexp PDF at Gamma samples and avoid log(0)
    hypo_pdf_at_gamma = np.interp(gamma_samples, bin_centers, hist_vals, left=epsilon, right=epsilon)
    hypo_pdf_at_gamma = np.maximum(hypo_pdf_at_gamma, epsilon)

    # get cross-entropy
    cross_entropy = -np.mean(np.log(hypo_pdf_at_gamma))
    return cross_entropy