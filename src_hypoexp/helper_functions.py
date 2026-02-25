
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

