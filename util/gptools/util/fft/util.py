from .. import ArrayOrTensor


sqrt2 = 1.4142135623730951
log2 = 0.6931471805599453
log2pi = 1.8378770664093453


def log_prob_stdnorm(y: ArrayOrTensor) -> ArrayOrTensor:
    """
    Evaluate the log probability of a standard normal random variable.
    """
    return - (log2pi + y * y) / 2
