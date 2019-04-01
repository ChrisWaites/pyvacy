import math
from pyvacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent

def epsilon(N, batch_size, noise_multiplier, epochs, delta=1e-5):
    """Calculates epsilon for stochastic gradient descent.

    Args:
        N (int): Total numbers of examples
        batch_size (int): Batch size
        noise_multiplier (float): Noise multiplier for DP-SGD
        epochs (float): number of epochs (may be fractional)
        delta (float): Target delta

    Returns:
        float: epsilon

    Example::
        >>> epsilon(10000, 256, 0.3, 100, 1e-5)
    """
    q = batch_size / N
    steps = int(math.ceil(epochs * N / batch_size))
    optimal_order = _ternary_search(lambda order: _apply_dp_sgd_analysis(q, noise_multiplier, steps, [order], delta), 1, 512, 0.1)
    return _apply_dp_sgd_analysis(q, noise_multiplier, steps, [optimal_order], delta)


def _apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Calculates epsilon for stochastic gradient descent.

    Args:
        q (float): Sampling probability, generally batch_size / number_of_samples
        sigma (float): Noise multiplier
        steps (float): Number of steps mechanism is applied
        orders (list(float)): Orders to try for finding optimal epsilon
        delta (float): Target delta

    Returns:
        float: epsilon

    Example::
        >>> epsilon(10000, 256, 0.3, 100, 1e-5)
    """
    rdp = compute_rdp(q, sigma, steps, orders)
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps

def _ternary_search(f, left, right, precision):
    """Performs a search over a closed domain [left, right] for the value which minimizes f."""
    while True:
        if abs(right - left) < precision:
            return (left + right) / 2

        left_third = left + (right - left) / 3
        right_third = right - (right - left) / 3

        if f(left_third) < f(right_third):
            right = right_third
        else:
            left = left_third

