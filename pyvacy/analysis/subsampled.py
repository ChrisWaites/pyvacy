import math
from scipy.special import beta

def binomial(x, y):
    return 1 / ((x + 1) * beta(x - y + 1, y + 1))

def epsilon(alpha, q, noise_multiplier):
    def _eps(_alpha):
        if _alpha == math.inf:
            return min(4 * (math.exp(_eps(2) - 1)), 2 * math.exp(_eps(2)))
        return _alpha / (2 * (noise_multiplier ** 2))

    s = 0.0
    for j in range(3, alpha + 1):
        s += (q ** j) * binomial(alpha, j) * math.exp((j - 1) * _eps(j)) * min(2, (math.exp(_eps(math.inf)) - 1) ** j)

    return (1 / (alpha - 1)) * math.log(1 + (q ** 2) * binomial(alpha, 2) * min(4 * (math.exp(_eps(2)) - 1), math.exp(2) * min(2, math.exp(_eps(math.inf)) - 1) ** j) + s)

