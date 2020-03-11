"""Functions to simulate various error types.
"""
import numpy as np
from scipy import stats, special


def generate_gaussian_noise(N, Y, target_R2 = 0.005):
    """Draw N observations of Gaussian noise.

    We set the standard deviation that would target for the R^2 specified. The
    default is 0.5%.

    Parameters
    ----------
    N: integer
        The number of observations to draw.
    Y: vector of reals.
        The true output of the data, ie. f(X).
    target_R2: real in [0, 1]
        The target R^2. For example, 0.5% would be 0.005. Default is 0.5%.
    """
    coef = (1 - target_R2) / target_R2
    sd = np.sqrt(coef * np.var(Y))
    noise = np.random.normal(0, sd, size = N)
    return noise


def generate_cauchy_noise(N, Y, target_R2 = 0.005):
    """Draw N observations of Cauchy noise.

    We set the scale parameter s/t it would lead to the given R^2 with Gaussian
    error.

    Parameters
    ----------
    N: integer
        The number of observations to draw.
    Y: vector of reals.
        The true output of the data, ie. f(X).
    target_R2: real in [0, 1]
        The target R^2, if errors were Gaussian. For example, 0.5% would be
        0.005. Default is 0.5%.
    """
    coef = (1 - target_R2) / target_R2
    scale = np.sqrt(coef * np.var(Y))
    return scale * np.random.standard_cauchy(N)


def generate_t2_noise(N, Y, target_R2 = 0.005):
    """Draw N observations from Student's t with DoF = 2.
    
    This has mean 0 but undefined variance. We set the scale parameter s/t it
    would lead to the given R^2 with Gaussian error.

    Parameters
    ----------
    N: integer
        The number of observations to draw.
    Y: vector of reals.
        The true output of the data, ie. f(X).
    target_R2: real in [0, 1]
        The target R^2, if errors were Gaussian. For example, 0.5% would be
        0.005. Default is 0.5%.
    """
    coef = (1 - target_R2) / target_R2
    scale = np.sqrt(coef * np.var(Y))
    noise = scale * np.random.standard_t(2, size = N) 
    return noise


def generate_skew_normal_errors(skew, N, Y, target_R2 = 0.005):
    """Draw N observations from a skew normal distribution.
    
    We set the scale parameter s/t it would lead to the given with Gaussian
    error. The errors are centered to have mean 0.

    Parameters
    ----------
    skew: real
        A parameter that adjusts the skewness of the errors. Negative leads to
        left-skewed distributions, and positive leads to right-skewed
        distributions.
    N: integer
        The number of observations to draw.
    Y: vector of reals.
        The true output of the data, ie. f(X).
    target_R2: real in [0, 1]
        The target R^2, if errors were Gaussian. For example, 0.5% would be
        0.005. Default is 0.5%.
    """
    coef = (1 - target_R2) / target_R2
    scale = np.sqrt(coef * np.var(Y))
    # Center to have mean 0. See https://en.wikipedia.org/wiki/Skew_normal_distribution.
    loc = -scale * skew / np.sqrt(np.square(skew) + 1) * np.sqrt(2 / np.pi)
    errors = stats.skewnorm(a = skew, loc = loc, scale = scale).rvs(N)
    return errors


def generate_gev_noise(c, N, Y, target_R2 = 0.005):
    """Draw N observations from a generalized extreme value distribution.

    The GEV distribution is described at
    https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous_genextreme.html.

    Parameters
    ----------
    c: real
        The shape parameter for the distribution. The distribution is skewed
        left if c > 0 and skewed right if c < 0. The larger the magnitude of c,
        the higher the kurtosis of the distribution.
    N: integer
        The number of observations to draw.
    Y: vector of reals.
        The true output of the data, ie. f(X).
    target_R2: real in [0, 1]
        The target R^2, if errors were Gaussian. For example, 0.5% would be
        0.005. Default is 0.5%.
    """
    coef = (1 - target_R2) / target_R2
    scale = np.sqrt(coef * np.var(Y))
    # Center the distribution to have mean 0.
    center = -1 / c * (1 - special.gamma(1 + c))
    noise = stats.genextreme(c = c, loc = center, scale = scale).rvs(size = N)
    return noise
