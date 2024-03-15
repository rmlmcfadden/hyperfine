from typing import Annotated, Sequence
import numpy as np
from scipy import special


def modified_beta(
    z: float,
    alpha: float,
    beta: float,
    z_max: float,
) -> float:
    """
    Modified beta distribution.
    """
    return np.piecewise(
        z,
        [
            z <= 0.0,
            (z > 0.0) & (z < z_max),
            z >= z_max,
        ],
        [
            lambda z: 0.0 * z,
            lambda z: np.power(z / z_max, alpha - 1.0)
            * np.power(1.0 - z / z_max, beta - 1.0)
            / special.beta(alpha, beta)
            / z_max,
            lambda z: 0.0 * z,
        ],
    )


def modified_beta_2(
    z: float,
    alpha_1: float,
    beta_1: float,
    z_max_1: float,
    fraction_1: float,
    alpha_2: float,
    beta_2: float,
    z_max_2: float,
) -> float:
    """
    Sum of two modified beta distributions.
    """
    return fraction_1 * modified_beta(z, alpha_1, beta_1, z_max_1) + (
        1.0 - fraction_1
    ) * modified_beta(z, alpha_2, beta_2, z_max_2)


def modified_beta_mean(alpha: float, beta: float, z_max: float) -> float:
    """
    Mean of the modified beta distribution
    """
    return z_max * alpha / (alpha + beta)


def modified_beta_2_mean(
    alpha_1: float,
    beta_1: float,
    z_max_1: float,
    fraction_1: float,
    alpha_2: float,
    beta_2: float,
    z_max_2: float,
) -> float:
    """
    Mean of the modified beta distribution 2
    """
    return fraction_1 * modified_beta_mean(alpha_1, beta_1, z_max_1) + (
        1.0 - fraction_1
    ) * modified_beta_mean(alpha_2, beta_2, z_max_2)


def skewed_gaussian(
    B: float,
    B_0: float,
    sigma_m: float,
    sigma_p: float,
) -> float:
    """
    Skewed Gaussian distribution (as defined in musrfit).

    Note: all parameters must have the same units (e.g., magnetic field)!
    """

    #
    return (
        np.sqrt(2.0 / np.pi)
        * (1.0 / (sigma_p + sigma_m))
        * np.piecewise(
            B,
            [B <= B_0, B > B_0],
            [
                lambda B: np.exp(-0.5 * np.square((B - B_0) / sigma_m)),
                lambda B: np.exp(-0.5 * np.square((B - B_0) / sigma_p)),
            ],
        )
    )


def weibull_pdf(
    x: Sequence[float],
    shape: Annotated[float, 0:None],
    scale: Annotated[float, 0:None],
) -> Sequence[float]:
    """Probability density function for a Weibull distribution.

    Args:
        x: Position.
        shape: Shape parameter.
        scale: Scale parameter.

    Returns:
        The probability density at x.

    """

    k = shape
    l = scale

    return np.piecewise(
        x,
        [
            x < 0.0,
            x >= 0.0,
        ],
        [
            lambda x: np.zeros(x.size),
            lambda x: (k / l) * np.power(x / l, k - 1.0) * np.exp(-np.power(x / l, k)),
        ],
    )


def weibull_cdf(
    x: Sequence[float],
    shape: Annotated[float, 0:None],
    scale: Annotated[float, 0:None],
) -> Sequence[float]:
    """Cumulative distribution function for a Weibull distribution.

    Args:
        x: Position.
        shape: Shape parameter.
        scale: Scale parameter.

    Returns:
        The cumulative probability density at x.
    """

    k = shape
    l = scale

    return np.piecewise(
        x,
        [
            x < 0.0,
            x >= 0.0,
        ],
        [
            lambda x: np.zeros(x.size),
            lambda x: 1.0 - np.exp(-np.power(x / l, k)),
        ],
    )


def weibull_mean(
    shape: Annotated[float, 0:None],
    scale: Annotated[float, 0:None],
) -> float:
    """Mean of a Weibull distribution.

    Args:
        x: Position.
        shape: Shape parameter.
        scale: Scale parameter.

    Returns:
        The mean of the distribution.
    """

    k = shape
    l = scale

    return l * special.gamma(1.0 + 1.0 / k)
