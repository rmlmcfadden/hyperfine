"""Probability distribution functions.
"""

from typing import Annotated, Sequence
import numpy as np
from scipy import special


def modified_beta_pdf(
    z: Sequence[float],
    alpha: Annotated[float, 0:None],
    beta: Annotated[float, 0:None],
    z_max: Annotated[float, 0:None],
) -> float:
    """Probability density function for a modified beta distribution.

    This modified form extends the beta distribution's domain from [0, 1] to [0, z_max].

    Args:
        z: Position.
        alpha: First shape parameter.
        beta: Second shape parameter.
        z_max: Upper bounds of the domain.

    Returns:
        The probability density at position z.
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


def modified_beta_2_pdf(
    z: Sequence[float],
    alpha_1: Annotated[float, 0:None],
    beta_1: Annotated[float, 0:None],
    z_max_1: Annotated[float, 0:None],
    fraction_1: Annotated[float, 0:1],
    alpha_2: Annotated[float, 0:None],
    beta_2: Annotated[float, 0:None],
    z_max_2: Annotated[float, 0:None],
) -> Sequence[float]:
    """Probability density function for a weighted sum of two modified beta distribution.

    This modified form extends the beta distribution's domain from [0, 1] to [0, max(z_max_1, z_max_2)].

    Args:
        z: Position.
        alpha_1: First shape parameter of the first component.
        beta_1: Second shape parameter of the first component.
        z_max_1: Upper bounds of the domain of the first component.
        fraction_1: Fractional weight of the first component.
        alpha_2: First shape parameter of the second component.
        beta_2: Second shape parameter of the second component.
        z_max_2: Upper bounds of the domain of the second component.

    Returns:
        The probability density at position z.
    """

    return fraction_1 * modified_beta_pdf(z, alpha_1, beta_1, z_max_1) + (
        1.0 - fraction_1
    ) * modified_beta_pdf(z, alpha_2, beta_2, z_max_2)


def modified_beta_mean(
    alpha: Annotated[float, 0:None],
    beta: Annotated[float, 0:None],
    z_max: Annotated[float, 0:None],
) -> float:
    """Mean position of a modified beta distribution.

    Args:
        alpha: First shape parameter.
        beta: Second shape parameter.
        z_max: Upper bounds of the domain.

    Returns:
        The mean position of the distribution.
    """

    return z_max * alpha / (alpha + beta)


def modified_beta_2_mean(
    alpha_1: Annotated[float, 0:None],
    beta_1: Annotated[float, 0:None],
    z_max_1: Annotated[float, 0:None],
    fraction_1: Annotated[float, 0:1],
    alpha_2: Annotated[float, 0:None],
    beta_2: Annotated[float, 0:None],
    z_max_2: Annotated[float, 0:None],
) -> float:
    """Mean position of a weighted sum of two modified beta distributions.

    Args:
        alpha_1: First shape parameter of the first component.
        beta_1: Second shape parameter of the first component.
        z_max_1: Upper bounds of the domain of the first component.
        fraction_1: Fractional weight of the first component.
        alpha_2: First shape parameter of the second component.
        beta_2: Second shape parameter of the second component.
        z_max_2: Upper bounds of the domain of the second component.

    Returns:
        The mean position of the distribution.
    """

    return fraction_1 * modified_beta_mean(alpha_1, beta_1, z_max_1) + (
        1.0 - fraction_1
    ) * modified_beta_mean(alpha_2, beta_2, z_max_2)


def skewed_gaussian_pdf(
    x: Sequence[float],
    mu: Annotated[float, None:None],
    sigma_m: Annotated[float, 0:None],
    sigma_p: Annotated[float, 0:None],
) -> Sequence[float]:
    """Probability density function for a skewed Gaussian distribution.

    This implementation uses a piecewise definition, wherein two scale parameters
    are used to define the width on either side of the distribution's location parameter.

    This convention here follows that of in musrfit).

    Note - all parameters must have the same units (e.g., magnetic field)!

    Args:
        x: Position.
        mu: Location parameter.
        sigma_m: Negative scale parameter (i.e., for positions less than the location parameter).
        sigma_p: Positive scale parameter (i.e., for positions greater than the location parameter).

    Returns:
        The probability density at position x.
    """

    return (
        np.sqrt(2.0 / np.pi)
        * (1.0 / (sigma_p + sigma_m))
        * np.piecewise(
            x,
            [
                x <= mu,
                x > mu,
            ],
            [
                lambda x: np.exp(-0.5 * np.square((x - mu) / sigma_m)),
                lambda x: np.exp(-0.5 * np.square((x - mu) / sigma_p)),
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
