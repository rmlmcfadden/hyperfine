"""Muon spin spectroscopy (μSR) utilities.
"""

from . import meissner, _meissner
from typing import Annotated, Sequence, Tuple
import numpy as np
from scipy import constants
import jax


#: The muon gyromagnetic ratio (μs\ :sup:`-1` G\ :sup:`-1`), as defined in musrfit.
gamma_mu = (
    2.0
    * np.abs(
        constants.value("muon mag. mom.") / constants.value("reduced Planck constant")
    )
    * 1e-4  # 1/T to 1/G
    * 1e-6  # 1/s to 1/us
)


def raw_asymmetry(
    L: float,
    R: float,
    l: float = 0.0,
    r: float = 0.0,
) -> float:
    r"""Raw asymmetry between two detectors.

    For a pair of detectors (L)eft and (R)ight, calculate their asymmetry:

    .. math:: A_{\mathrm{raw}} = \frac{(L - l) - (R - r)}{(L - l) + (R - r)}

    where
    :math:`A_{\mathrm{raw}}` is the raw asymmetry,
    :math:`L` is the left detector's total counts,
    :math:`l` is the left detector's background counts,
    :math:`R` is the right detector's total counts,
    and
    :math:`r` is the right detector's background counts.

    Args:
        L: Left detector total counts.
        R: Right detector total counts.
        l: Left detector background counts.
        r: Right detector background counts.

    Returns:
        The raw asymmetry.
    """

    # do the background correction
    L_c = L - l
    R_c = R - r

    # calculate the asymmetry
    return (L_c - R_c) / (L_c + R_c)


def raw_asymmetry_error(
    L: float,
    L_error: float,
    R: float,
    R_error: float,
    l: float = 0.0,
    l_error: float = 0.0,
    r: float = 0.0,
    r_error: float = 0.0,
) -> float:
    """Uncertainty in the raw asymmetry between two detectors.

    Calculate the uncertainty in the raw asymmetry using standard error
    propagation.

    Note: the calculation assumes all uncertainties are uncorrelated.

    Args:
        L: Left detector total counts.
        L_error: Left detector uncertainty.
        R: Right detector total counts.
        R_error: Right detector uncertainty.
        l: Left detector background counts.
        l_error: Left detector background uncertainty.
        r: Right detector background counts.
        r_error: Right detector background uncertainty.

    Returns:
        The uncertainty of the raw asymmetry.
    """

    # partial derivatives
    df_dL = -2 * (r - R) / (-l + L - r + R) ** 2
    df_dR = 2 * (l - L) / (-l + L - r + R) ** 2
    df_dl = 2 * (r - R) / (-l + L - r + R) ** 2
    df_dr = -2 * (l - L) / (-l + L - r + R) ** 2

    # propagated uncertainty
    return np.sqrt(
        np.square(df_dL * L_error)
        + np.square(df_dR * R_error)
        + np.square(df_dl * l_error)
        + np.square(df_dr * r_error)
    )


def corrected_asymmetry(
    L: float,
    R: float,
    l: float = 0.0,
    r: float = 0.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    r"""Corrected asymmetry between two detectors.

    For a pair of detectors (L)eft and (R)ight, calculate their corrected
    asymmetry:

    .. math:: A_{\mathrm{corr.}} = \frac{(\alpha - l) + (\alpha + 1) A_{\mathrm{raw}}}{(\alpha \beta + l) + (\alpha \beta - 1) A_{\mathrm{raw}}}

    where
    :math:`A_{\mathrm{corr.}}` is the corrected asymmetry,
    :math:`\alpha` is ratio of detector counts,
    :math:`\beta` is the ratio of detector asymmetries,
    and
    :math:`A_{\mathrm{raw}}` is the raw asymmetry

    .. math:: A_{\mathrm{raw}} = \frac{(L - l) - (R - r)}{(L - l) + (R - r)}

    where
    :math:`A_{\mathrm{raw}}` is the raw asymmetry,
    :math:`L` is the left detector's total counts,
    :math:`l` is the left detector's background counts,
    :math:`R` is the right detector's total counts,
    and
    :math:`r` is the right detector's background counts.

    See e.g., Noakes et al., Phys. Rev. B 35, 6597 (1987).
    https://doi.org/10.1103/PhysRevB.35.6597

    Args:
        L: Left detector total counts.
        R: Right detector total counts.
        l: Left detector background counts.
        r: Right detector background counts.
        alpha: Ratio of detector counts.
        beta: Ratio of detector asymmetries.

    Returns:
        The corrected asymmetry between two detectors.
    """

    A_raw = raw_asymmetry(L, R, l, r)

    numerator = (1.0 - alpha) + (alpha + 1.0) * A_raw
    denominator = (alpha * beta + 1.0) + (alpha * beta - 1.0) * A_raw

    return numerator / denominator


def corrected_asymmetry_error(
    L: float,
    L_error: float,
    R: float,
    R_error: float,
    l: float = 0.0,
    l_error: float = 0.0,
    r: float = 0.0,
    r_error: float = 0.0,
    alpha: float = 1.0,
    alpha_error: float = 0.0,
    beta: float = 1.0,
    beta_error: float = 0.0,
) -> float:
    """Uncertainty in the corrected asymmetry between two detectors.

    Calculate the uncertainty in the corrected asymmetry using standard error
    propagation.

    Note: the calculation assumes all uncertainties are uncorrelated.

    Args:
        L: Left detector total counts.
        L_error: Left detector uncertainty.
        R: Right detector total counts.
        R_error: Right detector uncertainty.
        l: Left detector background counts.
        l_error: Left detector background uncertainty.
        r: Right detector background counts.
        r_error: Right detector background uncertainty.
        alpha: Ratio of detector counts.
        alpha_error: Uncertainty in the ratio of detector counts.
        beta: Ratio of detector asymmetries.
        beta_error: Uncertainty in the ratio of detector asymmetries.

    Returns:
        The uncertainty of the raw asymmetry.
    """

    A_raw = raw_asymmetry(L, R, l, r)
    A_raw_error = raw_asymmetry_error(L, L_error, R, R_error, l, l_error, r, r_error)

    # common term
    denom = (A_raw * (alpha * beta - 1) + alpha * beta + 1) ** 2
    # partial derivatives
    df_dA_raw = 2 * alpha * (beta + 1) / denom
    df_dalpha = -(A_raw**2 - 1) * (beta + 1) / denom
    df_dbeta = -alpha * (A_raw + 1) * (alpha * A_raw + alpha + A_raw - 1) / denom

    # error propagation
    return np.sqrt(
        np.square(df_dA_raw * A_raw_error)
        + np.square(df_dalpha * alpha_error)
        + np.square(df_dbeta * beta_error)
    )


def skewed_gaussian_mean(
    B_0_G: float,
    sigma_m__us: float,
    sigma_p__us: float,
) -> float:
    r"""Mean of the skewed Gaussian distribution (as defined in musrfit).

    See also: https://lmu.web.psi.ch/docu/LEM_Memo/skewedGaussian/skewedGaussian.pdf

    Args:
        B_0_G: Peak field position (G).
        sigma_m__us: Negative damping parameter (μs\ :sup:`-1`).
        sigma_p__us: Positive damping parameter (μs\ :sup:`-1`).

    Returns:
        The mean of the skewed Gaussian distribution (G).
    """

    return B_0_G + np.sqrt(2.0 / np.pi) * (sigma_p__us - sigma_m__us) / gamma_mu


def skewed_gaussian_mean_error(
    B_0_G: float,
    B_0_error_G: float,
    sigma_m__us: float,
    sigma_m_error__us: float,
    sigma_p__us: float,
    sigma_p_error__us: float,
) -> float:
    r"""Uncertainty of the mean of the skewed Gaussian distribution (as defined in musrfit).

    See also: https://lmu.web.psi.ch/docu/LEM_Memo/skewedGaussian/skewedGaussian.pdf

    Args:
        B_0_G: Peak field position (G).
        B_0_error_G: Uncertainty in the peak field position (G).
        sigma_m__us: Negative damping parameter (μs\ :sup:`-1`).
        sigma_m_error__us: Uncertainty in the negative damping parameter (μs\ :sup:`-1`).
        sigma_p__us: Positive damping parameter (μs\ :sup:`-1`).
        sigma_p_error__us: Uncertainty in the positive damping parameter (μs\ :sup:`-1`).

    Returns:
        The uncertainty in the mean of the skewed Gaussian distribution (G).
    """

    # convert the units of the sigmas from 1/us to G
    sigma_m_G = sigma_m__us / gamma_mu
    sigma_m_error_G = sigma_m_error__us / gamma_mu
    sigma_p_G = sigma_p__us / gamma_mu
    sigma_p_error_G = sigma_p_error__us / gamma_mu

    # first derivatives
    dskg_dB_0 = 1.0
    dskg_dsigma_m = -1.0 * np.sqrt(2.0 / np.pi)
    dskg_dsigma_p = 1.0 * np.sqrt(2.0 / np.pi)

    # error propagation
    return np.sqrt(
        np.square(dskg_dB_0 * B_0_error_G)
        + np.square(dskg_dsigma_m * sigma_m_error_G)
        + np.square(dskg_dsigma_p * sigma_p_error_G)
    )


def three_gaussian_avg_and_error(
    b_1: float,
    b_1_error: float,
    b_2: float,
    b_2_error: float,
    b_3: float,
    b_3_error: float,
    a_1: float,
    a_1_error: float,
    a_2: float,
    a_2_error: float,
    a_3: float,
    a_3_error: float,
) -> Tuple[float, float]:
    """Mean (i.e., the 1st moment) of a triple-Gaussian distribution.

    Args:
        b_1: Mean of the 1st Gaussian distribution (G).
        b_1_error: Uncertainty of the mean of the 1st Gaussian distribution (G).
        b_2: Mean of the 2nd Gaussian distribution (G).
        b_2_error: Uncertainty of the mean of the 2nd Gaussian distribution (G).
        b_3: Mean of the 3rd Gaussian distribution (G).
        b_3_error: Uncertainty of the mean of the 3rd Gaussian distribution (G).
        a_1: Amplitude of the 1st Gaussian distribution.
        a_1_error: Uncertainty of the amplitude of the 1st Gaussian distribution.
        a_2: Amplitude of the 2nd Gaussian distribution.
        a_2_error: Uncertainty of the amplitude of the 2nd Gaussian distribution.
        a_3: Amplitude of the 3rd Gaussian distribution.
        a_3_error: Uncertainty of the amplitude of the 3rd Gaussian distribution.

    Returns:
        A tuple containing the distribution's mean and uncertainty (G).
    """

    # convenience terms
    sum_a = a_1 + a_2 + a_3

    # calculate the averge
    avgerage = (a_1 * b_1 + a_2 * b_2 + a_3 * b_3) / sum_a

    # calculate the partial derivatives
    pd_b_1 = a_1 / (a_1 + a_2 + a_3)
    pd_b_2 = a_2 / (a_1 + a_2 + a_3)
    pd_b_3 = a_3 / (a_1 + a_2 + a_3)
    pd_a_1 = (a_2 * (b_1 - b_2) + a_3 * (b_1 - b_3)) / (sum_a**2)
    pd_a_2 = (a_1 * (b_2 - b_1) + a_3 * (b_2 - b_3)) / (sum_a**2)
    pd_a_3 = (a_1 * (b_3 - b_1) + a_2 * (b_3 - b_2)) / (sum_a**2)

    # calculate the uncertainty
    error = np.sqrt(
        (pd_b_1 * b_1_error) ** 2
        + (pd_b_2 * b_2_error) ** 2
        + (pd_b_3 * b_3_error) ** 2
        + (pd_a_1 * a_1_error) ** 2
        + (pd_a_2 * a_2_error) ** 2
        + (pd_a_3 * a_3_error) ** 2
    )

    # return the tuple of values
    return (avgerage, error)


def _avg3(
    b_1: Annotated[float, None:None],
    b_2: Annotated[float, None:None],
    b_3: Annotated[float, None:None],
    a_1: Annotated[float, 0.0:None],
    a_2: Annotated[float, 0.0:None],
    a_3: Annotated[float, 0.0:None],
) -> float:
    """Mean of a triple Gaussian distribution.

    Args:
        b_1: Mean of the 1st Gaussian distribution (G).
        b_2: Mean of the 2nd Gaussian distribution (G).
        b_3: Mean of the 3rd Gaussian distribution (G).
        a_1: Amplitude of the 1st Gaussian distribution.
        a_2: Amplitude of the 2nd Gaussian distribution.
        a_3: Amplitude of the 3rd Gaussian distribution.

    Returns:
        The mean of the triple Gaussian distribution (G).
    """

    vals = (b_1, b_2, b_3, a_1, a_2, a_3)
    sum_a = a_1 + a_2 + a_3
    avg = 0.0

    for i in range(3):
        avg += (vals[3 + i] / sum_a) * vals[0 + i]

    return avg


def _var3(
    b_1: Annotated[float, None:None],
    b_2: Annotated[float, None:None],
    b_3: Annotated[float, None:None],
    a_1: Annotated[float, 0.0:None],
    a_2: Annotated[float, 0.0:None],
    a_3: Annotated[float, 0.0:None],
    s_1: Annotated[float, 0.0:None],
    s_2: Annotated[float, 0.0:None],
    s_3: Annotated[float, 0.0:None],
) -> float:
    r"""Variance of a triple Gaussian distribution.

    Args:
        b_1: Mean of the 1st Gaussian distribution (G).
        b_2: Mean of the 2nd Gaussian distribution (G).
        b_3: Mean of the 3rd Gaussian distribution (G).
        a_1: Amplitude of the 1st Gaussian distribution.
        a_2: Amplitude of the 2nd Gaussian distribution.
        a_3: Amplitude of the 3rd Gaussian distribution.
        s_1: Standard deviation of the 1st Gaussian distribution (G).
        s_2: Standard deviation of the 2nd Gaussian distribution (G).
        s_3: Standard deviation of the 3rd Gaussian distribution (G).

    Returns:
        The variance of a triple Gaussian distribution (G\ :sup:`2`).
    """

    vals = (b_1, b_2, b_3, a_1, a_2, a_3, s_1, s_2, s_3)
    sum_a = a_1 + a_2 + a_3
    var = 0.0

    avg = _avg3(*vals[:6])

    for i in range(3):
        var += (vals[3 + i] / sum_a) * (vals[6 + i] ** 2 + (vals[0 + i] - avg) ** 2)

    return var


def _std3(
    b_1: Annotated[float, None:None],
    b_2: Annotated[float, None:None],
    b_3: Annotated[float, None:None],
    a_1: Annotated[float, 0.0:None],
    a_2: Annotated[float, 0.0:None],
    a_3: Annotated[float, 0.0:None],
    s_1: Annotated[float, 0.0:None],
    s_2: Annotated[float, 0.0:None],
    s_3: Annotated[float, 0.0:None],
) -> float:
    r"""Standard deviation of a triple Gaussian distribution.

    Args:
        b_1: Mean of the 1st Gaussian distribution (G).
        b_2: Mean of the 2nd Gaussian distribution (G).
        b_3: Mean of the 3rd Gaussian distribution (G).
        a_1: Amplitude of the 1st Gaussian distribution.
        a_2: Amplitude of the 2nd Gaussian distribution.
        a_3: Amplitude of the 3rd Gaussian distribution.
        s_1: Standard deviation of the 1st Gaussian distribution (G).
        s_2: Standard deviation of the 2nd Gaussian distribution (G).
        s_3: Standard deviation of the 3rd Gaussian distribution (G).

    Returns:
        The standard deviation of a triple Gaussian distribution (G).
    """

    return jax.numpy.sqrt(_var3(b_1, b_2, b_3, a_1, a_2, a_3, s_1, s_2, s_3))


def three_gaussian_var_and_error(
    b_1: float,
    b_1_error: float,
    b_2: float,
    b_2_error: float,
    b_3: float,
    b_3_error: float,
    a_1: float,
    a_1_error: float,
    a_2: float,
    a_2_error: float,
    a_3: float,
    a_3_error: float,
    s_1: float,
    s_1_error: float,
    s_2: float,
    s_2_error: float,
    s_3: float,
    s_3_error: float,
) -> Tuple[float, float]:
    r"""Variance (i.e., the 2nd moment) of a triple-Gaussian distribution.

    Args:
        b_1: Mean of the 1st Gaussian distribution (G).
        b_1_error: Uncertainty of the mean of the 1st Gaussian distribution (G).
        b_2: Mean of the 2nd Gaussian distribution (G).
        b_2_error: Uncertainty of the mean of the 2nd Gaussian distribution (G).
        b_3: Mean of the 3rd Gaussian distribution (G).
        b_3_error: Uncertainty of the mean of the 3rd Gaussian distribution (G).
        a_1: Amplitude of the 1st Gaussian distribution.
        a_1_error: Uncertainty of the amplitude of the 1st Gaussian distribution.
        a_2: Amplitude of the 2nd Gaussian distribution.
        a_2_error: Uncertainty of the amplitude of the 2nd Gaussian distribution.
        a_3: Amplitude of the 3rd Gaussian distribution.
        a_3_error: Uncertainty of the amplitude of the 3rd Gaussian distribution.
        s_1: Damping rate of the 1st Gaussian distribution (μs\ :sup:`-1`).
        s_1_error: Uncertainty of the damping rate of the 1st Gaussian distribution(μs\ :sup:`-1`).
        s_2: Damping rate of the 2nd Gaussian distribution (μs\ :sup:`-1`).
        s_2_error: Uncertainty of the damping rate of the 2nd Gaussian distribution(μs\ :sup:`-1`).
        s_3: Damping rate of the 3rd Gaussian distribution (μs\ :sup:`-1`).
        s_3_error: Uncertainty of the damping rate of the 3rd Gaussian distribution(μs\ :sup:`-1`).

    Returns:
        A tuple containing the distribution's variance and uncertainty (G\ :sup:`2`).
    """

    # convenience terms
    args = (
        b_1,
        b_2,
        b_3,
        a_1,
        a_2,
        a_3,
        # convert the damping rates to fields (G)
        s_1 / gamma_mu,
        s_2 / gamma_mu,
        s_3 / gamma_mu,
    )

    # calculate the variance
    variance = _var3(*args)

    # calculate the partial derivatives
    pd_b_1 = jax.grad(_var3, argnums=0)
    pd_b_2 = jax.grad(_var3, argnums=1)
    pd_b_3 = jax.grad(_var3, argnums=2)
    pd_a_1 = jax.grad(_var3, argnums=3)
    pd_a_2 = jax.grad(_var3, argnums=4)
    pd_a_3 = jax.grad(_var3, argnums=5)
    pd_s_1 = jax.grad(_var3, argnums=6)
    pd_s_2 = jax.grad(_var3, argnums=7)
    pd_s_3 = jax.grad(_var3, argnums=8)

    # calculate the uncertainty
    error = np.sqrt(
        (pd_b_1(*args) * b_1_error) ** 2
        + (pd_b_2(*args) * b_2_error) ** 2
        + (pd_b_3(*args) * b_3_error) ** 2
        + (pd_a_1(*args) * a_1_error) ** 2
        + (pd_a_2(*args) * a_2_error) ** 2
        + (pd_a_3(*args) * a_3_error) ** 2
        + (pd_s_1(*args) * (s_1_error / gamma_mu)) ** 2
        + (pd_s_2(*args) * (s_2_error / gamma_mu)) ** 2
        + (pd_s_3(*args) * (s_3_error / gamma_mu)) ** 2
    )

    # return the tuple of values
    return (variance, error)


def three_gaussian_std_and_error(
    b_1: float,
    b_1_error: float,
    b_2: float,
    b_2_error: float,
    b_3: float,
    b_3_error: float,
    a_1: float,
    a_1_error: float,
    a_2: float,
    a_2_error: float,
    a_3: float,
    a_3_error: float,
    s_1: float,
    s_1_error: float,
    s_2: float,
    s_2_error: float,
    s_3: float,
    s_3_error: float,
) -> Tuple[float, float]:
    r"""Variance (i.e., the 2nd moment) of a triple-Gaussian distribution.

    Args:
        b_1: Mean of the 1st Gaussian distribution (G).
        b_1_error: Uncertainty of the mean of the 1st Gaussian distribution (G).
        b_2: Mean of the 2nd Gaussian distribution (G).
        b_2_error: Uncertainty of the mean of the 2nd Gaussian distribution (G).
        b_3: Mean of the 3rd Gaussian distribution (G).
        b_3_error: Uncertainty of the mean of the 3rd Gaussian distribution (G).
        a_1: Amplitude of the 1st Gaussian distribution.
        a_1_error: Uncertainty of the amplitude of the 1st Gaussian distribution.
        a_2: Amplitude of the 2nd Gaussian distribution.
        a_2_error: Uncertainty of the amplitude of the 2nd Gaussian distribution.
        a_3: Amplitude of the 3rd Gaussian distribution.
        a_3_error: Uncertainty of the amplitude of the 3rd Gaussian distribution.
        s_1: Damping rate of the 1st Gaussian distribution (μs\ :sup:`-1`).
        s_1_error: Uncertainty of the damping rate of the 1st Gaussian distribution(μs\ :sup:`-1`).
        s_2: Damping rate of the 2nd Gaussian distribution (μs\ :sup:`-1`).
        s_2_error: Uncertainty of the damping rate of the 2nd Gaussian distribution(μs\ :sup:`-1`).
        s_3: Damping rate of the 3rd Gaussian distribution (μs\ :sup:`-1`).
        s_3_error: Uncertainty of the damping rate of the 3rd Gaussian distribution(μs\ :sup:`-1`).

    Returns:
        A tuple containing the distribution's standard deviation and uncertainty (G).
    """

    # convenience terms
    args = (
        b_1,
        b_2,
        b_3,
        a_1,
        a_2,
        a_3,
        # convert the damping rates to fields (G)
        s_1 / gamma_mu,
        s_2 / gamma_mu,
        s_3 / gamma_mu,
    )

    # calculate the standard deviation
    std = _std3(*args)

    # calculate the partial derivatives
    pd_b_1 = jax.grad(_std3, argnums=0)
    pd_b_2 = jax.grad(_std3, argnums=1)
    pd_b_3 = jax.grad(_std3, argnums=2)
    pd_a_1 = jax.grad(_std3, argnums=3)
    pd_a_2 = jax.grad(_std3, argnums=4)
    pd_a_3 = jax.grad(_std3, argnums=5)
    pd_s_1 = jax.grad(_std3, argnums=6)
    pd_s_2 = jax.grad(_std3, argnums=7)
    pd_s_3 = jax.grad(_std3, argnums=8)

    # calculate the uncertainty
    error = np.sqrt(
        (pd_b_1(*args) * b_1_error) ** 2
        + (pd_b_2(*args) * b_2_error) ** 2
        + (pd_b_3(*args) * b_3_error) ** 2
        + (pd_a_1(*args) * a_1_error) ** 2
        + (pd_a_2(*args) * a_2_error) ** 2
        + (pd_a_3(*args) * a_3_error) ** 2
        + (pd_s_1(*args) * (s_1_error / gamma_mu)) ** 2
        + (pd_s_2(*args) * (s_2_error / gamma_mu)) ** 2
        + (pd_s_3(*args) * (s_3_error / gamma_mu)) ** 2
    )

    # return the tuple of values
    return (std, error)


def multi_gaussian_avg_and_error(
    mean: Sequence[float],
    mean_error: Sequence[float],
    fraction: Sequence[float],
    fraction_error: Sequence[float],
) -> Tuple[float, float]:
    """Mean (i.e., the 1st moment) of a multi-Gaussian distribution.

    Args:
        mean: Array of mean values (G).
        mean_error: Array of mean errors (G).
        fraction: Array of fractions.
        fraction: Array of fraction errors.

    Returns:
        A tuple containing the distribution's mean and uncertainty (G).
    """

    # make sure the arrays are all the same size
    assert len(mean) == len(mean_error)
    assert len(mean) == len(fraction)
    assert len(mean) == len(fraction_error)

    # make sure the fractions sum up to unity
    norm = sum(fraction)
    fraction = [f / norm for f in fraction]
    fraction_error = [fe / norm for fe in fraction_error]

    # calculate the (weighted) average
    # Eq. (6.73) in https://doi.org/10.1007/978-3-031-44959-8_6
    # value = np.average(mean, weights=fraction)
    value = sum([f * m for m, f in zip(mean, fraction)]) / sum(fraction)

    # calculate the uncertainty
    variance = 0.0
    for m, me, f, fe in zip(mean, mean_error, fraction, fraction_error):
        variance += (f * me) ** 2
        variance += (m * fe) ** 2

    error = np.sqrt(variance)

    # return the tuple containing the value & its uncertainty
    return (value, error)


def multi_gaussian_var_and_error(
    mean: Sequence[float],
    mean_error: Sequence[float],
    fraction: Sequence[float],
    fraction_error: Sequence[float],
    sigma: Sequence[float],
    sigma_error: Sequence[float],
) -> Tuple[float, float]:
    r"""Variance (i.e., the 2nd moment) of a multi-Gaussian distribution.

    Args:
        mean: Array of mean values (G).
        mean_error: Array of mean errors (G).
        fraction: Array of fractions.
        fraction: Array of fraction errors.
        sigma: Array of damping rates (μs\ :sup:`-1`).
        sigma_error: Array of damping rate errors (μs\ :sup:`-1`).

    Returns:
        A tuple containing the distribution's variance and uncertainty (G\ :sup:`2`).
    """

    # make sure the arrays are all the same size
    assert len(mean) == len(mean_error)
    assert len(mean) == len(fraction)
    assert len(mean) == len(fraction_error)
    assert len(mean) == len(sigma)
    assert len(mean) == len(sigma_error)

    # make sure the fractions sum up to unity
    norm = sum(fraction)
    fraction = [f / norm for f in fraction]
    fraction_error = [fe / norm for fe in fraction_error]

    # compute the mean of the field distribution
    avg, avg_error = multi_gaussian_avg_and_error(
        mean, mean_error, fraction, fraction_error
    )

    # convert the sigmas to standard deviations in G
    std = [s / gamma_mu for s in sigma]
    std_error = [se / gamma_mu for se in sigma_error]

    # compute the variance & its uncertainty
    # Eq. (6.74) in https://doi.org/10.1007/978-3-031-44959-8_6
    variance = 0.0
    variance_variance = 0.0

    for m, me, f, fe, s, se in zip(
        mean, mean_error, fraction, fraction_error, std, std_error
    ):
        # variance contribution
        variance += f * (s**2 + (m - avg) ** 2)

        # partial derivatives
        pd_f = s**2 + (m - avg) ** 2
        pd_s = 2.0 * f * s
        pd_m = 2.0 * f * (m - avg)
        pd_a = -2.0 * f * (m - avg)

        # variance uncertainty contribution
        variance_variance += (
            (pd_f * fe) ** 2
            + (pd_s * se) ** 2
            + (pd_m * me) ** 2
            + (pd_a * avg_error) ** 2
        )

    variance_error = np.sqrt(variance_variance)

    # return the tuple containing the value & its uncertainty
    return (variance, variance_error)


def multi_gaussian_std_and_error(
    mean: Sequence[float],
    mean_error: Sequence[float],
    fraction: Sequence[float],
    fraction_error: Sequence[float],
    sigma: Sequence[float],
    sigma_error: Sequence[float],
) -> Tuple[float, float]:
    r"""Standard deviation (i.e., square root of the 2nd moment) of a multi-Gaussian distribution.

    Args:
        mean: Array of mean values (G).
        mean_error: Array of mean errors (G).
        fraction: Array of fractions.
        fraction: Array of fraction errors.
        sigma: Array of damping rates (μs\ :sup:`-1`).
        sigma_error: Array of damping rate errors (μs\ :sup:`-1`).

    Returns:
        A tuple containing the distribution's standard deviation and uncertainty (G).
    """

    # make sure the arrays are all the same size
    assert len(mean) == len(mean_error)
    assert len(mean) == len(fraction)
    assert len(mean) == len(fraction_error)
    assert len(mean) == len(sigma)
    assert len(mean) == len(sigma_error)

    # get the variance and its uncertainty
    variance, variance_error = multi_gaussian_var_and_error(
        mean, mean_error, fraction, fraction_error, sigma, sigma_error
    )

    # compute the standard deviation and its uncertainty
    std = np.sqrt(variance)
    std_error = (0.5 / np.sqrt(variance)) * variance_error

    # return the tuple containing the value & its uncertainty
    return (std, std_error)
