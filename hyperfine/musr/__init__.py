from . import meissner, _meissner
import numpy as np
from scipy import constants


# muon gyromagnetic ratio in μs^-1 G^-1 (as defined in musrfit)
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
    l: float = 0,
    r: float = 0,
) -> float:
    """
    Definition of asymmetry between two detectors (L)eft and (R)ight.
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
    l: float = 0,
    l_error: float = 0,
    r: float = 0,
    r_error: float = 0,
) -> float:
    """
    Error propagation of the asymmetry
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
    l: float = 0,
    r: float = 0,
    alpha: float = 1,
    beta: float = 1,
) -> float:
    """
    Definition of corrected asymmetry.

    See e.g., Noakes et al., Phys. Rev. B 35, 6597 (1987).
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
    l: float = 0,
    l_error=0,
    r: float = 0,
    r_error: float = 0,
    alpha: float = 1,
    alpha_error: float = 0,
    beta: float = 1,
    beta_error: float = 0,
) -> float:
    """
    Error propagation for the corrected asymmetry calculation.
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
    """
    Mean of the skewed Gaussian distribution (as defined in musrfit).

    See also: https://lmu.web.psi.ch/docu/LEM_Memo/skewedGaussian/skewedGaussian.pdf
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
    """
    Uncertainty of the mean of the skewed Gaussian distribution (as defined in musrfit).

    See also: https://lmu.web.psi.ch/docu/LEM_Memo/skewedGaussian/skewedGaussian.pdf
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
