import numpy as np
from scipy import constants, special


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
    return fraction_1 * modified_beta_dist(z, alpha_1, beta_1, z_max_1) + (
        1.0 - fraction_1
    ) * modified_beta_dist(z, alpha_2, beta_2, z_max_2)


# muon gyromagnetic ratio in μs^-1 G^-1 (as defined in musrfit)
gamma_mu = (
    2.0
    * np.abs(
        constants.physical_constants["muon mag. mom."][0]
        / constants.physical_constants["reduced Planck constant"][0]
    )
    * 1e-4  # 1/T to 1/G
    * 1e-6  # 1/s to 1/us
)


def skewed_gaussian(
    B: float,
    B_0: float,
    sigma_m: float,
    sigma_p: float,
) -> float:
    """
    Skewed Gaussian distribution (as defined in musrfit).
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


def skewed_gaussian_mean(
    B_0: float,
    sigma_m: float,
    sigma_p: float,
) -> float:
    """
    Mean of the skewed Gaussian distribution (as defined in musrfit).
    """
    return B_0 + np.sqrt(2.0 / np.pi) * (sigma_p - sigma_m) / gamma_mu


def skewed_gaussian_mean_error(
    B_0: float,
    B_0_error: float,
    sigma_m: float,
    sigma_m_error: float,
    sigma_p: float,
    sigma_p_error: float,
) -> float:
    """
    Uncertainty of the mean of the skewed Gaussian distribution (as defined in musrfit).
    """

    # first derivatives
    dskg_dB_0 = 1.0
    dskg_dsigma_m = -1.0 * np.sqrt(2.0 / np.pi) / gamma_mu
    dskg_dsigma_p = 1.0 * np.sqrt(2.0 / np.pi) / gamma_mu
    # error propagation
    return np.sqrt(
        np.square(dskg_dB_0 * B_0_error)
        + np.square(dskg_dsigma_m * sigma_m_error)
        + np.square(dskg_dsigma_p * sigma_p_error)
    )
