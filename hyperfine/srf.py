import numpy as np


def B_two_exp(
    depth_nm: float,
    B_0_G: float,
    thickness_1_nm: float,
    lambda_1_nm: float,
    lambda_2_nm: float,
) -> float:
    """
    Naive London model for a superconductor-superconductor bilayer.
    """
    return np.piecewise(
        depth_nm,
        [
            depth_nm <= 0.0,  # vacuum
            (depth_nm > 0.0) & (depth_nm <= thickness_1_nm),  # top layer
            depth_nm > thickness_1_nm,  # bottom layer
        ],
        [
            lambda depth_nm: B_0_G,
            lambda depth_nm: B_0_G * np.exp(-depth_nm / lambda_1_nm),
            lambda depth_nm: B_0_G
            * np.exp(-thickness_1_nm / lambda_1_nm)
            * np.exp(-(depth_nm - thickness_1_nm) / lambda_2_nm),
        ],
    )


# bulk
def B_B(x: float, B_0: float, lambda_B: float) -> float:
    """
    Meissner screening profile for a bulk superconductor (London limit).
    """
    return np.piecewise(
        x,
        [
            x <= 0.0,
            x > 0.0,
        ],
        [
            lambda x: B_0,
            lambda x: B_0 * np.exp(-x / lambda_B),
        ],
    )


# thin film
def B_F(x: float, B_0: float, d_F: float, lambda_F: float):
    """
    Meissner screening profile for a thin film superconductor (London limit).
    """
    return np.piecewise(
        x,
        [
            x <= 0.0,
            (x > 0.0) & (x <= d_F),
            x > d_F,
        ],
        [
            lambda x: B_0,
            lambda x: B_0
            * np.cosh((0.5 * d_F - x) / lambda_F)
            / np.cosh(0.5 * d_F / lambda_F),
            lambda x: B_0,
        ],
    )


# Kubo review (2017)
# superconductor-superconductor interface
def B_SS(x, B_0, d_S, lambda_1, lambda_2):
    """
    Meissner screening profile for superconductor-superconductor (SS) bilayer.
    The expressions are derived from a London model.
    """
    return np.piecewise(
        x,
        [
            # outside of heterostructure
            x <= 0.0,
            # Eq. (21)
            (x > 0.0) & (x <= d_S),
            # Eq. (22)
            x > d_S,
        ],
        [
            # outside of heterostructure
            lambda x: B_0,
            # Eq. (21)
            lambda x: B_0
            * (
                np.cosh((d_S - x) / lambda_1)
                + (lambda_2 / lambda_1) * np.sinh((d_S - x) / lambda_1)
            )
            / (
                np.cosh(d_S / lambda_1)
                + (lambda_2 / lambda_1) * np.sinh(d_S / lambda_1)
            ),
            # Eq. (22)
            lambda x: B_0
            * np.exp(-(x - d_S) / lambda_2)
            / (
                np.cosh(d_S / lambda_1)
                + (lambda_2 / lambda_1) * np.sinh(d_S / lambda_1)
            ),
        ],
    )


# T. Kubo et al. Appl. Phys. Lett. 104, 032603 (2014).
# https://doi.org/10.1063/1.4862892
# Eqs. (1) to (3)
def B_SIS(x, B_0, d_S, lambda_1, d_I, lambda_2):
    """
    Meissner screening profile for superconductor-insulator-superconductor (SIS) multilayer.
    The expressions are derived from a London model.
    """
    #
    brackets = (lambda_2 / lambda_1) + (d_I / lambda_1)
    #
    # numerator = np.cosh((d_S - x) / lambda_1) + brackets * np.sinh((d_S - x) / lambda_1)
    #
    # denominator = np.cosh(d_S / lambda_1) + brackets * np.sinh(d_S / lambda_1)
    #
    return np.piecewise(
        x,
        [
            # outside of heterostructure
            x <= 0.0,
            # Eq. (1)
            (x > 0.0) & (x <= d_S),
            # Eq. (2)
            (x > d_S) & (x <= (d_S + d_I)),
            # Eq. (3)
            x > (d_S + d_I),
        ],
        [
            # outside of heterostructure
            lambda x: B_0,
            # Eq. (1)
            lambda x: B_0
            * (np.cosh((d_S - x) / lambda_1) + brackets * np.sinh((d_S - x) / lambda_1))
            / (np.cosh(d_S / lambda_1) + brackets * np.sinh(d_S / lambda_1)),
            # Eq. (2)
            lambda x: B_0
            * 1.0
            / (np.cosh(d_S / lambda_1) + brackets * np.sinh(d_S / lambda_1)),
            # Eq. (3)
            lambda x: B_0
            * np.exp(-(x - d_S - d_I) / lambda_2)
            / (np.cosh(d_S / lambda_1) + brackets * np.sinh(d_S / lambda_1)),
        ],
    )
