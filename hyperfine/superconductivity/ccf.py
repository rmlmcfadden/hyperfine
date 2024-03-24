"""Expressions for the counter-current flow (CCF) model of superconducting multilayers.


See, e.g.,:
T. Kubo,
Multilayer coating for higher accelerating fields in superconducting
radio-frequency cavities: a review of theoretical aspects,
Supercond. Sci. Technol. 30 023001 (2017).
https://doi.org/10.1088/1361-6668/30/2/023001
"""

from typing import Annotated, Sequence
import numpy as np
from .london import screening_profile_bulk


def screening_profile_biexp(
    depth_nm: Sequence[float],
    applied_field_G: Annotated[float, 0:None],
    dead_layer_nm: Annotated[float, 0:None],
    thickness_1_nm: Annotated[float, 0:None],
    lambda_1_nm: Annotated[float, 0:None],
    lambda_2_nm: Annotated[float, 0:None],
) -> Sequence[float]:
    """Naive biexponential Meissner screening in a superconductor-superconductor (SS) bilayer.

    This model is for comparative purposes only; it does properly account for
    the boundary/continuity conditions between two adjacent superconductors.

    Args:
        depth_nm: Depth below the surface (nm).
        applied_field_G: Applied magnetic field (G).
        dead_layer_nm: Non-superconducting surface dead layer thickness (nm).
        thickness_1_nm: Superconducting film thickness (nm).
        lambda_1_nm: Magnetic penetration depth of the superconducting film (nm).
        lambda_2_nm: Magnetic penetration depth of the superconducting substrate (nm).

    Returns:
        The Meissner screening profile at position ``depth_nm`` (G).
    """

    # correct for the dead layer
    z_nm = depth_nm - dead_layer_nm
    t_nm = thickness_1_nm - dead_layer_nm

    return np.piecewise(
        z_nm,
        [
            z_nm <= 0.0,  # vacuum
            (z_nm > 0.0) & (z_nm <= t_nm),  # top layer
            depth_nm > t_nm,  # bottom layer
        ],
        [
            lambda x: applied_field_G,
            lambda x: applied_field_G * np.exp(-x / lambda_1_nm),
            lambda x: applied_field_G
            * np.exp(-t_nm / lambda_1_nm)
            * np.exp(-(x - t_nm) / lambda_2_nm),
        ],
    )


def screening_profile_ss(
    x: Sequence[float],
    B_0: Annotated[float, 0:None],
    d_S: Annotated[float, 0:None],
    lambda_1: Annotated[float, 0:None],
    lambda_2: Annotated[float, 0:None],
) -> Sequence[float]:
    """Meissner screening for a superconductor-superconductor (SS) bilayer.

    The expression was derived from a London model of the SS layers.

    See Eqs. (21) and (22) in:
    T. Kubo,
    Multilayer coating for higher accelerating fields in superconducting
    radio-frequency cavities: a review of theoretical aspects,
    Supercond. Sci. Technol. 30 023001 (2017).
    https://doi.org/10.1088/1361-6668/30/2/023001

    Args:
        x: Depth below the suface (nm).
        B_0: Applied magnetic field (G).
        d_S: Superconducting film thickness (nm).
        lambda_1: Magnetic penetration depth of the superconducting film (nm).
        lambda_2: Magnetic penetration depth of the superconducting substrate (nm).

    Returns:
        The Meissner screening profile at position ``x`` (G).
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


def screening_profile_sis(
    x: Sequence[float],
    B_0: Annotated[float, 0:None],
    d_S: Annotated[float, 0:None],
    lambda_1: Annotated[float, 0:None],
    d_I: Annotated[float, 0:None],
    lambda_2: Annotated[float, 0:None],
) -> Sequence[float]:
    """Meissner screening profile for superconductor-insulator-superconductor (SIS) multilayer.

    The expression was derived from a London model of the SIsS layers.

    See Eqs. (1) to (3) in:
    T. Kubo et al. Appl. Phys. Lett. 104, 032603 (2014).
    https://doi.org/10.1063/1.4862892

    Args:
        x: Depth below the suface (nm).
        B_0: Applied magnetic field (G).
        d_S: Superconducting film thickness (nm).
        lambda_1: Magnetic penetration depth of the superconducting film (nm).
        d_I: Insulating layer thickness (nm).s
        lambda_2: Magnetic penetration depth of the superconducting substrate (nm).

    Returns:
        The Meissner screening profile at position ``x`` (G).

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
