"""Superconductivity-related formulas.
"""

from . import bcs, pippard, interpolation, intertype, london, ccf
import numpy as np


def get_penetration_depth_at_0K(
    penetration_depth_nm: float,
    penetration_depth_error_nm: float,
    temperature_K: float,
    temperature_error_K: float,
    critical_temperature_K: float = 9.25,
    critical_temperature_error_K: float = 0.01,
) -> tuple[float, float]:
    # calculate the extrapolated value
    value = penetration_depth_nm * np.sqrt(
        1.0 - np.power(temperature_K / critical_temperature_K, 4.0)
    )

    # calculate the uncertainty
    fac = np.sqrt(1.0 - np.power(temperature_K / critical_temperature_K, 4.0))
    dv_dp = fac
    # partial derivatives
    dv_dt = (
        -2.0
        * penetration_depth_nm
        * np.power(temperature_K, 3.0)
        * np.power(critical_temperature_K, -4.0)
        / fac
    )
    dv_dc = (
        2.0
        * penetration_depth_nm
        * np.power(temperature_K, 4.0)
        * np.power(critical_temperature_K, -5.0)
        / fac
    )
    # uncertainty
    uncertainty = np.sqrt(
        np.square(dv_dp * penetration_depth_error_nm)
        + np.square(dv_dt * temperature_error_K)
        + np.square(dv_dc * critical_temperature_error_K)
    )

    # return the tuple of values
    return (value, uncertainty)


def get_mean_free_path_at_0K(
    penetration_depth_nm: float,
    penetration_depth_error_nm: float,
    london_penetration_depth_nm: float = 30.0,
    london_penetration_depth_error_nm: float = 0.0,
    coherence_length_nm: float = 39.0,
    coherence_length_error_nm: float = 0.0,
) -> tuple[float, float]:
    # calculate the value
    value = (
        -np.square(london_penetration_depth_nm)
        * coherence_length_nm
        / (np.square(london_penetration_depth_nm) - np.square(penetration_depth_nm))
    )

    # calculate the uncertainty
    denom = np.square(london_penetration_depth_nm) - np.square(penetration_depth_nm)
    # partial derivatives
    dv_dl = (
        2.0
        * coherence_length_nm
        * london_penetration_depth_nm
        * np.square(penetration_depth_nm)
        / np.square(denom)
    )
    dv_dc = -np.square(london_penetration_depth_nm) / denom
    dv_dp = (
        -2.0
        * coherence_length_nm
        * np.square(london_penetration_depth_nm)
        * penetration_depth_nm
        / np.square(denom)
    )
    # uncertainty
    uncertainty = np.sqrt(
        np.square(dv_dl * london_penetration_depth_error_nm)
        + np.square(dv_dc * coherence_length_error_nm)
        + np.square(dv_dp * penetration_depth_error_nm)
    )

    # return the tuple of quantities
    return (value, uncertainty)


def get_effective_coherence_length_at_0K(
    coherence_length_nm: float,
    coherence_length_error_nm: float,
    mean_free_path_nm: float,
    mean_free_path_error_nm: float,
) -> tuple[float, float]:
    # calculate the value
    value = (coherence_length_nm * mean_free_path_nm) / (
        coherence_length_nm + mean_free_path_nm
    )
    # calculate the uncertainty
    denom = np.square(coherence_length_nm + mean_free_path_nm)
    decl_dcl = np.square(mean_free_path_nm) / denom
    decl_dmfp = np.square(coherence_length_nm) / denom
    uncertainty = np.sqrt(
        np.square(decl_dcl * coherence_length_error_nm)
        + np.square(decl_dmfp * mean_free_path_error_nm)
    )
    # return the tuple of quantities
    return (value, uncertainty)
