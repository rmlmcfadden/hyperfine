import numpy as np
from scipy import constants
import jax
from typing import Annotated

# use double-precision numbers
jax.config.update("jax_enable_x64", True)


def korringa_rate(
    temperature: float,
    slope: Annotated[float, 0:None],
) -> float:
    r"""Spin-lattice relaxation rate from a Korringa mechanism.

    Args:
        temperature: Absolute temperature (K).
        slope: Korringa slope (s\ :sup:`-1` K\ :sup:`-1`).

    Returns:
        The Korringa SLR rate (s\ :sup:`-1`).
    """
    return temperature * slope


def arrhenius_rate(
    temperature: float,
    attempt_frequency__s: Annotated[float, 0:None],
    activation_energy_eV: Annotated[float, 0:None],
) -> float:
    r"""Thermally activated rate following an Arrhenius law.

    Args:
        temperature: Absolute temperature (K).
        attempt_freqency__s: Attempt frequency or pre-exponential factor (s\ :sup:`-1`).
        activation_energy_eV: Activation energy or energy barrier (eV).

    Returns:
        The Arrhenius rate (s\ :sup:`-1`).
    """

    return attempt_frequency__s * jax.numpy.exp(
        -1.0
        * activation_energy_eV
        / (constants.value("Boltzmann constant in eV/K") * temperature)
    )


def amplitude_1(
    fraction_1: float,
    hop_rate_12: float,
    slr_rate_1: float,
    slr_rate_2: float,
) -> float:
    r"""Amplitude of site 1.

    Args:
        fraction_1: Fraction occupying site 1.
        hop_rate_12: Rate of site change from 1 -> 2 (s\ :sup:`-1`).
        slr_rate_1: SLR rate at site 1 (s\ :sup:`-1`).
        slr_rate_2: SLR rate at site 2 (s\ :sup:`-1`).

    Returns:
        The amplitude at site 1.
    """
    numerator = (
        hop_rate_12 - slr_rate_2 + slr_rate_1
    ) * fraction_1 - hop_rate_12 * fraction_1
    denominator = hop_rate_12 - slr_rate_2 + slr_rate_1

    return numerator / denominator


def amplitude_2(
    fraction_1: float,
    hop_rate_12: float,
    slr_rate_1: float,
    slr_rate_2: float,
) -> float:
    r"""Amplitude of site 2.

    Args:
        fraction_1: Fraction occupying site 1.
        hop_rate_12: Rate of site change from 1 -> 2 (s\ :sup:`-1`).
        slr_rate_1: SLR rate at site 1 (s\ :sup:`-1`).
        slr_rate_2: SLR rate at site 2 (s\ :sup:`-1`).

    Returns:
        The amplitude at site 2.
    """

    fraction_2 = 1.0 - fraction_1

    numerator = hop_rate_12 - slr_rate_2 * fraction_2 + slr_rate_1 * fraction_1
    denominator = hop_rate_12 - slr_rate_2 + slr_rate_1

    return numerator / denominator


def M(
    fraction_1: float,
    hop_rate_12: float,
    slr_rate_1: float,
    slr_rate_2: float,
) -> float:
    r"""Convenience term.

    Args:
        fraction_1: Fraction occupying site 1.
        hop_rate_12: Rate of site change from 1 -> 2 (s\ :sup:`-1`).
        slr_rate_1: SLR rate at site 1 (s\ :sup:`-1`).
        slr_rate_2: SLR rate at site 2 (s\ :sup:`-1`).

    Returns:
        The convenience term.
    """

    denominator = hop_rate_12 - slr_rate_2 + slr_rate_1
    return fraction_1 * (1.0 - hop_rate_12 / denominator)


def average_slr_rate(
    temperature: float,
    initial_fraction_1: Annotated[float, 0:1],
    attempt_frequency__s: Annotated[float, 0:None],
    activation_energy_eV: Annotated[float, 0:None],
    korringa_slope_1__s_K: Annotated[float, 0:None],
    korringa_slope_2__s_K: Annotated[float, 0:None],
) -> float:
    r"""Average SLR rate within the two-site model.

    Args:
        temperature: Absolute temperature (K).
        initial_faction_1: Fraction of population initially in site 1.
        attempt_freqency__s: Attempt frequency or pre-exponential factor (s\ :sup:`-1`).
        activation_energy_eV: Activation energy or energy barrier (eV).
        korringa_slope_1__s_K: Korringa slope at site 1 (s\ :sup:`-1` K\ :sup:`-1`).
        korringa_slope_2__s_K: Korringa slope at site 2 (s\ :sup:`-1` K\ :sup:`-1`).

    Returns:
        The site-averaged SLR rate (s\ :sup:`-1`).
    """

    hop_rate_12 = arrhenius_rate(
        temperature, attempt_frequency__s, activation_energy_eV
    )
    slr_rate_1 = korringa_rate(temperature, korringa_slope_1__s_K)
    slr_rate_2 = korringa_rate(temperature, korringa_slope_2__s_K)

    fraction_1 = M(initial_fraction_1, hop_rate_12, slr_rate_1, slr_rate_2)
    fraction_2 = 1.0 - fraction_1

    # weighted average
    return fraction_1 * slr_rate_1 + fraction_2 * slr_rate_2


def polarization_1(
    time: float,
    p_1_0: float,
    hop_rate_12: float,
    slr_rate_1: float,
) -> float:
    r"""Polarization at site 1.

    Args:
        time: Time after implantation (s).
        p_1_0: Initial polarization at site 1.
        hop_rate_12: Rate of site change from 1 -> 2 (s\ :sup:`-1`).
        slr_rate_1: SLR rate at site 1 (s\ :sup:`-1`).

    Returns:
        The polarization at site 1.
    """
    return p_1_0 * np.exp(-(slr_rate_1 + hop_rate_12) * time)


def polarization_2(
    time: float,
    p_1_0: float,
    p_2_0: float,
    hop_rate_12: float,
    slr_rate_1: float,
    slr_rate_2: float,
) -> float:
    r"""Polarization at site 2.

    Args:
        time: Time after implantation (s).
        p_1_0: Initial polarization at site 1.
        p_2_0: Initial polarization at site 2.
        hop_rate_12: Rate of site change from 1 -> 2 (s\ :sup:`-1`).
        slr_rate_1: SLR rate at site 1 (s\ :sup:`-1`).
        slr_rate_2: SLR rate at site 2 (s\ :sup:`-1`).

    Returns:
        The polarization at site 2.
    """

    numerator = (
        (p_2_0 + p_1_0) * hop_rate_12 - p_2_0 * slr_rate_2 + p_2_0 * slr_rate_1
    ) * np.exp(-slr_rate_2 * time) - p_1_0 * hop_rate_12 * np.exp(
        -(slr_rate_1 + hop_rate_12) * time
    )
    denominator = hop_rate_12 - slr_rate_2 + slr_rate_1

    return numerator / denominator


def polarization(
    time: float,
    p_1_0: float,
    p_2_0: float,
    hop_rate_12: float,
    slr_rate_1: float,
    slr_rate_2: float,
) -> float:
    r"""The total spin polarization within the two-site model.

    Args:
        time: Time after implantation (s).
        p_1_0: Initial polarization at site 1.
        p_2_0: Initial polarization at site 2.
        hop_rate_12: Rate of site change from 1 -> 2 (s\ :sup:`-1`).
        slr_rate_1: SLR rate at site 1 (s\ :sup:`-1`).
        slr_rate_2: SLR rate at site 2 (s\ :sup:`-1`).

    Returns:
        The total polarization.
    """

    return polarization_1(time, p_1_0, hop_rate_12, slr_rate_1) + polarization_2(
        time, p_1_0, p_2_0, hop_rate_12, slr_rate_1, slr_rate_2
    )
