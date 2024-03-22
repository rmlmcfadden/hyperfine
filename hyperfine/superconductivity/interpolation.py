"""Interpolation formulas.

Common interpolation formulas used in superconductivity as:
* Phenomenological models describing (universal) data trends.
* Analytic approximations for theoretical expressions whose solutions must be derived numerically.
"""

from typing import Annotated, Callable, Optional, Sequence
import numpy as np


def gap_cos_eV(
    T: Annotated[Sequence[float], 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None] = 1.0,
) -> float:
    """Approximate expression for the superconducting energy gap.

    See Eq. (1) in:
    https://doi.org/10.1103/PhysRev.149.368

    Args:
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting energy gap at 0 K (eV).

    Returns:
        The superconducting energy gap (eV).
    """

    t = T / T_c
    arg = (np.pi / 2.0) * (t**2)

    return Delta_0 * np.sqrt(np.cos(arg))


def gap_tanh_eV(
    T: Annotated[Sequence[float], 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None] = 1.0,
) -> float:
    """Approximate expression for the superconducting energy gap.

    See Eq. (5) in:
    https://doi.org/10.1016/S0921-4534(02)02319-5

    Args:
        T: Absolute temperature.
        T_c: Absolute superconducting transition temperature.
        Delta_0: Superconducting energy gap at 0 K (eV).

    Returns:
        The superconducting energy gap (eV).
    """

    rt = T_c / T

    return Delta_0 * np.tanh(1.82 * np.power(1.018 * (rt - 1.0), 0.51))


def lambda_two_fluid_nm(
    T: Annotated[Sequence[float], 0.0:None],
    T_c: Annotated[float, 0.0:None],
    lambda_0: Annotated[float, 0.0:None],
    n: Annotated[float, 0.0:None] = 4.0,
) -> float:
    """Gorter-Casimir two-fluid expression for the temperature dependence of
    the magnetic penetration depth.

    Args:
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        lambda_0: Magnetic penetration depth at 0 K (nm).

    Returns:
        The magnetic penetration depth at finite temperature.
    """

    return lambda_0 / np.sqrt(1.0 - np.power(T / T_c, n))
