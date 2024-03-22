"""Expressions from Bardeen-Cooper-Schrieffer (BCS) theory for nonlocal field screening.

J. Bardeen, L. N. Cooper, and J. R. Schrieffer,
Theory of Superconductivity,
Phys. Rev. 108, 1175 (1957).
https://doi.org/10.1103/PhysRev.108.1175

J. Halbritter,
On the penetration of the magnetic field into a superconductor,
Z. Physik 243, 201–219 (1971).
https://doi.org/10.1007/BF01394851
"""

from typing import Annotated, Callable, Optional, Sequence
import numpy as np
from scipy import constants, integrate
from .interpolation import gap_cos_eV


def _a(
    T: Annotated[float, 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None],
) -> float:
    """???

    Args:
        T: Absolute temperature.
        T_c: Absolute superconducting transition temperature.
        Delta_0: Superconducting energy gap (eV) at 0 K.

    Returns:
        The ???.
    """

    k_B = constants.value("Boltzmann constant in eV/K")

    return (np.pi * k_B * T) / gap_cos_eV(T, T_c, Delta_0)


def _f_n(
    n: Annotated[int, 0:None],
    T: Annotated[float, 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None],
) -> float:
    """???

    Args:
        n: Summation index.
        T: Absolute temperature.
        T_c: Absolute superconducting transition temperature.
        Delta_0: Superconducting energy gap (eV) at 0 K.

    Returns:
        The ???.
    """

    return np.sqrt(1.0 + ((2.0 * n + 1.0) * _a(T, T_c, Delta_0)) ** 2)


def _xi_n(
    n: Annotated[int, 0:None],
    T: Annotated[float, 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None],
    xi_0: Annotated[float, 0.0:None],
    ell: Annotated[float, 0.0:None],
) -> float:
    """???

    Args:
        n: Summation index.
        T: Absolute temperature.
        T_c: Absolute superconducting transition temperature.
        Delta_0: Superconducting energy gap (eV) at 0 K.
        xi_0: BCS coherence length (nm) at 0 K.
        ell: Electron mean-free-path (nm).

    Returns:
        The ???.
    """

    normalized_gap = gap_cos_eV(T, T_c, Delta_0) / Delta_0

    recip_xi_n = (2.0 / np.pi) * (_f_n(n, T, T_c, Delta_0) / xi_0) * normalized_gap + (
        1.0 / ell
    )

    return 1.0 / recip_xi_n


def _Lambda_n(
    n: Annotated[int, 0:None],
    T: Annotated[float, 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None],
    xi_0: Annotated[float, 0.0:None],
    ell: Annotated[float, 0.0:None],
    lambda_L: Annotated[float, 0.0:None],
) -> float:
    """???

    Args:
        n: Summation index.
        T: Absolute temperature.
        T_c: Absolute superconducting transition temperature.
        Delta_0: Superconducting energy gap (eV) at 0 K.
        xi_0: BCS coherence length (nm) at 0 K.
        ell: Electron mean-free-path (nm).

    Returns:
        The ???.
    """

    term_1 = ((lambda_L**2) * (_f_n(n, T, T_c, Delta_0) ** 3)) / (
        2.0 * _a(T, T_c, Delta_0)
    )

    term_2 = 1.0 + (_xi_n(n, T, T_c, Delta_0, xi_0, ell) / ell)

    return term_1 * term_2


def _g(
    x: Annotated[float, 0.0:None],
) -> float:
    """???

    Args:
        x: Dimensionless input.

    Returns:
        ???
    """

    return (3.0 / 2.0) * (1.0 / x**3) * ((1.0 + x**2) * np.arctan(x) - x)


def K_BCS(
    q: Annotated[float, 0.0:None],
    T: Annotated[float, 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None],
    xi_0: Annotated[float, 0.0:None],
    ell: Annotated[float, 0.0:None],
    lambda_L: Annotated[float, 0.0:None],
) -> float:
    """The BCS response function.

    Args:
        q: wavevector.
        T: Absolute temperature.
        T_c: Absolute superconducting transition temperature.
        Delta_0: Superconducting energy gap (eV) at 0 K.
        xi_0: BCS coherence length (nm) at 0 K.
        ell: Electron mean-free-path (nm).
        lambda_L: London penetration depth (nm).

    Returns:
        The BCS response function.
    """

    K = 0.0
    delta_K = np.finfo(float).max
    # tolerance = 1.0e-8
    # tolerance = np.finfo(float).eps
    tolerance = np.sqrt(np.finfo(float).eps)
    # tolerance = np.cbrt(np.finfo(float).eps)
    n = 0

    while delta_K > tolerance:

        _xi = _xi_n(n, T, T_c, Delta_0, xi_0, ell)
        _lambda = _Lambda_n(n, T, T_c, Delta_0, xi_0, ell, lambda_L)

        delta_K = _g(q * _xi) / _lambda

        # print(f"K{n} = {delta_K}")

        K += delta_K
        n = n + 1

    return K


def integrand_diffusive(
    q: Annotated[float, 0.0:None],
    T: Annotated[float, 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None],
    xi_0: Annotated[float, 0.0:None],
    ell: Annotated[float, 0.0:None],
    lambda_L: Annotated[float, 0.0:None],
) -> float:
    """The BCS integrand for diffusive scattering.

    Args:
        q: Wavevector (1/nm).
        T: Absolute temperature (K).
        T_c: Critical temperature (K).
        Delta_0: Superconducting gap at 0 K (eV).
        xi_0: BCS coherence length (nm).
        ell: Electron mean-free-path (nm).
        lambda_L: London penetration depth (nm).

    Returns:
        The BCS integrand for diffusive scattering.

    """

    return np.log1p(K_BCS(q, T, T_c, Delta_0, xi_0, ell, lambda_L) / q**2)


def lambda_diffusive(
    T: Annotated[float, 0.0:None],
    T_c: Annotated[float, 0.0:None],
    Delta_0: Annotated[float, 0.0:None],
    xi_0: Annotated[float, 0.0:None],
    ell: Annotated[float, 0.0:None],
    lambda_L: Annotated[float, 0.0:None],
) -> float:
    """Evaluate the magnetic penetration depth within BCS theory.

    The calculation assumes diffuse scattering of electrons at the material's surface.

    Args:
        T: Absolute temperature.
        T_c: Absolute superconducting transition temperature.
        Delta_0: Superconducting energy gap (eV) at 0 K.
        xi_0: BCS coherence length (nm) at 0 K.
        ell: Electron mean-free-path (nm).
        lambda_L: London penetration depth (nm).

    Returns:
        The magnetic penetration depth (nm) at 0 K.
    """

    integral, _ = integrate.quad(
        integrand_diffusive,
        0.0,
        np.inf,
        args=(T, T_c, Delta_0, xi_0, ell, lambda_L),
        full_output=False,
        epsabs=np.cbrt(np.finfo(float).eps),  # 1.4e-8
        epsrel=np.cbrt(np.finfo(float).eps),  # 1.4e-8
        limit=np.iinfo(np.int32).max,  # default = 50
        # points=(0.0),
    )

    return np.pi / integral
