from typing import Annotated, Sequence
import numpy as np
from scipy import constants, integrate
from .interpolation import gap_cos_eV, gap_tanh_eV, lambda_two_fluid_nm


def j_0(
    T: Annotated[Sequence[float], 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
) -> float:
    """???

    Args:
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting gap energy at 0 K (eV).

    Returns:
        ???
    """

    k_B = constants.value("Boltzmann constant in eV/K")

    # use exponent n = 2 for closer agreement with the BCS result!
    # this is identical to the implementation used in musrfit
    l_norm = lambda_two_fluid_nm(T, T_c, 1.0, 2.0)

    g_norm = gap_cos_eV(T, T_c, Delta_0) / Delta_0
    arg = gap_cos_eV(T, T_c, Delta_0) / (2.0 * k_B * T)

    return (l_norm**2) * g_norm * np.tanh(arg)


def xi_Pippard(
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """
    Evaluate the effective Pippard coherence length for a finite electron mean-free-path.

    Args:
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length (nm) at 0 K.
        alpha: numerical constant on the order of unity.

    Returns:
        The effective Pippard coherence length (nm).
    """

    recip_xi_0 = 1.0 / xi_0
    recip_l = 1.0 / (alpha * l)

    return 1.0 / (recip_xi_0 + recip_l)


def K_Pippard(
    q: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """
    Evaluate the Pippard response function.

    Args:
        q: wavevector (1/nm).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length (nm) at 0 K.
        alpha: numerical constant on the order of unity.

    Returns:
        The Pippard response function K(q) at q.
    """

    xi = xi_Pippard(l, xi_0, alpha)

    q_cutoff = np.sqrt(np.finfo(float).eps)

    if q > q_cutoff:
        a = (1.0 / lambda_L**2) * (xi / xi_0)
        b = (3.0 / 2.0) * (1.0 / (q * xi) ** 3)
        c = 1.0 + (q * xi) ** 2
        d = np.arctan(q * xi)
        e = q * xi

        return a * (b * (c * d - e))
    else:
        a_cutoff = (1.0 / lambda_L**2) * (xi / xi_0)
        b_cutoff = (3.0 / 2.0) * (1.0 / (q_cutoff * xi) ** 3)
        c_cutoff = 1.0 + (q_cutoff * xi) ** 2
        d_cutoff = np.arctan(q_cutoff * xi)
        e_cutoff = q_cutoff * xi

        return a_cutoff * (b_cutoff * (c_cutoff * d_cutoff - e_cutoff))


def integrand_diffusive(
    q: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """
    Integrand for calculating the magnetic penetration depth.

    The calculation assumes diffuse scattering of electrons at the material's surface.

    Args:
        q: wavevector (1/nm).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length (nm) at 0 K.
        alpha: numerical constant on the order of unity.

    Returns:
        The integrand at a given wavevector q.
    """

    K = K_Pippard(q, lambda_L, l, xi_0, alpha)

    return np.log1p(K / q**2)


def integrand_specular_profile(
    q: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """
    Integrand for calculating the magnetic penetration depth.

    The calculation assumes diffuse scattering of electrons at the material's surface.

    Args:
        q: wavevector (1/nm).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length (nm) at 0 K.
        alpha: numerical constant on the order of unity.

    Returns:
        The integrand at a given wavevector q.
    """

    K = K_Pippard(q, lambda_L, l, xi_0, alpha)

    return q / (K + q * q)


def specular_profile(
    z: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """
    Field screening profile B(z).

    The calculation assumes specular scattering of electrons at the material's surface.

    Args:
        z: depth (nm).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length (nm) at 0 K.
        alpha: numerical constant on the order of unity.

    Returns:
        The field screening profile B(z).
    """

    if z == 0.0:
        return 1.0

    integral, _ = integrate.quad(
        integrand_specular_profile,
        0.0,
        np.inf,
        args=(lambda_L, l, xi_0, alpha),
        full_output=False,
        epsabs=np.sqrt(np.finfo(float).eps),  # 1.4e-8
        epsrel=np.sqrt(np.finfo(float).eps),  # 1.4e-8
        limit=np.iinfo(np.int32).max,  # default = 50
        # points=(0.0),
        weight="sin",
        wvar=z,
    )

    return (2.0 / np.pi) * integral


def specular_profile_dl(
    z: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
    dl: Annotated[float, 0:None] = 0.0,
) -> float:
    """
    Field screening profile B(z).

    The calculation assumes specular scattering of electrons at the material's surface.

    Args:
        z: depth (nm).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length (nm) at 0 K.
        alpha: numerical constant on the order of unity.
        dl: non-superconducting dead layer (nm).

    Returns:
        The field screening profile B(z).
    """

    z_corr = z - dl

    if z_corr < 0.0:
        return 1.0

    return specular_profile(z_corr, lambda_L, l, xi_0, alpha)


def lambda_diffusive(
    l: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """
    Evaluate the magnetic penetration depth within Pippard theory.

    The calculation assumes diffuse scattering of electrons at the material's surface.

    Args:
        l: electron mean-free-path (nm).
        lambda_L: London penetration depth (nm).
        xi_0: Pippard coherence length (nm) at 0 K.
        alpha: numerical constant on the order of unity.

    Returns:
        The magnetic penetration depth (nm) at 0 K.
    """

    """
    integral, _ = quadax.quadts(
        integrand_diffusive,
        (0.0, np.inf),
        args=(lambda_L, l, xi_0, alpha),
        full_output=False,
        epsabs=1.4e-8,
        epsrel=1.4e-8,
        max_ninter=50,  # np.iinfo(np.int32).max causes memory usage to balloon
        order=61,  # {41, 61, 81, 101}
    )
    """

    integral, _ = integrate.quad(
        integrand_diffusive,
        0.0,
        np.inf,
        args=(lambda_L, l, xi_0, alpha),
        full_output=False,
        epsabs=np.sqrt(np.finfo(float).eps),  # 1.4e-8
        epsrel=np.sqrt(np.finfo(float).eps),  # 1.4e-8
        limit=np.iinfo(np.int32).max,  # default = 50
        # points=(0.0),
    )

    return np.pi / integral


def lambda_diffusive2(
    l: Sequence[float],
    lambda_L: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> Sequence[float]:
    """ """

    results = np.array(len(l))

    for i, ll in enumerate(l):
        results[i] = lambda_diffusive(ll, lambda_L, xi_0, alpha)

    return results
