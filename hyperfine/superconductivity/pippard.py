"""Expressions from Pippard's nonlocal screening model.

A. B. Pippard,
An experimental and theoretical study of the relation between magnetic field
and current in a superconductor,
Proc. R. Soc. London A 216, 547-568 (1953).
https://doi.org/10.1098/rspa.1953.0040
"""

from typing import Annotated, Sequence
import numpy as np
from scipy import constants, integrate, interpolate
from .interpolation import gap_cos_eV, gap_tanh_eV, lambda_two_fluid_nm
from ._muhlschlegel import _interp_gap_bcs


def j_0_t(
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
) -> float:
    r"""Bardeen-Cooper-Schrieffer (BCS) range function :math:`J(0,T)` (i.e., at :math:`R = 0`).

    Args:
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting gap energy at 0 K (eV).

    Returns:
        The BCS range function at temperature :math:`T`.

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import pippard

           T = np.linspace(0.0, 1.0, 100)
           args = (1.0, 1.43e-3)
           plt.plot(T, pippard.j_0_t(T, *args), "-")
           plt.xlabel("$T / T_{c}$")
           plt.ylabel("$J(0,T)$")
           plt.show()

    """

    # convenience terms
    k_B = constants.value("Boltzmann constant in eV/K")
    t = T / T_c
    e = _interp_gap_bcs(t)
    arg = Delta_0 * e / (2.0 * k_B * T)

    # assume [λ(t) / λ(0)]^2 = 1 / (1 - t^2), instead of the t^4 dependence of the two-fluid model.
    # this gives for closer agreement with the (numeric) BCS result!
    # this choice is identical to the implementation used by musrfit

    return (e / (1.0 - t**2)) * np.tanh(arg)


def j_0_t_wc(
    T: Sequence[float],
    T_c: Annotated[float, 0:None],
) -> Sequence[float]:
    r"""Bardeen-Cooper-Schrieffer (BCS) range function :math:`J(0,T)` (i.e., at :math:`R = 0`).

    This implementation assumes weak electron-phonon coupling.

    Args:
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).

    Returns:
        The BCS range function at temperature :math:`T`.

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import pippard

           T = np.linspace(0.0, 1.0, 100)
           T_c = 1.0
           plt.plot(T, pippard.j_0_t_wc(T, T_c), "-")
           plt.xlabel("$T / T_{c}$")
           plt.ylabel("$J(0,T)$")
           plt.show()

    """

    # convenience terms
    k_B = constants.value("Boltzmann constant in eV/K")
    gamma_EM = 0.57721566490153286060651209008240243104215933593992
    t = T / T_c
    e = _interp_gap_bcs(t)
    c = np.pi / np.exp(gamma_EM)  # 1.764...
    arg = (c / 2.0) * (e / t)

    # assume [λ(t) / λ(0)]^2 = 1 / (1 - t^2), instead of the t^4 dependence of the two-fluid model.
    # this gives for closer agreement with the (numeric) BCS result!
    # this choice is identical to the implementation used by musrfit

    return (e / (1.0 - t**2)) * np.tanh(arg)


def xi_Pippard(
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    r"""Evaluate the effective Pippard coherence length for a finite electron mean-free-path.

    Args:
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting gap energy at 0 K (eV).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length at 0 K (nm).
        alpha: numerical constant on the order of unity.

    Returns:
        The effective Pippard coherence length (nm).

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import pippard

           T = np.linspace(0.0, 1.0, 100)
           args = (1.0, 1.43e-3, 200.0, 50.0)
           xi = np.array([pippard.xi_Pippard(tt, *args) for tt in T])
           plt.plot(T, xi, "-")
           plt.xlabel("$T / T_{c}$")
           plt.ylabel(r"$\xi_{0}(T)$ (nm)")
           plt.show()

    """

    recip_xi_0 = j_0_t(T, T_c, Delta_0) / xi_0
    recip_l = alpha / l

    return 1.0 / (recip_xi_0 + recip_l)


def K_Pippard(
    q: Annotated[float, 0:None],
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    r"""Evaluate the Pippard response function.

    Args:
        q: wavevector (1/nm).
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting gap energy at 0 K (eV).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length at 0 K (nm).
        alpha: numerical constant on the order of unity.

    Returns:
        The Pippard response function K(q) at q.

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import pippard

           q = np.logspace(-4, 4, 200)
           args = (0.0, 10.0, 1.43e-3, 30.0, 300.0, 40.0)
           k = np.array([pippard.K_Pippard(qq, *args) for qq in q])
           plt.plot(q, k, "-")
           plt.xlabel("$q$ (nm$^{-1}$)")
           plt.ylabel(r"$K_{\mathrm{Pippard}}(q)$ (nm$^{-2}$)")
           plt.xscale("log")
           plt.yscale("log")
           plt.show()

    """

    # calculate the temperature-dependent values for the coherence length xi
    # and magnetic penetration depth lambda_T
    xi = xi_Pippard(T, T_c, Delta_0, l, xi_0, alpha)
    lambda_T = lambda_two_fluid_nm(T, T_c, lambda_L)

    # define a low-q cutoff to prevent numeric oscillations from finite
    # floating-point precision
    q_cutoff = np.sqrt(np.finfo(float).eps)

    if q > q_cutoff:
        a = (1.0 / lambda_T**2) * (xi / xi_0)
        b = (3.0 / 2.0) * (1.0 / (q * xi) ** 3)
        c = 1.0 + (q * xi) ** 2
        d = np.arctan(q * xi)
        e = q * xi

        return a * (b * (c * d - e))
    else:
        a_cutoff = (1.0 / lambda_T**2) * (xi / xi_0)
        b_cutoff = (3.0 / 2.0) * (1.0 / (q_cutoff * xi) ** 3)
        c_cutoff = 1.0 + (q_cutoff * xi) ** 2
        d_cutoff = np.arctan(q_cutoff * xi)
        e_cutoff = q_cutoff * xi

        return a_cutoff * (b_cutoff * (c_cutoff * d_cutoff - e_cutoff))


def integrand_diffusive(
    q: Annotated[float, 0:None],
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """Integrand for calculating the magnetic penetration depth.

    The calculation assumes diffuse scattering of electrons at the material's surface.

    Args:
        q: wavevector (1/nm).
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting gap energy at 0 K (eV).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length at 0 K (nm).
        alpha: numerical constant on the order of unity.

    Returns:
        The integrand at a given wavevector q.

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import pippard

           q = np.logspace(-4, 4, 200)
           args = (0.0, 10.0, 1.43e-3, 30.0, 300.0, 40.0)
           id = np.array([pippard.integrand_diffusive(qq, *args) for qq in q])
           plt.plot(q, id, "-")
           plt.xlabel("$q$ (nm$^{-1}$)")
           plt.ylabel("$I(q)$")
           plt.xscale("log")
           plt.yscale("log")
           plt.show()

    """

    K = K_Pippard(q, T, T_c, Delta_0, lambda_L, l, xi_0, alpha)

    return np.log1p(K / q**2)


def integrand_specular_profile(
    q: Annotated[float, 0:None],
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
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
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting gap energy at 0 K (eV).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length at 0 K (nm).
        alpha: numerical constant on the order of unity.

    Returns:
        The integrand at a given wavevector q.

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import pippard

           q = np.logspace(-4, 4, 200)
           args = (0.0, 10.0, 1.43e-3, 30.0, 300.0, 40.0)
           isp = np.array([pippard.integrand_specular_profile(qq, *args) for qq in q])
           plt.plot(q, isp, "-")
           plt.xlabel("$q$ (nm$^{-1}$)")
           plt.ylabel("$I(q)$ (nm)")
           plt.xscale("log")
           plt.yscale("log")
           plt.show()

    """

    K = K_Pippard(q, T, T_c, Delta_0, lambda_L, l, xi_0, alpha)

    return q / (K + q * q)


def specular_profile(
    z: Annotated[float, 0:None],
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """Field screening profile B(z).

    The calculation assumes specular scattering of electrons at the material's surface.

    Args:
        z: depth (nm).
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting gap energy at 0 K (eV).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length at 0 K (nm).
        alpha: numerical constant on the order of unity.

    Returns:
        The field screening profile B(z).

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import pippard

           z = np.linspace(0.0, 200.0, 100)
           args = (0.0, 10.0, 1.43e-3, 30.0, 600.0, 300.0)
           b = np.array([pippard.specular_profile(zz, *args) for zz in z])
           plt.plot(z, b, "-")
           plt.xlabel("$z$ (nm)")
           plt.ylabel("$B(z)$ (nm)")
           plt.show()

    """

    if z == 0.0:
        return 1.0

    integral, _ = integrate.quad(
        integrand_specular_profile,
        0.0,
        np.inf,
        args=(T, T_c, Delta_0, lambda_L, l, xi_0, alpha),
        full_output=False,
        epsabs=np.sqrt(np.finfo(float).eps),  # 1.4e-8
        epsrel=np.sqrt(np.finfo(float).eps),  # 1.4e-8
        limit=np.iinfo(np.int32).max // 4,  # default = 50
        # points=(0.0),
        weight="sin",
        wvar=z,
    )

    return (2.0 / np.pi) * integral


def specular_profile_dl(
    z: Annotated[float, 0:None],
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
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
        T: Absolute temperature (K).
        T_c: Superconducting transition temperature (K).
        Delta_0: Superconducting gap energy at 0 K (eV).
        lambda_L: London penetration depth (nm).
        l: electron mean-free-path (nm).
        xi_0: Pippard coherence length at 0 K (nm).
        alpha: numerical constant on the order of unity.
        dl: non-superconducting dead layer (nm).

    Returns:
        The field screening profile B(z).

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import pippard

           z = np.linspace(0.0, 200.0, 100)
           args = (0.0, 10.0, 1.43e-3, 30.0, 600.0, 300.0, 1.0, 10.0)
           b = np.array([pippard.specular_profile_dl(zz, *args) for zz in z])
           plt.plot(z, b, "-")
           plt.xlabel("$z$ (nm)")
           plt.ylabel("$B(z)$ (nm)")
           plt.show()

    """

    z_corr = z - dl

    if z_corr < 0.0:
        return 1.0

    return specular_profile(z_corr, T, T_c, Delta_0, lambda_L, l, xi_0, alpha)


def lambda_diffusive(
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
    l: Annotated[float, 0:None],
    lambda_L: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> float:
    """
    Evaluate the magnetic penetration depth within Pippard theory.

    The calculation assumes diffuse scattering of electrons at the material's surface.

    Args:
        T: Annotated[float, 0:None],
        T_c: Annotated[float, 0:None],
        Delta_0: Annotated[float, 0:None],
        l: electron mean-free-path (nm).
        lambda_L: London penetration depth (nm).
        xi_0: Pippard coherence length at 0 K (nm).
        alpha: numerical constant on the order of unity.

    Returns:
        The magnetic penetration depth (nm) at 0 K.
    """

    integral, _ = integrate.quad(
        integrand_diffusive,
        0.0,
        np.inf,
        args=(T, T_c, Delta_0, lambda_L, l, xi_0, alpha),
        full_output=False,
        epsabs=np.sqrt(np.finfo(float).eps),  # 1.4e-8
        epsrel=np.sqrt(np.finfo(float).eps),  # 1.4e-8
        limit=np.iinfo(np.int32).max // 4,  # default = 50
        # points=(0.0),
    )

    return np.pi / integral


def lambda_diffusive2(
    T: Annotated[float, 0:None],
    T_c: Annotated[float, 0:None],
    Delta_0: Annotated[float, 0:None],
    l: Sequence[float],
    lambda_L: Annotated[float, 0:None],
    xi_0: Annotated[float, 0:None],
    alpha: Annotated[float, 0:None] = 1.0,
) -> Sequence[float]:
    """
    Evaluate the magnetic penetration depth within Pippard theory.

    The calculation assumes diffuse scattering of electrons at the material's surface.

    Args:
        T: Annotated[float, 0:None],
        T_c: Annotated[float, 0:None],
        Delta_0: Annotated[float, 0:None],
        l: electron mean-free-path (nm).
        lambda_L: London penetration depth (nm).
        xi_0: Pippard coherence length at 0 K (nm).
        alpha: numerical constant on the order of unity.

    Returns:
        The magnetic penetration depth (nm) at 0 K.
    """

    results = np.array(len(l))

    for i, ll in enumerate(l):
        results[i] = lambda_diffusive(T, T_c, Delta_0, ll, lambda_L, xi_0, alpha)

    return results
