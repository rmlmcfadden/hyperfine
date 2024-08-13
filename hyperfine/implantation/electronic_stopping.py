"""Electronic stopping functions"""

from typing import Sequence, Annotated
import numpy as np


# ICRU Report 49, Eqs. (3.1a), (3.1b), and (3.1c)
# @jax.jit
def proton_stopping_cross_section_high_energy(
    T: Sequence[float],
    A_2: Annotated[float, 0:None],
    A_3: Annotated[float, 0:None],
    A_4: Annotated[float, 0:None],
    A_5: Annotated[float, 0:None],
) -> Sequence[float]:
    r"""Low-energy proton electronic stopping cross section in the 'high-energy' region.

    Empirical Varelas-Biersack parameterization of the proton electronic stopping cross section for scaled energies :math:`T \geq 10` keV.

    See Eqs. (3.1a)-(3.1c) in: https://doi.org/10.1093/jicru_os25.2.18

    Args:
        T: Scaled projectile energy (keV).
        A_2: Empirical stopping coefficient :math:`A_{2}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV\ :sup:`-0.45`\ ).
        A_3: Empirical stopping coefficient :math:`A_{3}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV).
        A_4: Empirical stopping coefficient :math:`A_{4}` (keV).
        A_5: Empirical stopping coefficient :math:`A_{5}` (keV\ :sup:`-1`\ ).

    Returns:
        The proton stopping cross section at scaled energy :math:`T` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1`\ ).
    """

    e_low = A_2 * np.power(T, 0.45)
    e_high = (A_3 / T) * np.log(1.0 + (A_4 / T) + A_5 * T)
    return (e_low * e_high) / (e_low + e_high)


# ICRU Report 49, Eq. (3.2)
# @jax.jit
def proton_stopping_cross_section_low_energy(
    T: Sequence[float],
    A_1: Annotated[float, 0:None],
) -> Sequence[float]:
    r"""Low-energy proton electronic stopping cross section in the 'low-energy' region.

    Empirical Varelas-Biersack parameterization of the proton electronic stopping cross section for scaled energies :math:`T < 10` keV.

    See Eq. (3.2) in: https://doi.org/10.1093/jicru_os25.2.18

    Args:
        T: Scaled projectile energy (keV).
        A_1: Empirical stopping coefficient :math:`A_{2}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV\ :sup:`-0.5`\ ).

    Returns:
        The proton stopping cross section at scaled energy :math:`T` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1`\ ).
    """

    return A_1 * np.sqrt(T)


# ICRU Report 49, Eqs. (3.1) and (3.2)
# @jax.jit
def proton_stopping_cross_section(
    T: Sequence[float],
    A_1: Annotated[float, 0:None],
    A_2: Annotated[float, 0:None],
    A_3: Annotated[float, 0:None],
    A_4: Annotated[float, 0:None],
    A_5: Annotated[float, 0:None],
) -> Sequence[float]:
    r"""Low-energy proton electronic stopping cross section.

    Empirical Varelas-Biersack parameterization of the proton electronic stopping cross section for scaled energies :math:`T` around the stopping power maximum.

    See Eqs. (3.1a)-(3.1c) & (3.2) in: https://doi.org/10.1093/jicru_os25.2.18

    Args:
        T: Scaled projectile energy (keV).
        A_1: Empirical stopping coefficient :math:`A_{2}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV\ :sup:`-0.5`\ ).
        A_2: Empirical stopping coefficient :math:`A_{2}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV\ :sup:`-0.45`\ ).
        A_3: Empirical stopping coefficient :math:`A_{3}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV).
        A_4: Empirical stopping coefficient :math:`A_{4}` (keV).
        A_5: Empirical stopping coefficient :math:`A_{5}` (keV\ :sup:`-1`\ ).

    Returns:
        The proton stopping cross section at scaled energy :math:`T` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1`\ ).
    """

    return np.piecewise(
        T,
        [
            T < 10,  # low-energy region
            T >= 10,  # high-energy region
        ],
        [
            lambda T: proton_stopping_cross_section_low_energy(T, A_1),
            lambda T: proton_stopping_cross_section_high_energy(T, A_2, A_3, A_4, A_5),
        ],
    )


# convenience function
# @jax.jit
def calculate_A_1(
    A_2: Annotated[float, 0:None],
    A_3: Annotated[float, 0:None],
    A_4: Annotated[float, 0:None],
    A_5: Annotated[float, 0:None],
) -> float:
    """Calculate the empirical stopping coefficient :math:`A_{1}` using the other four :math:`A_{i}`\ s.

    Useful when there is no data in the 'low-energy' region (i.e., when :math:`T < 10` keV) and smooth continuity of the Varelas-Biersack expression is desired.

    Args:
        A_2: Empirical stopping coefficient :math:`A_{2}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV\ :sup:`-0.45`\ ).
        A_3: Empirical stopping coefficient :math:`A_{3}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV).
        A_4: Empirical stopping coefficient :math:`A_{4}` (keV).
        A_5: Empirical stopping coefficient :math:`A_{5}` (keV\ :sup:`-1`\ ).

    Returns:
        The empirical stopping coefficient :math:`A_{1}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV\ :sup:`-0.5`).
    """

    # crossover point between the high- and low-energy regions.
    T = 10.0

    return proton_stopping_cross_section_high_energy(T, A_2, A_3, A_4, A_5) / np.sqrt(T)


# four parameter version for fitting
# @jax.jit
def proton_stopping_cross_section4(
    T: Sequence[float],
    A_2: Annotated[float, 0:None],
    A_3: Annotated[float, 0:None],
    A_4: Annotated[float, 0:None],
    A_5: Annotated[float, 0:None],
) -> Sequence[float]:
    r"""Low-energy proton electronic stopping cross section using four :math:`A_{i}`s.

    Empirical Varelas-Biersack parameterization of the proton electronic stopping cross section for scaled energies :math:`T` around the stopping power maximum. This version uses four of the five :math:`A_{i}`\ s (\ :math:`i = {2,3,4,5}`\ ) and implicitly determines :math:`A_{1}` from them. This alternative parameterization is useful when fitting stopping cross section data without any measurements in the 'low-energy' portion (i.e., when :math:`T < 10` keV).

    See Eqs. (3.1a)-(3.1c) & (3.2) in: https://doi.org/10.1093/jicru_os25.2.18

    Args:
        T: Scaled projectile energy (keV).
        A_2: Empirical stopping coefficient :math:`A_{2}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV\ :sup:`-0.45`\ ).
        A_3: Empirical stopping coefficient :math:`A_{3}` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1` keV).
        A_4: Empirical stopping coefficient :math:`A_{4}` (keV).
        A_5: Empirical stopping coefficient :math:`A_{5}` (keV\ :sup:`-1`\ ).

    Returns:
        The proton stopping cross section at scaled energy :math:`T` (10\ :sup:`-15` eV cm\ :sup:`2` atom\ :sup:`-1`\ ).

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.implantation.electronic_stopping import proton_stopping_cross_section4

           A = (6.73, 1.03e4, 2.8e2, 3.9e-3)
           T = np.logspace(0, 4, 200)
           S_e = proton_stopping_cross_section4(T, *A)

           plt.plot(T, S_e, "-")
           plt.xlabel("$T$ (keV)")
           plt.ylabel("$S_{e}$ ($10^{-15}$ eV cm$^{2}$ atom$^{-1}$)")
           plt.xscale("log")
           plt.show()
    """

    # calculate the "missing" coefficent
    A_1 = calculate_A_1(A_2, A_3, A_4, A_5)

    # evaluate the stopping cross section
    return proton_stopping_cross_section(T, A_1, A_2, A_3, A_4, A_5)
