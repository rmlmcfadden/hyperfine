"""Calculate demagnetization factors for simple geometries.
"""

import numpy as np
from scipy import integrate


def N_ellipsoid(a: float, b: float, c: float) -> float:
    """Demagnetization factor N for a general ellipsoid.

    Calculate the demagnetization factor for an ellipsoid with dimensions:
    2a x 2b x 2c (field applied parallel to the c-axis).

    Calculated using numeric integration according to Eqs. (20) and (21) in:

    R. Prozorov and V. G. Kogan,
    "Effective demagnetizing factors of diamagnetic samples of various shapes",
    Phys. Rev. Appl. 10, 014030 (2018).
    https://doi.org/10.1103/PhysRevApplied.10.014030


    Args:
        a: Ellipsoid semi-axis.
        b: Ellipsoid semi-axis.
        c: Ellipsoid semi-axis.

    Returns:
        The demagnetization factor N.
    """

    # ensure that the semi-axes have physically meaningful dimensions
    assert a >= 0.0
    assert b >= 0.0
    assert c >= 0.0

    # integrand in Eq. (20)
    def integrand(s: float) -> float:
        # convenience variable in Eq. (21)
        R = np.sqrt((s + 1.0) * (s + (b / a) ** 2) * (s + (c / a) ** 2))
        denominator = (s + (c / a) ** 2) * R
        return 1.0 / denominator

    # perform the numeric integration using adaptive Gaussian quadrature
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
    result, _ = integrate.quad(
        integrand,
        0.0,  # lower integration limit
        np.inf,  # upper integration limit
        epsabs=np.sqrt(np.finfo(float).eps),  # absolute error tolerance
        epsrel=np.sqrt(np.finfo(float).eps),  # relative error tolerance
        limit=np.iinfo(np.int32).max,  # maximum number of subintervals
    )

    # terms in Eq. (20) outside of the integrand
    prefactor = 0.5 * (b / a) * (c / a)

    # return the demagnetizing factor
    return prefactor * result


def N_cuboid(a: float, b: float, c: float) -> float:
    """Effective demagnetizing factor for a rectangular cuboid.

    Calculate the (effective) demagnetization factor N for a rectangular cuboid
    using an approximate analytic formula that interpolates between the
    limiting cases of an infinitely thin (c -> 0, N -> 1) and an infinitely
    thick (c -> +inf, N -> 0) sample. The field is assumed to be applied
    parallel to the c-axis.

    Calculated using Eq. (22) from:

    R. Prozorov and V. G. Kogan,
    "Effective demagnetizing factors of diamagnetic samples of various shapes",
    Phys. Rev. Appl. 10, 014030 (2018).
    https://doi.org/10.1103/PhysRevApplied.10.014030

    Args:
        a: Length of the rectangular cuboid.
        b: Width of the rectangular cuboid.
        c: Height of the rectangular cuboid.

    Returns:
        The demagnetization factor N.
    """

    # ensure that the semi-axes have physically meaningful dimensions
    assert a >= 0.0
    assert b >= 0.0
    assert c >= 0.0

    return (4.0 * a * b) / (4.0 * a * b + 3.0 * c * (a + b))
