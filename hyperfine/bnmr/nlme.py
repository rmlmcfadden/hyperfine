import numpy as np
from scipy import constants
from typing import Annotated


def kappa_GL(
    penetration_depth_nm: Annotated[float, 0:None],
    London_penetration_depth_nm: Annotated[float, 0:None],
    BCS_coherence_length_nm: Annotated[float, 0:None],
) -> float:
    """
    See e.g., Eq. 2.45 in Section 2.4.4 of https://cds.cern.ch/record/1518890
    (note: beware the typos in the surrounding equations...)
    """

    factor = 2.0 * np.sqrt(3.0) / np.pi

    ratio = (penetration_depth_nm * penetration_depth_nm) / (
        London_penetration_depth_nm * BCS_coherence_length_nm
    )

    return factor * ratio


def B_c_prime_T(
    lambda_GL_nm: Annotated[float, 0:None],
    xi_GL_nm: Annotated[float, 0:None],
) -> float:
    """
    See Eq. (2) in: https://doi.org/10.1103/PhysRevResearch.4.013156
    """

    # convert the GL parameters from nm to m
    m_per_nm = 1e-9
    lambda_GL_m = lambda_GL_nm * m_per_nm
    xi_GL_m = xi_GL_nm * m_per_nm

    # evaluate the "critical" field
    return constants.value("mag. flux quantum") / (
        np.power(2.0, 3.0 / 2.0) * np.pi * lambda_GL_m * xi_GL_m
    )


def scaled_penetration_depth_nm(
    effective_field_T: Annotated[float, 0:None],
    effective_penetration_depth_nm: Annotated[float, 0:None],
    London_penetration_depth_nm: Annotated[float, 0:None],
    BCS_coherence_length_nm: Annotated[float, 0:None],
) -> float:
    """
    See Eq. (2) in: https://doi.org/10.1103/PhysRevResearch.4.013156
    """

    # make sure the effective penetration depth isn't below its "floor"
    if effective_penetration_depth_nm < London_penetration_depth_nm:
        print(
            "WARNING: 'effective_penetration_depth_nm' < 'London_penetration_depth_nm'!"
        )
        effective_penetration_depth_nm = London_penetration_depth_nm

    # calculate the Ginzburg-Landau parameter and coherence length
    kappa = kappa_GL(
        effective_penetration_depth_nm,
        London_penetration_depth_nm,
        BCS_coherence_length_nm,
    )
    xi = effective_penetration_depth_nm / kappa

    # thermodynamic critical field in Tesla
    B_c_prime = B_c_prime_T(effective_penetration_depth_nm, xi)

    # Eq. (2) in https://doi.org/10.1103/PhysRevResearch.4.013156
    numerator = kappa * (kappa + np.power(2.0, 1.5))
    denominator = 8.0 * np.square(kappa + np.sqrt(2.0))

    # return the scaled penetration depth
    return (
        1.0 + (numerator / denominator) * np.square(effective_field_T / B_c_prime)
    ) * effective_penetration_depth_nm
