"""Intertype superconductivity domain boundaries.
"""

from typing import Annotated, Sequence
import numpy as np


def _kappa_intertype(
    t: Sequence[float],
    c: Annotated[float, None:None],
) -> Sequence[float]:
    r"""Helper function for intertype critical parameter :math:`\kappa_{x}^{y}`.

    See:
    A. Vagov et al., Superconductivity between standard types: Multiband versus
    single-band materials, Phys. Rev. B 93, 174503 (2016).
    https://doi.org/10.1103/PhysRevB.93.174503

    Args:
        t: Reduced temperature (:math:`t \in [0, 1]`).
        c: Numeric coefficient.

    Returns:
        The critical parameter :math:`\kappa_{x}^{y}`.

    """

    # Ginzburg-Landau type-I/II boundary.
    kappa_0 = 1.0 / np.sqrt(2.0)

    tau = 1.0 - t

    return kappa_0 * (1.0 + c * tau)


def kappa_s_star(
    t: Sequence[float],
) -> Sequence[float]:
    r"""Intertype critical parameter :math:`\kappa_{s}^{*}`.

    See Eq. (28) in:
    A. Vagov et al., Superconductivity between standard types: Multiband versus
    single-band materials, Phys. Rev. B 93, 174503 (2016).
    https://doi.org/10.1103/PhysRevB.93.174503

    Args:
        t: Reduced temperature (:math:`t \in [0, 1]`).

    Returns:
        The critical parameter :math:`\kappa_{s}^{*}`.

    """

    return _kappa_intertype(t, -0.027)


def kappa_1_star(
    t: Sequence[float],
) -> Sequence[float]:
    r"""Intertype critical parameter :math:`\kappa_{1}^{*}`.

    See Eq. (29) in:
    A. Vagov et al., Superconductivity between standard types: Multiband versus
    single-band materials, Phys. Rev. B 93, 174503 (2016).
    https://doi.org/10.1103/PhysRevB.93.174503

    Args:
        t: Reduced temperature (:math:`t \in [0, 1]`).

    Returns:
        The critical parameter :math:`\kappa_{1}^{*}`.

    """

    return _kappa_intertype(t, 0.093)


def kappa_li_star(
    t: Sequence[float],
) -> Sequence[float]:
    r"""Intertype critical parameter :math:`\kappa_{li}^{*}`.

    See Eq. (30) in:
    A. Vagov et al., Superconductivity between standard types: Multiband versus
    single-band materials, Phys. Rev. B 93, 174503 (2016).
    https://doi.org/10.1103/PhysRevB.93.174503

    Args:
        t: Reduced temperature (:math:`t \in [0, 1]`).

    Returns:
        The critical parameter :math:`\kappa_{li}^{*}`.

    """

    return _kappa_intertype(t, 0.95)


def kappa_2_star(
    t: Sequence[float],
) -> Sequence[float]:
    r"""Intertype critical parameter :math:`\kappa_{li}^{*}`.

    See Eq. (30) in:
    A. Vagov et al., Superconductivity between standard types: Multiband versus
    single-band materials, Phys. Rev. B 93, 174503 (2016).
    https://doi.org/10.1103/PhysRevB.93.174503

    Args:
        t: Reduced temperature (:math:`t \in [0, 1]`).

    Returns:
        The critical parameter :math:`\kappa_{2}^{*}`.

    """

    return _kappa_intertype(t, -0.407)
