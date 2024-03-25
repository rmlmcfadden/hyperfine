"""Tabulated quantities computed from Bardeen-Cooper-Schrieffer (BCS) theory.

See:
B. Mühlschlegel, *Die thermodynamischen Funktionen des Supraleiters*,
Z. Phys. **155**, 313–327 (1959).
https://doi.org/10.1007/BF01332932
"""

import numpy as np
from scipy import interpolate


#: Reduced temperatures :math:`t = T / T_c`.
_t = np.array(
    [
        1.00,
        0.98,
        0.96,
        0.94,
        0.92,
        0.90,
        0.88,
        0.86,
        0.84,
        0.82,
        0.80,
        0.78,
        0.76,
        0.74,
        0.72,
        0.70,
        0.68,
        0.66,
        0.64,
        0.62,
        0.60,
        0.58,
        0.56,
        0.54,
        0.52,
        0.50,
        0.48,
        0.46,
        0.44,
        0.42,
        0.40,
        0.38,
        0.36,
        0.34,
        0.32,
        0.30,
        0.28,
        0.26,
        0.24,
        0.22,
        0.20,
        0.18,
        0.16,
        0.14,
    ]
)[
    ::-1
]  # reverse the order


#: Reduced gap energies :math:`\Delta (T) / \Delta(0)`.
_e = np.array(
    [
        0.0000,
        0.2436,
        0.3416,
        0.4148,
        0.4749,
        0.5263,
        0.5715,
        0.6117,
        0.6480,
        0.6810,
        0.7110,
        0.7386,
        0.7640,
        0.7874,
        0.8089,
        0.8288,
        0.8471,
        0.8640,
        0.8796,
        0.8939,
        0.9070,
        0.9190,
        0.9299,
        0.9399,
        0.9488,
        0.9569,
        0.9641,
        0.9704,
        0.9760,
        0.9809,
        0.9850,
        0.9885,
        0.9915,
        0.9938,
        0.9957,
        0.9971,
        0.9982,
        0.9989,
        0.9994,
        0.9997,
        0.9999,
        1.0000,
        1.0000,
        1.0000,
    ]
)[
    ::-1
]  # reverse the order


#: Reduced temperature dependence of the Bardeen-Cooper-Schrieffer (BCS) reduced energy gap.
_interp_gap_bcs = interpolate.interp1d(
    _t,
    _e,
    kind="linear",
    axis=-1,
    copy=True,
    bounds_error=False,
    fill_value=(1.0, 0.0),
    assume_sorted=True,
)
