import numpy as np


def tl_B_c1_T(
    temperature_K: float,
    critical_temperature_K: float,
    critical_field_T: float,
) -> float:
    """
    Tuyn's law for the lower critical field.
    """
    return np.piecewise(
        temperature_K,
        [
            temperature_K <= critical_temperature_K,
            temperature_K > critical_temperature_K,
        ],
        [
            lambda x: critical_field_T
            * (1.0 - (temperature_K / critical_temperature_K) ** 2),
            lambda x: 0.0,
        ],
    )


def tl_B_c2_T(
    temperature_K: float,
    critical_temperature_K: float,
    critical_field_T: float,
) -> float:
    """
    Tuyn's law for the upper critical field.
    """
    return np.piecewise(
        temperature_K,
        [
            temperature_K <= critical_temperature_K,
            temperature_K > critical_temperature_K,
        ],
        [
            lambda x: critical_field_T
            * (1.0 - (temperature_K / critical_temperature_K) ** 2)
            / (1.0 + (temperature_K / critical_temperature_K) ** 2),
            lambda x: 0.0,
        ],
    )


def magnetic_volume_susceptibility(
    applied_field_T: float,
    temperature_K: float,
    critical_temperature_K: float,
    lower_critical_field_T: float,
    upper_critical_field_T: float,
    normal_state_susceptibility: float = 0.0,
) -> float:
    """
    Magnetic (volume) susceptibility (SI units) for a type-II superconductor.
    """

    # calculate B_c1 and B_c2 at finite temperature using Tuyn's law
    B_c1_T = tl_B_c1_T(temperature_K, critical_temperature_K, lower_critical_field_T)
    B_c2_T = tl_B_c2_T(temperature_K, critical_temperature_K, upper_critical_field_T)

    # print(f"B_c1({temperature_K} K) = {B_c1_T} T")
    # print(f"B_c2({temperature_K} K) = {B_c2_T} T")

    def _decay_exp(
        b_T: float,
        b_c1_T: float,
        b_c2_T: float,
    ) -> float:

        # coordinate corrections & length scale calculation
        x = b_T - b_c1_T
        dx = b_c2_T - b_c1_T

        if dx <= 0.0:
            return 0.0 * b_T

        # amplitudes & weightings used in the decay
        amplitudes = [0.5, 0.5]
        weights = [20.0, 10.0]

        # multiexponential decay
        result = 0.0
        for a, w in zip(amplitudes, weights):
            result += a * np.exp(-w * x / dx)

        return result

    def _decay_parab(
        b_T: float,
        b_c1_T: float,
        b_c2_T: float,
    ) -> float:

        # coordinate corrections & length scale calculation
        x = b_T - b_c1_T
        dx = b_c2_T - b_c1_T

        if dx <= 0.0:
            return 0.0 * b_T

        n = 4

        k = 1.0 / (dx**n)

        return k * (b_T - b_c2_T) ** n

    def _decay_log(
        b_T: float,
        b_c1_T: float,
        b_c2_T: float,
    ) -> float:

        if b_c2_T - b_c1_T <= 0.0:
            return 0.0 * b_T

        numerator = b_c1_T * np.log(b_c2_T / b_T)
        denominator = np.log(b_c2_T / b_c1_T)

        magnetization = numerator / denominator

        return magnetization / b_T

    superconducting_state_susceptibility = np.piecewise(
        applied_field_T,
        [
            (applied_field_T < B_c1_T) & (applied_field_T < B_c2_T),
            (applied_field_T >= B_c1_T) & (applied_field_T <= B_c2_T),
            applied_field_T > B_c2_T,
        ],
        [
            lambda x: -1.0,
            lambda x: -1.0 * _decay_log(x, B_c1_T, B_c2_T),
            lambda x: 0.0,
        ],
    )

    return normal_state_susceptibility + superconducting_state_susceptibility
