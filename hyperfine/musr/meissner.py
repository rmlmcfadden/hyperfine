import numpy as np
import pandas as pd
from scipy import constants, integrate, interpolate, special
from .. import distributions


# class for handling the details required for calculating the mean field
# at a given energy by integrating over the implantation profile
class DepthAveragingCalculator:
    # constructor
    def __init__(self, file_name, interpolation="linear"):
        # read the implantation distribution parameterss
        self.df = pd.read_csv(file_name, delimiter=",")

        # interpolation functions for the stopping distribution parameters
        self.z_max_1 = interpolate.interp1d(
            self.df["Energy (keV)"],
            self.df["z_max_1"],
            kind=interpolation,
            copy=True,
            bounds_error=True,
            assume_sorted=False,
        )
        self.alpha_1 = interpolate.interp1d(
            self.df["Energy (keV)"],
            self.df["alpha_1"],
            kind=interpolation,
            copy=True,
            bounds_error=True,
            assume_sorted=False,
        )
        self.beta_1 = interpolate.interp1d(
            self.df["Energy (keV)"],
            self.df["beta_1"],
            kind=interpolation,
            copy=True,
            bounds_error=True,
            assume_sorted=False,
        )
        self.fraction_1 = interpolate.interp1d(
            self.df["Energy (keV)"],
            self.df["fraction_1"],
            kind=interpolation,
            copy=True,
            bounds_error=True,
            assume_sorted=False,
        )
        self.z_max_2 = interpolate.interp1d(
            self.df["Energy (keV)"],
            self.df["z_max_2"],
            kind=interpolation,
            copy=True,
            bounds_error=True,
            assume_sorted=False,
        )
        self.alpha_2 = interpolate.interp1d(
            self.df["Energy (keV)"],
            self.df["alpha_2"],
            kind=interpolation,
            copy=True,
            bounds_error=True,
            assume_sorted=False,
        )
        self.beta_2 = interpolate.interp1d(
            self.df["Energy (keV)"],
            self.df["beta_2"],
            kind=interpolation,
            copy=True,
            bounds_error=True,
            assume_sorted=False,
        )

    # the final implantation distribution function
    def stopping_distribution(
        self,
        z: float,
        alpha_1: float,
        beta_1: float,
        z_max_1: float,
        fraction_1: float,
        alpha_2: float,
        beta_2: float,
        z_max_2: float,
    ) -> float:
        return distributions.modified_beta_2(
            z, alpha_1, beta_1, z_max_1, fraction_1, alpha_2, beta_2, z_max_2
        )

    # convenience function
    def stopping_distribution_e(self, depth_nm: float, energy_keV: float) -> float:
        return self.stopping_distribution(
            depth_nm,
            self.alpha_1(energy_keV),
            self.beta_1(energy_keV),
            self.z_max_1(energy_keV),
            self.fraction_1(energy_keV),
            self.alpha_2(energy_keV),
            self.beta_2(energy_keV),
            self.z_max_2(energy_keV),
        )

    # calculate the mean implantation depth
    def calculate_mean_depth(self, energy_keV: float) -> float:
        # weighted average
        return distributions.modified_beta_2_mean(
            self.alpha_1(energy_keV),
            self.beta_1(energy_keV),
            self.z_max_1(energy_keV),
            self.fraction_1(energy_keV),
            self.alpha_2(energy_keV),
            self.beta_2(energy_keV),
            self.z_max_2(energy_keV),
        )

    # London equation (Meissner state)
    def _london(
        self,
        z: float,
        applied_field_G: float,
        penetration_depth_nm: float,
        dead_layer_nm: float,
    ) -> float:
        return np.piecewise(
            z,
            [
                z <= dead_layer_nm,
                z > dead_layer_nm,
            ],
            [
                lambda x: applied_field_G + x * 0.0,
                lambda x: applied_field_G
                * np.exp(-(x - dead_layer_nm) / penetration_depth_nm),
            ],
        )

    def london_ms(
        self,
        z: float,
        applied_field_G: float,
        penetration_depth_nm: float,
        dead_layer_nm: float,
        demagnetization_factor: float = 0.0,
    ) -> float:
        return self._london(z, applied_field_G, penetration_depth_nm, dead_layer_nm) / (
            1.0 - demagnetization_factor
        )

    # helper function
    def calculate_mean_field(
        self,
        energy_keV: float,
        applied_field_G: float,
        dead_layer_nm: float,
        penetration_depth_nm: float,
        demagnetization_factor: float,
    ):
        # product of the london model w/ the implantion distribution
        def integrand(z: float) -> float:
            return self.london_ms(
                z,
                applied_field_G,
                penetration_depth_nm,
                dead_layer_nm,
                demagnetization_factor,
            ) * self.stopping_distribution_e(z, energy_keV)

        # do the numeric integration using adaptive Gaussian quadrature
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
        result = integrate.quad(
            integrand,
            0.0,  # lower integration limit
            max(  # upper integration limit
                np.max(self.z_max_1(energy_keV)), np.max(self.z_max_2(energy_keV))
            ),
            epsabs=np.sqrt(np.finfo(float).eps),  # absolute error tolerance
            epsrel=np.sqrt(np.finfo(float).eps),  # relative error tolerance
            limit=np.iinfo(np.int32).max,  # maximum number of subintervals
            points=[  # potential singularities/discontinuities in the integrand
                0.0,
                dead_layer_nm,
                self.z_max_1(energy_keV),
                self.z_max_2(energy_keV),
            ],
        )
        return result[0]

    # functor version of calculate_mean_field (can take an array of energies)!
    def __call__(
        self,
        energy_keV: np.array,
        applied_field_G: float,
        dead_layer_nm: float,
        penetration_depth_nm: float,
        demagnetization_factor: float,
    ) -> np.array:
        # energy_keV = np.asarray(energy_keV)
        results = np.empty(energy_keV.size)
        for i, e_keV in enumerate(energy_keV):
            results[i] = self.calculate_mean_field(
                e_keV,
                applied_field_G,
                dead_layer_nm,
                penetration_depth_nm,
                demagnetization_factor,
            )
        return results
