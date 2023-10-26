import numpy as np
import pandas as pd
from scipy import constants, integrate, interpolate, special
from .. import distributions
from .meissner import DepthAveragingCalculator


# class for handling the details required for calculating the mean field
# at a given energy by integrating over the implantation profile
class DepthAveragingCalculatorMLM(DepthAveragingCalculator):
    # constructor
    def __init__(
        self,
        file_name: str,
        interpolation: str = "linear",
        numeric_solution_table: str = "T120t12hr.txt",
    ):

        super().__init__(file_name, interpolation)

        # numerical solution to the modified London model (for 120C baking)
        self.df_mlm = pd.read_csv(
            numeric_solution_table,
            delimiter="\t",
            names=[
                "z (nm)",
                "B/B_applied",
            ],
        )

        """
        self.df_mlm["z_corr (nm)"] = np.linspace(
            self.df_mlm["z (nm)"].min(),
            self.df_mlm["z (nm)"].max(),
            self.df_mlm["z (nm)"].size,
        )
        """

        # print( self.df_mlm["z (nm)"].equals(self.df_mlm["z_corr (nm)"]) )

        self.mlm = interpolate.interp1d(
            self.df_mlm["z (nm)"],
            self.df_mlm["B/B_applied"],
            kind=interpolation,
            copy=True,
            bounds_error=True,
            assume_sorted=False,
        )

    def screening_profile(
        self,
        depth_nm: float,
        applied_field_G: float,
        dead_layer_nm: float,
        demagnetization_factor: float,
    ) -> float:
        effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

        return np.piecewise(
            depth_nm,
            [
                depth_nm <= dead_layer_nm,
                depth_nm > dead_layer_nm,
            ],
            [
                lambda z: effective_field_G + 0.0 * z,
                lambda z: effective_field_G * self.mlm(z - dead_layer_nm),
            ],
        )

    # helper function
    def calculate_mean_field_mlm(
        self,
        energy_keV: float,
        applied_field_G: float,
        dead_layer_nm: float,
        demagnetization_factor: float,
    ):
        # product of the london model w/ the implantion distribution
        def integrand(z: float) -> float:
            return self.screening_profile(
                z, applied_field_G, dead_layer_nm, demagnetization_factor
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
        demagnetization_factor: float,
    ) -> np.array:
        # energy_keV = np.asarray(energy_keV)
        results = np.empty(energy_keV.size)
        for i, e_keV in enumerate(energy_keV):
            results[i] = self.calculate_mean_field_mlm(
                e_keV,
                applied_field_G,
                dead_layer_nm,
                demagnetization_factor,
            )
        return results
