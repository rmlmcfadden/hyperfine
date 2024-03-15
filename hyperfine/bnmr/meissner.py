import numpy as np
import pandas as pd
from scipy import constants, integrate, interpolate, special
from .. import distributions
from .susceptibility import tl_B_c1_T, magnetic_volume_susceptibility
from .nlme import scaled_penetration_depth_nm
from typing import Annotated


class DepthAveragingCalculator:
    """
    class for handling the details required for calculating the mean field
    at a given energy by integrating over the implantation profile
    """

    # constructor
    def __init__(self, file_name: str, interpolation: str = "linear"):
        # read the implantation distribution parameters
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

    # implantation distribution function
    def modified_beta_distribution(
        self, z: float, alpha: float, beta: float, float, z_max: float
    ) -> float:
        return distributions.modified_beta(z, alpha, beta, z_man)

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
        return distributions.modified_beta_2_pdf(
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

    def lorentzian(self, B: float, B_d: float, tau_c: float) -> float:
        # gyromagnetic ratios
        # https://www-nds.iaea.org/publications/indc/indc-nds-0794/
        gamma_8Li = 2.0 * np.pi * 6.30221e6  # Hz / T
        gamma_93Nb = 2.0 * np.pi * 10.439565e6  # Hz / T
        # Larmor frequencies
        omega_I = gamma_8Li * B  # probe nucleus
        omega_S = gamma_93Nb * B  # host nucleus
        omega_d_2 = (
            np.abs(gamma_8Li * gamma_93Nb) * B_d * B_d
        )  # probe-host dipole-dipole coupling
        # generic spectral density function
        j = lambda omega, nu: nu / (nu * nu + omega * omega)
        nu_c = 1.0 / tau_c  # NMR correlation rate
        # heteronuclear dipole-dipole SLR rate
        return omega_d_2 * (
            (1.0 / 3.0) * j(omega_I - omega_S, nu_c)
            + (1.0 / 1.0) * j(omega_I, nu_c)
            + (2.0 / 1.0) * j(omega_I + omega_S, nu_c)
        )

    def lambda_two_fluid(
        self,
        temperature: float,
        critical_temperature: float,
        penetration_depth_0K: float,
    ) -> float:
        return np.piecewise(
            temperature,
            [
                temperature < critical_temperature,
                temperature >= critical_temperature,
            ],
            [
                lambda x: penetration_depth_0K
                / np.sqrt(1.0 - np.power(temperature / critical_temperature, 4)),
                lambda x: np.inf,
            ],
        )

    def critical_temperature(
        self,
        applied_field: float,
        critical_field: float,
        critical_temperature_0T: float,
    ) -> float:
        """
        Inverted version of Tuyn's law (for Bc1 or Bc)
        """
        return np.piecewise(
            applied_field,
            [
                applied_field < critical_field,
                applied_field >= critical_field,
            ],
            [
                lambda x: critical_temperature_0T * np.sqrt(1.0 - (x / critical_field)),
                lambda x: x * 0,
            ],
        )

    def critical_temperature2(
        self,
        applied_field: float,
        critical_field: float,
        critical_temperature_0T: float,
    ) -> float:
        """
        Inverted version of Tuyn's law (for Bc2)
        """
        return np.piecewise(
            applied_field,
            [
                applied_field < critical_field,
                applied_field >= critical_field,
            ],
            [
                lambda x: critical_temperature_0T
                * np.sqrt((1.0 - (x / critical_field)) / (1.0 + (x / critical_field))),
                lambda x: x * 0,
            ],
        )

    def london_model(
        self,
        z,
        B_applied: float,
        dead_layer: float,
        penetration_depth: float,
    ) -> float:
        return np.piecewise(
            z,
            [
                z <= dead_layer,
                z > dead_layer,
            ],
            [
                lambda x: B_applied + 0.0 * x,
                lambda x: B_applied * np.exp(-(x - dead_layer) / penetration_depth),
            ],
        )

    def london_model_film(
        self,
        z,
        B_applied: float,
        dead_layer: float,
        penetration_depth: float,
        film_thickness: float,
    ) -> float:
        effective_thickness = film_thickness - dead_layer
        return np.piecewise(
            z,
            [
                z <= dead_layer,
                (z > dead_layer) & (z <= effective_thickness),
                z > film_thickness,
            ],
            [
                lambda x: B_applied + 0.0 * x,
                lambda x: B_applied
                * np.cosh((x - 0.5 * effective_thickness) / penetration_depth)
                / np.cosh(0.5 * effective_thickness / penetration_depth),
                lambda x: B_applied + 0.0 * x,
            ],
        )

    def field_enhancement_factor(
        self,
        applied_field_T: float,
        temperature_K: float,
        critical_temperature_K: float,
        lower_critical_field_T: float,
        upper_critical_field_T: float,
        demagnetization_factor: float,
    ) -> float:
        return 1.0 / (
            1.0
            + magnetic_volume_susceptibility(
                applied_field_T,
                temperature_K,
                critical_temperature_K,
                lower_critical_field_T,
                upper_critical_field_T,
                0.0,  # normal state susceptibility
            )
            * demagnetization_factor
        )

    def B_london_vortex_continuum_T(
        self,
        depth_nm: float,
        penetration_depth_nm: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        applied_field_T: Annotated[float, 0:None],
        average_field_T: Annotated[float, 0:None],
    ) -> float:
        """
        See Eq. (3) and below in: Brandt PRL 67, (1991).
        """

        return np.piecewise(
            depth_nm,
            [
                depth_nm <= dead_layer_nm,
                depth_nm > dead_layer_nm,
            ],
            [
                lambda x: applied_field_T + 0.0 * x,
                lambda x: average_field_T
                + (applied_field_T - average_field_T)
                * np.exp(-(x - dead_layer_nm) / penetration_depth_nm),
            ],
        )

    # helper function
    def calculate_mean_slr_rate_E(
        self,
        energy_keV: float,
        applied_field_T: float,
        dead_layer_nm: float,
        penetration_depth_nm: float,
        dipolar_field_T: float,
        correlation_time_s: float,
    ) -> float:

        # product of the London model w/ the SLR rate "ingredients"
        def integrand(z):
            B = self.london_model(
                z, applied_field_T, dead_layer_nm, penetration_depth_nm
            )
            lor = self.lorentzian(B, dipolar_field_T, correlation_time_s)
            rho = self.stopping_distribution_e(z, energy_keV)
            return lor * rho

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
                self.z_max_1(energy_keV),
                self.z_max_2(energy_keV),
                dead_layer_nm,
            ],
        )
        return result[0]

    # helper function
    def calculate_mean_slr_rate_E_T(
        self,
        energy_keV: float,
        applied_field_T: float,
        dead_layer_nm: float,
        penetration_depth_nm: float,
        dipolar_field_T: float,
        correlation_time_s: float,
        temperature_K: float,
        critical_temperature_K: float,
        lower_critical_field_T: float,
        upper_critical_field_T: float,
        slope_s_K: float,
        curie_constant_K_s: float,
        surface_constant_s: float,
        demagnetization_factor: float,
        slope_scaling_constant: float,
        slope_scaling_exponent: float,
    ) -> float:

        # correct applied field for demagnetization
        effective_field_T = (
            self.field_enhancement_factor(
                applied_field_T,
                temperature_K,
                critical_temperature_K,
                lower_critical_field_T,
                upper_critical_field_T,
                demagnetization_factor,
            )
            * applied_field_T
        )

        # pre-compute some values
        effective_T_c = self.critical_temperature2(
            effective_field_T, upper_critical_field_T, critical_temperature_K
        )

        oxide_layer_thickness_nm = 5.0

        # product of the London model w/ the SRL rate "ingredients"
        def integrand(z: float) -> float:

            # calculate the effective magnetic penetration depth
            # (i.e., at finite temperature)
            effective_lambda = self.lambda_two_fluid(
                temperature_K, effective_T_c, penetration_depth_nm
            )

            # calculate the scaled magnetic penetration depth
            # (i.e., at finite magnetic field)
            scaled_lambda = scaled_penetration_depth_nm(
                effective_field_T,
                effective_lambda,
                29.0,  # Nb's London penetration depth (nm)
                40.0,  # Nb's BCS coherence length (nm)
            )

            # calculate the lower critical field at finite temperature
            lower_critical_field_finite_temp_T = tl_B_c1_T(
                temperature_K,
                critical_temperature_K,
                lower_critical_field_T,
            )

            # select the correct local field model based on the field regime
            if effective_field_T <= lower_critical_field_finite_temp_T:
                # simple London model when in the Meissner state
                B = self.london_model(
                    z,
                    effective_field_T,
                    dead_layer_nm,
                    scaled_lambda,
                )
            else:
                # continuum London model when in the vortex state
                B = self.B_london_vortex_continuum_T(
                    z,
                    scaled_lambda,
                    dead_layer_nm,
                    effective_field_T,
                    applied_field_T,
                )

            # Lorentzian-like contribution from dipole-dipole relaxation
            lor = self.lorentzian(B, dipolar_field_T, correlation_time_s)

            # empirical scaling of the slope with applied field
            # + 1.271e-2 = Korringa slope @ high field
            scaled_slope_s_K = slope_s_K * (
                1.0
                + np.power(
                    effective_field_T / slope_scaling_constant, -slope_scaling_exponent
                )
            )

            # linear contribution to the slr rate
            linear = scaled_slope_s_K * temperature_K

            # curie = (
            #     surface_constant_s + curie_constant_K_s / temperature_K
            #     if z <= 5.0
            #     else 0.0
            # )
            # curie = (curie_constant_K_s / temperature_K) if z <= oxide_layer_thickness_nm else (curie_constant_K_s / temperature_K) * np.exp(-(z-oxide_layer_thickness_nm))
            curie = (
                (curie_constant_K_s / np.power(temperature_K, surface_constant_s))
                # * np.power(effective_field_T, -2.0)
                if z <= oxide_layer_thickness_nm
                else 0.0
            )
            # curie = curie_constant_K_s if z <= oxide_layer_thickness_nm else 0.0
            # curie = (curie_constant_K_s / temperature_K) if z <= oxide_layer_thickness_nm else (curie_constant_K_s / temperature_K) * np.power(1.0 + z / surface_constant_s, -2.0)
            """            
            curie = (
                surface_constant_s * np.exp(curie_constant_K_s / temperature_K)
                if z <= oxide_layer_thickness_nm
                else 0.0
            )
            """
            # handle potential nan
            if curie == np.nan:
                curie = np.finfo(float).max

            rate = lor + linear + curie
            rho = self.stopping_distribution_e(z, energy_keV)

            return rate * rho

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
                0.0,  #
                oxide_layer_thickness_nm,  # 5 nm Nb2O5 surface oxide layer
                self.z_max_1(energy_keV),
                self.z_max_2(energy_keV),
                dead_layer_nm,
            ],
        )
        return result[0]

    # functor version of calculate_mean_slr_rate (can take an array of energies)!
    def __call__(
        self,
        energy_keV: list,
        applied_field_T: float,
        dead_layer_nm: float,
        penetration_depth_nm: float,
        dipolar_field_T: float,
        correlation_time_s: float,
        temperature_K: float,
        critical_temperature_K: float,
        lower_critical_field_T: float,
        upper_critical_field_T: float,
        slope_s_K: float,
        curie_constant_K_s: float,
        surface_constant_s: float,
        demagnetization_factor: float,
        slope_scaling_constant: float,
        slope_scaling_exponent: float,
    ):
        energy_keV = np.asarray(energy_keV)
        if energy_keV.size == 1:
            return self.calculate_mean_slr_rate_E_T(
                energy_keV,
                applied_field_T,
                dead_layer_nm,
                penetration_depth_nm,
                dipolar_field_T,
                correlation_time_s,
                temperature_K,
                critical_temperature_K,
                lower_critical_field_T,
                upper_critical_field_T,
                slope_s_K,
                curie_constant_K_s,
                surface_constant_s,
                demagnetization_factor,
                slope_scaling_constant,
                slope_scaling_exponent,
            )
        results = np.empty(energy_keV.size)
        for i, e_keV in enumerate(energy_keV):
            results[i] = self.calculate_mean_slr_rate_E_T(
                e_keV,
                applied_field_T,
                dead_layer_nm,
                penetration_depth_nm,
                dipolar_field_T,
                correlation_time_s,
                temperature_K,
                critical_temperature_K,
                lower_critical_field_T,
                upper_critical_field_T,
                slope_s_K,
                curie_constant_K_s,
                surface_constant_s,
                demagnetization_factor,
                slope_scaling_constant,
                slope_scaling_exponent,
            )
        return results
