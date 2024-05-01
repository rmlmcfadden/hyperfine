"""Facilities for analyzing Meissner screening profiles determined from LEM measurements.
"""

from typing import Annotated, Sequence, override
import numpy as np
import pandas as pd
from scipy import constants, integrate, interpolate, special
from .. import distributions
from ..superconductivity import london, pippard


class DepthAveragingCalculator:
    """Calculator for convolving a Meissner screening profile with a muon stopping distribution.

    Class for handling the details required for calculating the mean magnetic
    field at a given muon implantation energy by convolving a Meissner
    screening profile with the corresponding stopping distribution.

    This instance assumes local electrodynamics for the screening profile.
    """

    def __init__(
        self,
        file_name: str,
        interpolation: str = "linear",
    ) -> None:
        """Constructor.

        Args:
            file_name: Name of the CSV containing the stopping profile coefficients.
            interpolation: Type of interpolation scheme used for the stopping profile coefficients.
        """

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

    def stopping_distribution(
        self,
        depth_nm: Sequence[float],
        energy_keV: Annotated[float, 0:None],
    ) -> Sequence[float]:
        """Probability density function for the muon stopping distribution.

        The distribution is assumed to follow a weighted sum of two modified beta distributions.

        Args:
            depth_nm: Depth below the surface (nm).
            energy_keV: Muon implantation energy (keV).

        Returns:
            The probability density at depth_nm.
        """

        return distributions.modified_beta_2_pdf(
            depth_nm,
            self.alpha_1(energy_keV),
            self.beta_1(energy_keV),
            self.z_max_1(energy_keV),
            self.fraction_1(energy_keV),
            self.alpha_2(energy_keV),
            self.beta_2(energy_keV),
            self.z_max_2(energy_keV),
        )

    def calculate_mean_depth(
        self,
        energy_keV: Annotated[float, 0:None],
    ) -> float:
        """Calculate the mean muon stopping depth for a given implantation energy.

        The stopping distribution is assumed to follow a weighted sum of two
        modified beta distributions.

        Args:
            energy_keV: Muon implantation energy (keV).

        Returns:
            The mean stopping depth (nm).
        """

        return distributions.modified_beta_2_mean(
            self.alpha_1(energy_keV),
            self.beta_1(energy_keV),
            self.z_max_1(energy_keV),
            self.fraction_1(energy_keV),
            self.alpha_2(energy_keV),
            self.beta_2(energy_keV),
            self.z_max_2(energy_keV),
        )

    def _london(
        self,
        z: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        penetration_depth_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> Sequence[float]:
        """London model for the Meissner screening profile.

        Args:
            z: Depth below the surface (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Thickness of the non-superconducting dead layer (nm).
            penetration_depth_nm: Effective magnetic penetration depth (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The magnetic field value at depth z below the surface (G).
        """

        # determine the geometrically enhanced field value
        effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

        return np.piecewise(
            z,
            [
                z <= dead_layer_nm,
                z > dead_layer_nm,
            ],
            [
                lambda x: effective_field_G + x * 0.0,
                lambda x: effective_field_G
                * np.exp(-(x - dead_layer_nm) / penetration_depth_nm),
            ],
        )

    def london_ms(
        self,
        z: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        penetration_depth_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> Sequence[float]:
        """London model for the Meissner screening profile.

        Args:
            z: Depth below the surface (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Thickness of the non-superconducting dead layer (nm).
            penetration_depth_nm: Effective magnetic penetration depth (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The magnetic field value at depth z below the surface (G).
        """

        return self._london(
            z,
            applied_field_G,
            dead_layer_nm,
            penetration_depth_nm,
            demagnetization_factor,
        )

    def calculate_mean_field(
        self,
        energy_keV: float,
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        penetration_depth_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1],
    ) -> float:
        """Helper function for calculating the mean magnetic field below the surface.

        Args:
            energy_keV: Muon implantation energy (keV).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Thickness of the non-superconducting dead layer (nm).
            penetration_depth_nm: Effective magnetic penetration depth (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The mean magnetic field below the surface.
        """

        # product of the london model w/ the implantion distribution
        def integrand(z: float) -> float:
            return self.london_ms(
                z,
                applied_field_G,
                dead_layer_nm,
                penetration_depth_nm,
                demagnetization_factor,
            ) * self.stopping_distribution(z, energy_keV)

        # do the numeric integration using adaptive Gaussian quadrature
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
        result, _ = integrate.quad(
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
        return result

    def __call__(
        self,
        energy_keV: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        penetration_depth_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:None],
    ) -> Sequence[float]:
        """Functor for calculating the mean magnetic field below the surface.

        Can accept arrays of energies as input!

        Args:
            energy_keV: Muon implantation energy (keV).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Thickness of the non-superconducting dead layer (nm).
            penetration_depth_nm: Effective magnetic penetration depth (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The mean magnetic field below the surface.
        """

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


class DepthAveragingCalculatorNL(DepthAveragingCalculator):
    """Calculator for convolving a Meissner screening profile with a muon stopping distribution.

    Class for handling the details required for calculating the mean magnetic
    field at a given muon implantation energy by convolving a Meissner
    screening profile with the corresponding stopping distribution.

    This instance assumes nonlocal electrodynamics for the screening profile.
    """

    def __init__(
        self,
        file_name: str,
        interpolation: str = "linear",
    ) -> None:
        """Constructor.

        Args:
            file_name: Name of the CSV containing the stopping profile coefficients.
            interpolation: Type of interpolation scheme used for the stopping profile coefficients.
        """

        # initialize from the base class
        super().__init__(file_name, interpolation)

    def pippard_ms(
        self,
        depth_nm: float,
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        london_penetration_depth_nm: Annotated[float, 0:None],
        bcs_coherence_length_nm: Annotated[float, 0:None],
        mean_free_path_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1],
        temperature_K: Annotated[float, 0:None],
        critical_temperature_K: Annotated[float, 0:None],
        gap_0K_eV: Annotated[float, 0:None],
    ) -> float:
        """Pippard's nonlocal screening model.

        Assumes specular reflection of electrons at the surface.

        Args:
            depth_nm: Depth (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer (nm).
            london_penetration_depth_nm: London penetration depth (nm).
            bcs_coherence_length_nm: BCS coherence length (nm).
            mean_free_path_nm: Electron mean-free-path (nm).
            demagnetization_factor: Effective demagnetization factor.
            temperature_K: Absolute temperature (K).
            critical_temperature_K: Superconducting transition temperature (K).
            gap_0K_eV: Superconducting gap energy at 0 K (eV).

        Returns:
            The field screening profile at a given depth (G).
        """

        return (
            # nonlocal field screening model
            pippard.specular_profile_dl(
                depth_nm,
                temperature_K,
                critical_temperature_K,
                gap_0K_eV,
                london_penetration_depth_nm,
                mean_free_path_nm,
                bcs_coherence_length_nm,
                1.0,
                dead_layer_nm,
            )
            # effective applied magnetic field
            * applied_field_G
            / (1.0 - demagnetization_factor)
        )

    def mean_field_integrand(
        self,
        z: float,
        energy_keV: float,
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        london_penetration_depth_nm: Annotated[float, 0:None],
        bcs_coherence_length_nm: Annotated[float, 0:None],
        mean_free_path_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1],
        temperature_K: Annotated[float, 0:None],
        critical_temperature_K: Annotated[float, 0:None],
        gap_0K_eV: Annotated[float, 0:None],
    ) -> float:
        """Integrand for calculating the mean magnetic field at a given implantation energy.

        Args:
            z: Depth (nm).
            energy_keV: Implantation energy (keV).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer (nm).
            london_penetration_depth_nm: London penetration depth (nm).
            bcs_coherence_length_nm: BCS coherence length (nm).
            mean_free_path_nm: Electron mean-free-path (nm).
            demagnetization_factor: Effective demagnetization factor.
            temperature_K: Absolute temperature (K).
            critical_temperature_K: Superconducting transition temperature (K).
            gap_0K_eV: Superconducting gap energy at 0 K (eV).

        Returns:
            Integrand for the average magnetic field at a given implantation energy (G).
        """

        return self.pippard_ms(
            z,
            applied_field_G,
            dead_layer_nm,
            london_penetration_depth_nm,
            bcs_coherence_length_nm,
            mean_free_path_nm,
            demagnetization_factor,
            temperature_K,
            critical_temperature_K,
            gap_0K_eV,
        ) * self.stopping_distribution(z, energy_keV)

    @override
    def calculate_mean_field(
        self,
        energy_keV: float,
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        london_penetration_depth_nm: Annotated[float, 0:None],
        bcs_coherence_length_nm: Annotated[float, 0:None],
        mean_free_path_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1],
        temperature_K: Annotated[float, 0:None],
        critical_temperature_K: Annotated[float, 0:None],
        gap_0K_eV: Annotated[float, 0:None],
    ) -> float:
        """Calculate average magnetic field at a given implantation energy.

        Args:
            energy_keV: Implantation energy (keV).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer (nm).
            london_penetration_depth_nm: London penetration depth (nm).
            bcs_coherence_length_nm: BCS coherence length (nm).
            mean_free_path_nm: Electron mean-free-path (nm).
            demagnetization_factor: Effective demagnetization factor.
            temperature_K: Absolute temperature (K).
            critical_temperature_K: Superconducting transition temperature (K).
            gap_0K_eV: Superconducting gap energy at 0 K (eV).

        Returns:
            The average magnetic field at a given implantation energy (G).
        """

        # do the numeric integration using adaptive Gaussian quadrature
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
        result, _ = integrate.quad(
            self.mean_field_integrand,
            0.0,  # lower integration limit
            max(  # upper integration limit
                np.max(self.z_max_1(energy_keV)), np.max(self.z_max_2(energy_keV))
            ),
            args=(
                energy_keV,
                applied_field_G,
                dead_layer_nm,
                london_penetration_depth_nm,
                bcs_coherence_length_nm,
                mean_free_path_nm,
                demagnetization_factor,
                temperature_K,
                critical_temperature_K,
                gap_0K_eV,
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

        return result

    @override
    def __call__(
        self,
        energy_keV: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        london_penetration_depth_nm: Annotated[float, 0:None],
        bcs_coherence_length_nm: Annotated[float, 0:None],
        mean_free_path_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1],
        temperature_K: Annotated[float, 0:None],
        critical_temperature_K: Annotated[float, 0:None],
        gap_0K_eV: Annotated[float, 0:None],
    ) -> Sequence[float]:
        """Calculate average magnetic field at a given implantation energy.

        Functor version!

        Args:
            energy_keV: Implantation energy (keV).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer (nm).
            london_penetration_depth_nm: London penetration depth (nm).
            bcs_coherence_length_nm: BCS coherence length (nm).
            mean_free_path_nm: Electron mean-free-path (nm).
            demagnetization_factor: Effective demagnetization factor.
            temperature_K: Absolute temperature (K).
            critical_temperature_K: Superconducting transition temperature (K).
            gap_0K_eV: Superconducting gap energy at 0 K (eV).

        Returns:
            The average magnetic field at a given implantation energy.
        """

        # make everything numpy arrays
        energy_keV = np.asarray(energy_keV)
        results = np.empty(energy_keV.size)

        for i, e_keV in enumerate(energy_keV):
            results[i] = self.calculate_mean_field(
                e_keV,
                applied_field_G,
                dead_layer_nm,
                london_penetration_depth_nm,
                bcs_coherence_length_nm,
                mean_free_path_nm,
                demagnetization_factor,
                temperature_K,
                critical_temperature_K,
                gap_0K_eV,
            )
        return results


class DepthAveragingCalculatorGLE(DepthAveragingCalculator):
    """Calculator for convolving a Meissner screening profile with a muon stopping distribution.

    Class for handling the details required for calculating the mean magnetic
    field at a given muon implantation energy by convolving a Meissner
    screening profile with the corresponding stopping distribution.

    This instance assumes local electrodynamics following the generalized
    London equation (GLE).
    """

    def __init__(
        self,
        file_name: str,
        interpolation: str = "linear",
    ) -> None:
        """Constructor.

        Args:
            file_name: Name of the CSV containing the stopping profile coefficients.
            interpolation: Type of interpolation scheme used for the stopping profile coefficients.
        """

        # generalized London equation (GLE) solver
        self.gle = london.GLESolver()

        # initialize from the base class
        super().__init__(file_name, interpolation)

    def mean_field_integrand(
        self,
        z: float,
        energy_keV: float,
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        surface_penetration_depth_nm: Annotated[float, 0:None],
        bulk_penetration_depth_nm: Annotated[float, 0:None],
        diffusion_length_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> float:
        """Integrand for calculating the mean magnetic field at a given implantation energy.

        Args:
            z: Depth (nm).
            energy_keV: Implantation energy (keV).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer (nm).
            surface_penetration_depth_nm: Magnetic penetration depth at the surface (nm).
            bulk_penetration_depth_nm: Magnetic penetration depth in the bulk (nm).
            diffusion_length_nm: Length of inhomogenous defect region (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            Integrand for the average magnetic field at a given implantation energy (G).
        """

        return self.gle.screening_profile(
            z,
            applied_field_G,
            dead_layer_nm,
            surface_penetration_depth_nm,
            bulk_penetration_depth_nm,
            diffusion_length_nm,
            demagnetization_factor,
        ) * self.stopping_distribution(z, energy_keV)

    @override
    def calculate_mean_field(
        self,
        energy_keV: float,
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        surface_penetration_depth_nm: Annotated[float, 0:None],
        bulk_penetration_depth_nm: Annotated[float, 0:None],
        diffusion_length_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> float:
        """Calculate average magnetic field at a given implantation energy.

        Args:
            energy_keV: Implantation energy (keV).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer (nm).
            surface_penetration_depth_nm: Magnetic penetration depth at the surface (nm).
            bulk_penetration_depth_nm: Magnetic penetration depth in the bulk (nm).
            diffusion_length_nm: Length of inhomogenous defect region (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The average magnetic field at a given implantation energy (G).
        """

        # do the numeric integration using adaptive Gaussian quadrature
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
        result, _ = integrate.quad(
            self.mean_field_integrand,
            0.0,  # lower integration limit
            max(  # upper integration limit
                np.max(self.z_max_1(energy_keV)), np.max(self.z_max_2(energy_keV))
            ),
            args=(
                energy_keV,
                applied_field_G,
                dead_layer_nm,
                surface_penetration_depth_nm,
                bulk_penetration_depth_nm,
                diffusion_length_nm,
                demagnetization_factor,
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

        return result

    @override
    def __call__(
        self,
        energy_keV: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        surface_penetration_depth_nm: Annotated[float, 0:None],
        bulk_penetration_depth_nm: Annotated[float, 0:None],
        diffusion_length_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> float:
        """Calculate average magnetic field at a given implantation energy.

        Functor version!

        Args:
            energy_keV: Implantation energy (keV).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer (nm).
            surface_penetration_depth_nm: Magnetic penetration depth at the surface (nm).
            bulk_penetration_depth_nm: Magnetic penetration depth in the bulk (nm).
            diffusion_length_nm: Length of inhomogenous defect region (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The average magnetic field at a given implantation energy (G).
        """

        # make everything numpy arrays
        energy_keV = np.asarray(energy_keV)
        results = np.empty(energy_keV.size)

        for i, e_keV in enumerate(energy_keV):
            results[i] = self.calculate_mean_field(
                e_keV,
                applied_field_G,
                dead_layer_nm,
                surface_penetration_depth_nm,
                bulk_penetration_depth_nm,
                diffusion_length_nm,
                demagnetization_factor,
            )
        return results
