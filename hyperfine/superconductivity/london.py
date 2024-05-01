"""Expressions for the London model of magnetic field screening.
"""

import numpy as np
from scipy import constants, integrate, special
from typing import Annotated, Sequence


def screening_profile_bulk(
    depth_nm: Sequence[float],
    applied_field_G: Annotated[float, 0:None],
    dead_layer_nm: Annotated[float, 0:None],
    penetration_depth_nm: Annotated[float, 0:None],
    demagnetization_factor: Annotated[float, 0:1],
) -> Sequence[float]:
    """Meissner screening profile for a bulk superconductor.

    Meissner screening profile derived from London theory for a bulk
    superconductor, wherein the material's electromagnetic response is treated
    in the local limit.

    Args:
        depth_nm: Depth below the surface (nm).
        applied_field_G: Applied magnetic field (G).
        dead_layer_nm: Thickness of the non-superconducting surface dead layer (nm).
        penetration_depth_nm: Magnetic penetration depth (nm).
        demagnetization_factor: Effective demagnetization factor.

    Returns:
        The Meissner screening profile at depth ``depth_nm`` (G).

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import london

           z = np.linspace(0.0, 200.0, 100)
           args = (100.0, 10.0, 50.0, 0.05)
           plt.plot(z, london.screening_profile_bulk(z, *args), "-")
           plt.xlabel("$z$ (nm)")
           plt.ylabel("$B(z)$ (nm)")
           plt.show()

    """

    # effective field at the sample's surface
    effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

    # correct for the dead layer thickness
    z_nm = depth_nm - dead_layer_nm

    # evaluate the expression as piecewise components
    return np.piecewise(
        z_nm,
        [
            z_nm <= 0.0,
            z_nm > 0.0,
        ],
        [
            lambda x: effective_field_G * np.ones(len(x)),
            lambda x: effective_field_G * np.exp(-x / penetration_depth_nm),
        ],
    )


def screening_profile_film(
    depth_nm: Sequence[float],
    applied_field_G: Annotated[float, 0:None],
    dead_layer_nm: Annotated[float, 0:None],
    penetration_depth_nm: Annotated[float, 0:None],
    film_thickness_nm: Annotated[float, 0:None],
    demagnetization_factor: Annotated[float, 0:1],
) -> Sequence[float]:
    """Meissner screening profile for a thin film superconductor.

    Meissner screening profile derived from London theory for a thin film
    superconductor, wherein the material's electromagnetic response is treated
    in the local limit.

    Args:
        depth_nm: Depth below the surface (nm).
        applied_field_G: Applied magnetic field (G).
        dead_layer_nm: Thickness of the non-superconducting surface dead layer (nm).
        film_thickness_nm: Film thickness (nm).
        penetration_depth_nm: Magnetic penetration depth (nm).
        demagnetization_factor: Effective demagnetization factor.

    Returns:
        The Meissner screening profile at depth ``depth_nm`` (G).

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.superconductivity import london

           z = np.linspace(0.0, 200.0, 100)
           args = (100.0, 10.0, 50.0, 190.0, 0.0)
           plt.plot(z, london.screening_profile_film(z, *args), "-")
           plt.xlabel("$z$ (nm)")
           plt.ylabel("$B(z)$ (nm)")
           plt.show()

    """

    # effective field at the sample's surface
    effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

    # correct for the dead layer thickness
    z_nm = depth_nm - dead_layer_nm
    t_nm = film_thickness_nm - dead_layer_nm

    # evaluate the expression as piecewise components
    return np.piecewise(
        z_nm,
        [
            z_nm <= 0.0,
            (z_nm > 0.0) & (z_nm <= t_nm),
            z_nm > t_nm,
        ],
        [
            lambda x: effective_field_G * np.ones(len(x)),
            lambda x: effective_field_G
            * np.cosh((0.5 * t_nm - x) / penetration_depth_nm)
            / np.cosh(0.5 * t_nm / penetration_depth_nm),
            lambda x: effective_field_G * np.ones(len(x)),
        ],
    )


class GLESolver:
    """Generalized London Equation (GLE) Solver.

    Numerically solve the GLE for a depth-dependent magnetic penetration depth.

    See:
    M. Checchin et al., Appl. Phys. Lett. 117, 032601 (2020).
    https://doi.org/10.1063/5.0013698

    See also Eq. (1) in:
    M. S. Pamianchi at al., Phys. Rev. B 50, 13659 (1994).
    https://doi.org/10.1103/PhysRevB.50.13659

    Attributes:
        _x_nodes: x-values used as the initial mesh by the solver.
        _y_guess: y-values used as the guess for the function/derivative by the solver.
        _lambda_s: Magnetic penetration depth at the surface (nm).
        _lambda_0: Magnetic penetration depth in the bulk (nm).
        _delta: Diffusion length of the impurity layer (nm).
        _sol: Object encapsulating the solver's solution.
    """

    def __init__(
        self,
        x_nodes_min: float = 0.0,
        x_nodes_max: float = 1000.0,
        x_nodes_num: int = 1001,
    ) -> None:
        """Constructor for the GLE Solver.

        Args:
            x_nodes_min: Minimum of the x-values used as the initial mesh by the solver.
            x_nodes_max: Maximum of the x-values used as the initial mesh by the solver.
            x_nodes_num: Number of x-values used as the initial mesh by the solver.
        """

        # create the x-nodes
        self._x_nodes = np.linspace(x_nodes_min, x_nodes_max, num=x_nodes_num)

        # assign some internal variables with nonsensical values; this is to
        # diminish any likelihood of corresponding to a user's "real" input,
        # which is important for the arg check done in self.solve
        self._lambda_s = -1.0
        self._lambda_0 = -1.0
        self._delta = -1.0

        # create an empty initial guess for the solver
        # values will be updated dynamically when calling self.solve
        self._y_guess = np.zeros((2, self._x_nodes.size))

    def _lambda(
        self,
        x: Sequence[float],
        lambda_s: Annotated[float, 0:None],
        lambda_0: Annotated[float, 0:None],
        delta: Annotated[float, 0:None],
    ) -> Sequence[float]:
        """Postulated depth-dependence of the magnetic penetration depth.

        See Eq. (1) in:
        M. Checchin et al., Appl. Phys. Lett. 117, 032601 (2020).
        https://doi.org/10.1063/5.0013698

        Args:
            x: Depth below the surface (nm).
            lambda_s: Magnetic penetration depth at the surface (nm).
            lambda_0: Magnetic penetration depth in the bulk (nm).
            delta: Diffusion length of impurities causing depth-dependence (nm).

        Returns:
            The magnetic penetration depth at depth x (nm).

        """

        return (lambda_s - lambda_0) * special.erfc(x / delta) + lambda_0

    def _lambda_prime(
        self,
        x: Sequence[float],
        lambda_s: Annotated[float, 0:None],
        lambda_0: Annotated[float, 0:None],
        delta: Annotated[float, 0:None],
    ) -> Sequence[float]:
        """First derivative of the postulated depth-dependence of the magnetic penetration depth.

        See Eq. (1) in:
        M. Checchin et al., Appl. Phys. Lett. 117, 032601 (2020).
        https://doi.org/10.1063/5.0013698

        Args:
            x: Depth below the surface (nm).
            lambda_s: Magnetic penetration depth at the surface (nm).
            lambda_0: Magnetic penetration depth in the bulk (nm).
            delta: Diffusion length of impurities causing depth-dependence (nm).

        Returns:
            The first derivative of the magnetic penetration depth at depth x.

        """

        return (lambda_s - lambda_0) * (
            -2.0 * np.exp(-((x / delta) ** 2)) / (np.sqrt(np.pi) * delta)
        )

    def _gle_derivs(
        self,
        t: Sequence[float],
        y: Sequence[float],
    ) -> Sequence[float]:
        """Right-hand side of the system of equations to solve, re-written as 1st-order expressions.

        Indexes [0] refer to the function being solved for.
        Indexes [1] refer to the function's derivative.

        Args:
            t: x-values.
            y: y-values

        Returns:
            An array of the system of 1st-order equations to solve.

        """

        # variable substitution
        b, nu = y

        # evaluate the penetration depth and its derivative
        args = (self._lambda_s, self._lambda_0, self._delta)
        l = self._lambda(t, *args)
        l_prime = self._lambda_prime(t, *args)

        return np.vstack(
            [
                nu,
                -(2.0 / l) * l_prime * nu + (1.0 / l**2) * b,
            ]
        )

    def _bc(
        self,
        ya: Sequence[float],
        yb: Sequence[float],
    ) -> Sequence[float]:
        """Boundary conditions for the solver.

        Indexes [0] refer to the function being solved for.
        Indexes [1] refer to the function's derivative.

        Args:
            ya: Array of lower bounds.
            yb: Array of upper bounds.

        Returns:
            An array of penalties for the boundary conditions.

        """

        return np.array(
            [
                ya[0] - 1,  # B(x) / B_0 = 1
                yb[1],  # B'(np.inf) / B_0 = 0
            ]
        )

    def solve(
        self,
        lambda_s: Annotated[float, 0:None],
        lambda_0: Annotated[float, 0:None],
        delta: Annotated[float, 0:None],
        tolerance: float = np.sqrt(np.finfo(float).eps),  # 1e3 * np.finfo(float).eps,
        max_x_nodes: int = np.iinfo(np.int32).max,
    ) -> None:
        """Solve the GLE numerically.

        Args:
            lambda_s: Magnetic penetration depth at the surface (nm).
            lambda_0: Magnetic penetration depth in the bulk (nm).
            delta: Diffusion length of the impurity layer (nm).
            tolerance: Convergence criteria for the solver.
            max_x_nodes: Maximum number of x nodes used by the solver.
        """

        # collect the function/class arguments for comparison
        fcn_args = (lambda_s, lambda_0, delta)
        cls_vals = (self._lambda_s, self._lambda_0, self._delta)

        # check if a solution has already been generated for the same inputs
        if fcn_args != cls_vals:

            # assign the arguments to data members
            self._lambda_s = lambda_s
            self._lambda_0 = lambda_0
            self._delta = delta

            # dynamically create the initial guesses for the solver
            self._y_guess = np.zeros((2, self._x_nodes.size))
            self._y_guess[0] = np.exp(-self._x_nodes / self._lambda_0)
            self._y_guess[1] = (-1.0 / self._lambda_0) * np.exp(
                -self._x_nodes / self._lambda_0
            )

            # solve the system of equations using boundary conditions and save the result
            self._sol = integrate.solve_bvp(
                self._gle_derivs,
                self._bc,
                self._x_nodes,
                self._y_guess,
                p=None,
                S=None,
                fun_jac=None,
                bc_jac=None,
                tol=tolerance,
                max_nodes=max_x_nodes,
                verbose=0,
                # bc_tol=np.sqrt(np.finfo(float).eps),
            )

    def screening_profile(
        self,
        z_nm: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        penetration_depth_surface_nm: Annotated[float, 0:None],
        penetration_depth_bulk_nm: Annotated[float, 0:None],
        diffusion_length_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> Sequence[float]:
        """Calculate the Meissner screening profile.

        Args:
            z_nm: Depth below the surface (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer thickness (nm).
            penetration_depth_surface_nm: Magnetic penetration depth at the surface (nm).
            penetration_depth_bulk_nm: Magnetic penetration depth in the bulk (nm).
            diffusion_length_nm: Diffusion length of the impurity layer (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The Meissner screening profile at depth z (G).

        Example:
            .. plot::

               import numpy as np
               import matplotlib.pyplot as plt
               from hyperfine.superconductivity import london

               gles = london.GLESolver()
               z = np.linspace(0.0, 200.0, 100)
               args = (100.0, 10.0, 100.0, 30.0, 50.0, 0.05)
               plt.plot(z, gles.screening_profile(z, *args), "-")
               plt.xlabel("$z$ (nm)")
               plt.ylabel("$B(z)$ (nm)")
               plt.show()

        """

        # solve the GLE
        self.solve(
            penetration_depth_surface_nm,
            penetration_depth_bulk_nm,
            diffusion_length_nm,
        )

        # calculate the geometrically enhanced field value
        effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

        # correct the depth for the dead layer
        z_corr_nm = z_nm - dead_layer_nm

        # return the current density
        return np.piecewise(
            z_corr_nm,
            [
                z_corr_nm < 0.0,
                z_corr_nm >= 0.0,
            ],
            [
                lambda x: np.full(x.shape, effective_field_G),
                lambda x: effective_field_G * self._sol.sol(x)[0],
            ],
        )

    def current_density(
        self,
        z_nm: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        penetration_depth_surface_nm: Annotated[float, 0:None],
        penetration_depth_bulk_nm: Annotated[float, 0:None],
        diffusion_length_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> Sequence[float]:
        """Calculate the current density profile.

        Args:
            z_nm: Depth below the surface (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer thickness (nm).
            penetration_depth_surface_nm: Magnetic penetration depth at the surface (nm).
            penetration_depth_bulk_nm: Magnetic penetration depth in the bulk (nm).
            diffusion_length_nm: Diffusion length of the impurity layer (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The current density profile at depth z (A m^-2).

        Example:
            .. plot::

               import numpy as np
               import matplotlib.pyplot as plt
               from hyperfine.superconductivity import london

               gles = london.GLESolver()
               z = np.linspace(0.0, 200.0, 100)
               args = (100.0, 10.0, 100.0, 30.0, 50.0, 0.05)
               plt.plot(z, gles.current_density(z, *args), "-")
               plt.xlabel("$z$ (nm)")
               plt.ylabel("$J(z)$ (A m$^{-2}$)")
               plt.show()

        """

        # solve the GLE
        self.solve(
            penetration_depth_surface_nm,
            penetration_depth_bulk_nm,
            diffusion_length_nm,
        )

        # calculate the geometrically enhanced field value
        effective_field_G = applied_field_G / (1.0 - demagnetization_factor)

        # calculate the prefactor for the conversion
        G_per_T = 1e4
        # nm_per_m = 1e9
        m_per_nm = 1e-9
        mu_0 = constants.value("vacuum mag. permeability") * G_per_T

        j_0 = -1.0 * effective_field_G / mu_0

        # correct the depth for the dead layer
        z_corr_nm = z_nm - dead_layer_nm

        # return the screening profile
        return np.piecewise(
            z_corr_nm,
            [
                z_corr_nm < 0.0,
                z_corr_nm >= 0.0,
            ],
            [
                lambda x: np.full(x.shape, 0.0 * j_0),
                lambda x: j_0 * self._sol.sol(x)[1] / m_per_nm,
            ],
        )

    def __call__(
        self,
        z_nm: Sequence[float],
        applied_field_G: Annotated[float, 0:None],
        dead_layer_nm: Annotated[float, 0:None],
        penetration_depth_surface_nm: Annotated[float, 0:None],
        penetration_depth_bulk_nm: Annotated[float, 0:None],
        diffusion_length_nm: Annotated[float, 0:None],
        demagnetization_factor: Annotated[float, 0:1] = 0.0,
    ) -> Sequence[float]:
        """Calculate the Meissner screening profile (alias for self.screening_profile).

        Args:
            z_nm: Depth below the surface (nm).
            applied_field_G: Applied magnetic field (G).
            dead_layer_nm: Non-superconducting dead layer thickness (nm).
            penetration_depth_surface_nm: Magnetic penetration depth at the surface (nm).
            penetration_depth_bulk_nm: Magnetic penetration depth in the bulk (nm).
            diffusion_length_nm: Diffusion length of the impurity layer (nm).
            demagnetization_factor: Effective demagnetization factor.

        Returns:
            The Meissner screening profile at depth z (G).

        Example:
            .. plot::

               import numpy as np
               import matplotlib.pyplot as plt
               from hyperfine.superconductivity import london

               gles = london.GLESolver()
               z = np.linspace(0.0, 200.0, 100)
               args = (100.0, 10.0, 100.0, 30.0, 50.0, 0.05)
               plt.plot(z, gles(z, *args), "-")
               plt.xlabel("$z$ (nm)")
               plt.ylabel("$B(z)$ (nm)")
               plt.show()

        """

        return self.screening_profile(
            z_nm,
            applied_field_G,
            dead_layer_nm,
            penetration_depth_surface_nm,
            penetration_depth_bulk_nm,
            diffusion_length_nm,
            demagnetization_factor,
        )
