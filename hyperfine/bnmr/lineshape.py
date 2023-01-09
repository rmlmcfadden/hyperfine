from dataclasses import dataclass, field
import numpy as np
from scipy.constants import physical_constants


@dataclass
class QuadrupoleSplitting:
    """
    Genernal class for calculating (exact) quadrupole splittings of an NMR line.
    """

    I: float  # nuclear spin quantum number
    gamma: float  # gyromagnetic ratio (Hz/T)
    Q: float  # nuclear electric quadrupole moment (mb)
    N: int = field(init=False)  # 2 * I + 1
    I_x: np.array = field(init=False, repr=False)
    I_y: np.array = field(init=False, repr=False)
    I_z: np.array = field(init=False, repr=False)
    I_p: np.array = field(init=False, repr=False)
    I_m: np.array = field(init=False, repr=False)
    I_2: np.array = field(init=False, repr=False)

    def __post_init__(self):
        # make sure the nuclear spin is compatible with a non-zero quadrupole moment
        assert self.I >= 1.0
        # make sure the nuclear spin is a half-integer value
        assert self.I % 0.5 == 0.0

        # convenience quantities
        self.N = int(2 * self.I + 1)
        shape = (self.N, self.N)

        # dynamically create the spin matrices
        self.I_z = np.zeros(shape)
        self.I_p = np.zeros(shape)
        self.I_m = np.zeros(shape)

        # fill the I_z, I_p, and I_m matrix elements dynamically
        for i in range(self.N):
            for j in range(self.N):
                # diagonal elements
                if i == j:
                    self.I_z[i][j] = self.I - float(j)
                # lower off-diagonal elements
                if i == j + 1:
                    self.I_m[i][j] = np.sqrt(
                        self.I * (self.I + 1) - (j - self.I) * ((j - self.I) + 1)
                    )
                # upper off-diagonal elements
                if i == j - 1:
                    self.I_p[i][j] = np.sqrt(
                        self.I * (self.I + 1) - (j - self.I) * ((j - self.I) - 1)
                    )

        # create the others
        self.I_x = 0.5 * (self.I_p + self.I_m)
        self.I_y = -0.5j * (self.I_p - self.I_m)
        self.I_2 = self.I * (self.I + 1) * np.identity(self.N)

        # check the initialization using known commutation relationships
        absolute_tolerance = np.sqrt(np.finfo(float).eps)
        relative_tolerance = np.sqrt(np.finfo(float).eps)
        assert np.allclose(
            self.I_x @ self.I_y - self.I_y @ self.I_x,
            1j * self.I_z,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        assert np.allclose(
            self.I_y @ self.I_z - self.I_z @ self.I_y,
            1j * self.I_x,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        assert np.allclose(
            self.I_z @ self.I_x - self.I_x @ self.I_z,
            1j * self.I_y,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        assert np.allclose(
            self.I_z @ self.I_p - self.I_p @ self.I_z,
            self.I_p,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        assert np.allclose(
            self.I_z @ self.I_m - self.I_m @ self.I_z,
            -1 * self.I_m,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )
        assert np.allclose(
            self.I_p @ self.I_m - self.I_m @ self.I_p,
            2 * self.I_z,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
        )

    def lorentzian(
        self, x: float, position: float, fwhm: float, amplitude: float
    ) -> float:
        """
        Lorentzian lineshape.
        """
        gamma = 0.5 * fwhm
        arg = (x - position) / gamma
        return amplitude / (1.0 + arg * arg)

    def quadrupole_coupling(self, efg: float) -> float:
        """
        Quadrupole coupling constant (in Hz).
        """
        # physical constants
        e, _, _ = physical_constants["elementary charge"]
        h, _, _ = physical_constants["Planck constant"]
        # conversion factors
        m2_per_mb = 1e-31
        V_per_m_per_V_per_A = 1e20
        # value in Hz
        return e * (self.Q * m2_per_mb) * (efg * 1e20) / h

    def quadrupole_frequency(self, efg: float) -> float:
        """
        Quadrupole frequency (in Hz).
        """
        # quadrupole coupling
        C_Q = self.quadrupole_coupling(efg)
        # value in Hz
        return C_Q / (4.0 * self.I * (2.0 * self.I - 1.0))

    def V_xx(self, eta: float, theta: float, phi: float) -> float:
        return 0.5 * (
            3.0 * np.sin(theta) * np.sin(theta)
            - 1.0
            + eta * np.cos(theta) * np.cos(theta) * np.cos(2.0 * phi)
        )

    def V_yy(self, eta: float, theta: float, phi: float) -> float:
        return -0.5 * (1.0 + eta * np.cos(2.0 * phi))

    def V_zz(self, eta: float, theta: float, phi: float) -> float:
        return 0.5 * (
            3.0 * np.cos(theta) * np.cos(theta)
            - 1.0
            + eta * np.sin(theta) * np.sin(theta) * np.cos(2.0 * phi)
        )

    def V_xy(self, eta: float, theta: float, phi: float) -> float:
        return -0.5 * eta * np.cos(theta) * np.sin(2.0 * phi)

    def V_yx(self, eta: float, theta: float, phi: float) -> float:
        return self.V_xy(eta, theta, phi)

    def V_xz(self, eta: float, theta: float, phi: float) -> float:
        return -0.5 * np.sin(theta) * np.cos(theta) * (3.0 - eta * np.cos(2.0 * phi))

    def V_zx(self, eta: float, theta: float, phi: float) -> float:
        return self.V_xz(eta, theta, phi)

    def V_yz(self, eta: float, theta: float, phi: float) -> float:
        return -0.5 * eta * np.sin(theta) * np.sin(2.0 * phi)

    def V_zy(self, eta: float, theta: float, phi: float) -> float:
        return self.V_yz(eta, theta, phi)

    def V(self, eta: float, theta: float, phi: float) -> np.array:
        """
        Electric field gradient (EFG) tensor.
        """
        return np.array(
            [
                [
                    self.V_xx(eta, theta, phi),
                    self.V_xy(eta, theta, phi),
                    self.V_xz(eta, theta, phi),
                ],
                [
                    self.V_yx(eta, theta, phi),
                    self.V_yy(eta, theta, phi),
                    self.V_yz(eta, theta, phi),
                ],
                [
                    self.V_zx(eta, theta, phi),
                    self.V_zy(eta, theta, phi),
                    self.V_zz(eta, theta, phi),
                ],
            ]
        )

    def V_pas(self, eta: float, theta: float, phi: float) -> np.array:
        """
        Electric field gradient (EFG) tensor in its principle axis system (PAS).
        """
        V = self.V(eta, theta, phi)
        e_vals, e_vecs = np.linalg.eigh(V, UPLO="L")
        # sort the eigenvalues/vectors according to the NMR literature convention:
        # EFG tensors (i.e., |V_11| <= |V_22| <= |V_33|)
        # https://stackoverflow.com/a/50562995
        # sort based on ascending magnitude
        sort_order = np.argsort(e_vals**2)
        s_e_vals = e_vals[sort_order]
        s_e_vecs = e_vecs[:, sort_order]
        return (s_e_vals, s_e_vecs)

    def get_energy_levels(
        self, nu_0: float, efg: float, eta: float, theta: float, phi: float
    ) -> np.array:
        """
        Energy levels of the Zeeman + quadrupole Hamiltonians.
        """
        assert eta >= 0.0 and eta <= 1.0

        A_Q = self.quadrupole_frequency(efg)

        # components of the EFG tensor in the PAS w.r.t. the quantization axis
        # written in terms of: eta, theta, phi
        # Eqs. (22) - (27)
        V_xx = self.V_xx(eta, theta, phi)
        V_yy = self.V_yy(eta, theta, phi)
        V_zz = self.V_zz(eta, theta, phi)
        V_xy = self.V_xy(eta, theta, phi)
        V_xz = self.V_xz(eta, theta, phi)
        V_yz = self.V_yz(eta, theta, phi)

        # construct the NMR Hamiltonian
        H_Zeeman = -1.0 * nu_0 * self.I_z
        # Eq. (28)
        H_Quadrupole = A_Q * (
            V_zz * (3.0 * self.I_z @ self.I_z - self.I_2)
            + (V_xx - V_yy) * (self.I_x @ self.I_x - self.I_y @ self.I_y)
            + 2.0 * V_xy * (self.I_x @ self.I_y + self.I_y @ self.I_x)
            + 2.0 * V_xz * (self.I_x @ self.I_z + self.I_z @ self.I_x)
            + 2.0 * V_yz * (self.I_y @ self.I_z + self.I_z @ self.I_y)
        )
        H_NMR = H_Zeeman + H_Quadrupole

        # compute the eigenvalues/vectors
        e_vals, e_vecs = np.linalg.eigh(H_NMR, UPLO="L")
        return e_vals

    def get_n_quantum_transitions(self, eigenvalues: np.array, n: int = 1) -> np.array:
        """
        Calculate the positions of the n-quantum transitions for a set of eigenvalues.
        """
        assert n >= 1
        assert n <= self.N - 1
        nqt = np.empty(eigenvalues.size - n)
        for i in range(nqt.size):
            nqt[i] = (eigenvalues[i + n] - eigenvalues[i]) / n
        return nqt


@dataclass
class QuadrupoleSplitting8Li(QuadrupoleSplitting):
    """
    Class for calculating the quadrupole splitting of lithium-8.
    """

    def __init__(self):
        super().__init__(
            I=2,
            gamma=6.30221e6,  # https://www-nds.iaea.org/publications/indc/indc-nds-0794/
            Q=31.4,  # https://www-nds.iaea.org/publications/indc/indc-nds-0833/
        )

    def get_sq_lineshape(
        self,
        frequency: float,
        efg: float,
        eta: float,
        nu_0: float,
        theta: float,
        phi: float,
        sq_fwhm: float,
        sq_amplitude_1: float,
        sq_amplitude_2: float,
        sq_amplitude_3: float,
        sq_amplitude_4: float,
    ) -> float:
        """
        1-quantum lineshape.
        """
        eigenvalues = self.get_energy_levels(nu_0, efg, eta, theta, phi)
        sq_transitions = self.get_n_quantum_transitions(eigenvalues, 1)
        sq_amplitudes = np.array(
            [sq_amplitude_1, sq_amplitude_2, sq_amplitude_3, sq_amplitude_4]
        )
        sq_lineshape = (
            self.lorentzian(frequency, sq_transitions[0], sq_fwhm, sq_amplitudes[0])
            + self.lorentzian(frequency, sq_transitions[1], sq_fwhm, sq_amplitudes[1])
            + self.lorentzian(frequency, sq_transitions[2], sq_fwhm, sq_amplitudes[2])
            + self.lorentzian(frequency, sq_transitions[3], sq_fwhm, sq_amplitudes[3])
        )
        return sq_lineshape

    def get_dq_lineshape(
        self,
        frequency: float,
        efg: float,
        eta: float,
        nu_0: float,
        theta: float,
        phi: float,
        dq_fwhm: float,
        dq_amplitude_1: float,
        dq_amplitude_2: float,
        dq_amplitude_3: float,
    ) -> float:
        """
        2-quantum lineshape.
        """
        eigenvalues = self.get_energy_levels(nu_0, efg, eta, theta, phi)
        dq_transitions = self.get_n_quantum_transitions(eigenvalues, 2)
        dq_amplitudes = np.array([dq_amplitude_1, dq_amplitude_2, dq_amplitude_3])
        dq_lineshape = (
            self.lorentzian(frequency, dq_transitions[0], dq_fwhm, dq_amplitudes[0])
            + self.lorentzian(frequency, dq_transitions[1], dq_fwhm, dq_amplitudes[1])
            + self.lorentzian(frequency, dq_transitions[2], dq_fwhm, dq_amplitudes[2])
        )
        return dq_lineshape

    def get_tq_lineshape(
        self,
        frequency: float,
        efg: float,
        eta: float,
        nu_0: float,
        theta: float,
        phi: float,
        tq_fwhm: float,
        tq_amplitude_1: float,
        tq_amplitude_2: float,
    ) -> float:
        """
        3-quantum lineshape.
        """
        eigenvalues = self.get_energy_levels(nu_0, efg, eta, theta, phi)
        tq_transitions = self.get_n_quantum_transitions(eigenvalues, 3)
        tq_amplitudes = np.array([tq_amplitude_1, tq_amplitude_2])
        tq_lineshape = self.lorentzian(
            frequency, tq_transitions[0], tq_fwhm, tq_amplitudes[0]
        ) + self.lorentzian(frequency, tq_transitions[1], tq_fwhm, tq_amplitudes[1])
        return tq_lineshape

    def get_qq_lineshape(
        self,
        frequency: float,
        efg: float,
        eta: float,
        nu_0: float,
        theta: float,
        phi: float,
        qq_fwhm: float,
        qq_amplitude_1: float,
    ) -> float:
        """
        4-quantum lineshape.
        """
        eigenvalues = self.get_energy_levels(nu_0, efg, eta, theta, phi)
        qq_transitions = self.get_n_quantum_transitions(eigenvalues, 4)
        qq_amplitudes = np.array([qq_amplitude_1])
        qq_lineshape = self.lorentzian(
            frequency, qq_transitions[0], tq_fwhm, qq_amplitudes[0]
        )
        return qq_lineshape

    def get_lineshape(
        self,
        frequency: float,
        efg: float,
        eta: float,
        nu_0: float,
        theta: float,
        phi: float,
        sq_fwhm: float,
        sq_amplitude_1: float,
        sq_amplitude_2: float,
        sq_amplitude_3: float,
        sq_amplitude_4: float,
        dq_fwhm: float,
        dq_amplitude_1: float,
        dq_amplitude_2: float,
        dq_amplitude_3: float,
        bg_fwhm: float,
        bg_amplitude: float,
        baseline: float,
        slope: float,
    ) -> float:
        """
        Typical quadrupole-split lithium-8 lineshape:
        (1-quantum + 2-quantum transitions & a 'background' at the Larmor frequency.
        """
        # eigenvalues = self.get_energy_levels(nu_0, efg, eta, theta, phi)
        sq_lineshape = self.get_sq_lineshape(
            frequency,
            efg,
            eta,
            nu_0,
            theta,
            phi,
            sq_fwhm,
            sq_amplitude_1,
            sq_amplitude_2,
            sq_amplitude_3,
            sq_amplitude_4,
        )
        dq_lineshape = self.get_dq_lineshape(
            frequency,
            efg,
            eta,
            nu_0,
            theta,
            phi,
            dq_fwhm,
            dq_amplitude_1,
            dq_amplitude_2,
            dq_amplitude_3,
        )
        bg_lineshape = self.lorentzian(frequency, nu_0, bg_fwhm, bg_amplitude)
        return baseline + slope * frequency - sq_lineshape - dq_lineshape - bg_lineshape
