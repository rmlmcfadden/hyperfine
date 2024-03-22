"""Facilities for creating/parsing SRIM input/output files.

The Stopping and Range of Ions in Matter (SRIM) is a suite of programs for
calculating and simulating projectile-matter interactions during
ion-implantation.

http://srim.org/
"""

import string
import warnings
import numpy as np


def get_line_number(
    filename: str = "RANGE_3D.txt",
    phrase: str = "-------  ----------- ----------- -----------",
    encoding: str = "gbk",
) -> int:
    """Deduce where the data columns start in the ``RANGE_3D.txt`` file output by ``TRIM.exe``.

    Args:
        filename: Name of the ``RANGE_3D.txt`` file output from ``SRIM``.
        phrase: Character sequence to find in the file.
        encoding: Encoding of the ``RANGE_3D.txt`` file.

    Returns:
        The line number in ``RANGE_3D.txt`` that matches ``phrase``.
    """

    # https://stackoverflow.com/a/3961303
    with open(filename, "r", encoding=encoding) as fh:
        for i, line in enumerate(fh, 1):
            if phrase in line:
                return i
    # return a negative index if the phrase is not found
    return -1


def _unix2dos(filename: str) -> None:
    """Convenience method replicating the functionality of ``unix2dos``.

    ``unix2dos`` is a non-standard Unix program for converting line breaks in a
    text file from Unix format (Line feed) to DOS format (carriage return +
    Line feed).

    See: https://en.wikipedia.org/wiki/Unix2dos

    Args:
        filename: Name of the file run ``unix2dos`` on.
    """

    with open(filename, "rb") as input_file:
        input_string = input_file.read()

    # systematically replace & harmonize all line breaks
    output_string = (
        input_string.replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
        .replace(b"\n", b"\r\n")
    )

    if len(output_string) != len(input_string):
        print(
            "WARNING : "
            + f"output string length ({len(output_string)})"
            + " != "
            + f"input string length ({len(input_string)})"
        )

    with open(filename, "wb") as output_file:
        output_file.write(output_string)


def create_trim_dat(
    atomic_number: int,
    energy_eV: float,
    angle_mean_deg: float = 0.0,
    angle_sigma_deg: float = np.finfo(float).eps,
    description: str = "",
    total_ions: int = 99999,
) -> None:
    """Create a ``TRIM.DAT`` file for use in an advanced ``TRIM.exe`` calculation.

    Args:
        atomic_number: Atomic number of the projectile.
        energy_eV: Energy of the projectile (eV).
        angle_mean_deg: Mean angle of incidence with respect to the surface normal (degrees).
        angle_sigma_deg: Standard deviation of the angle of incidence (degrees).
        description: Description of simulation.
        total_ions: Total number of ions to generate.
    """

    assert (atomic_number >= 1) & (atomic_number <= 92)
    assert energy_eV >= 0.0
    assert angle_sigma_deg > 0.0

    with open("TRIM.dat", "w", encoding="gbk") as fh:
        # write the (empty) header lines
        for i in range(7):
            fh.write("\n")
        fh.write(description + "\n")
        fh.write("\n")
        columns = [
            "Event Name",
            "Atomic Number",
            "Energy (eV)",
            "X (A)",
            "Y (A)",
            "Z (A)",
            "cos(X)",
            "cos(Y)",
            "cos(Z)",
        ]
        fh.write(" ".join(columns) + "\n")

        # initialize the pseudo random number generator (PRNG)
        prng = np.random.Generator(np.random.PCG64(seed=None))

        for _ in range(total_ions):
            # generate a random 5 character alphanumeric event identifier
            event = "".join(
                prng.choice(
                    [*(string.ascii_letters + string.digits)],
                    size=5,
                )
            )

            # start at the surface
            x = 0.0

            # start at the same lateral point
            y = 0.0
            z = 0.0

            # random polar angles
            theta_deg = np.abs(
                prng.normal(
                    loc=angle_mean_deg,
                    scale=angle_sigma_deg,
                )  # assume a normal distribution
            )  # must be positive (i.e., a folded normal distribution)
            phi_deg = prng.uniform(
                0.0,
                360.0,
            )

            # convert from degrees to radians
            theta_rad = np.deg2rad(theta_deg)
            phi_rad = np.deg2rad(phi_deg)

            # directional cosines
            cos_x = np.cos(theta_rad)  # 1.0
            cos_y = np.sin(theta_rad) * np.cos(phi_rad)  # 0.0
            cos_z = np.sin(theta_rad) * np.sin(phi_rad)  # 0.0

            # verify
            assert np.isclose(cos_x * cos_x + cos_y * cos_y + cos_z * cos_z, 1.0)

            row = [
                f"{event}",
                f"{atomic_number}",
                f"{energy_eV}",
                f"{x}",
                f"{y}",
                f"{z}",
                f"{cos_x}",
                f"{cos_y}",
                f"{cos_z}",
            ]

            fh.write(" ".join(row) + "\n")

        # end things with an empty line
        fh.write("\n")

    # TRIM.dat needs to be windows compatible
    _unix2dos("TRIM.dat")


def _ellipsoid_unit_normal_vector(
    x: float,
    y: float,
    z: float,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    e_x_0: float = 0.0,
    e_y_0: float = 0.0,
    e_z_0: float = 0.0,
) -> np.array:
    """Calculate the unit vector normal to the surface of an ellipsoid at point :math:`p = (x, y, z)`.

    Args:
        x: Spatial position :math:`x`.
        y: Spatial position :math:`y`.
        z: Spatial position :math:`z`.
        a: Ellipsoid semi-axis :math:`a`.
        b: Ellipsoid semi-axis :math:`b`.
        c: Ellipsoid semi-axis :math:`c`.
        e_x_0: Ellipsoid :math:`x`-coordinate centre.
        e_y_0: Ellipsoid :math:`y`-coordinate centre.
        e_z_0: Ellipsoid :math:`z`-coordinate centre.

    Returns:
        The unit vector normal to the ellipsoid surface at :math:`p`.
    """

    df_dx = 2.0 * (x - e_x_0) / np.square(a)
    df_dy = 2.0 * (y - e_y_0) / np.square(b)
    df_dz = 2.0 * (z - e_z_0) / np.square(c)
    magnitude = np.sqrt(np.square(df_dx) + np.square(df_dy) + np.square(df_dz))

    return np.array([df_dx, df_dy, df_dz]) / magnitude


def create_trim_dat_ellipsoid(
    beam_atomic_number: int,
    beam_energy_eV: float,
    beam_fwhm_y_mm: float = 3.0,
    beam_fwhm_z_mm: float = 3.0,
    beam_position_y_mm: float = 0.0,
    beam_position_z_mm: float = 0.0,
    ellipsoid_semiaxis_a_mm: float = 1.0,
    ellipsoid_semiaxis_b_mm: float = 5.0,
    ellipsoid_semiaxis_c_mm: float = 2.0,
    ellipsoid_position_x_mm: float = 0.0,
    ellipsoid_position_y_mm: float = 0.0,
    ellipsoid_position_z_mm: float = 0.0,
    description: str = "",
    total_ions: int = 99999,
) -> None:
    """Create a ``TRIM.DAT`` file for use in an advanced ``TRIM.exe`` calculation.

    In a standard TRIM calculation, the program assumes a flat target with
    infinite lateral dimensions. Within this constraint, this function attempts
    to transform the projectile ion trajectories to approximate incidence with
    an ellipsoid-shaped target.

    As with all TRIM calculations, the beam/projectile direction is initially
    parallel the :math:`x`-axis.

    Args:
        beam_atomic_number: Projectile atomic number.
        beam_energy_eV: Projectile energy (eV).
        beam_fwhm_y_mm: FWHM of the projectile beam's lateral spread in the :math:`y`-direction.
        beam_fwhm_z_mm: FWHM of the projectile beam's lateral spread in the :math:`z`-direction.
        ellipsoid_semiaxis_a_mm: Ellipsoid semi-axis :math:`a`.
        ellipsoid_semiaxis_b_mm: Ellipsoid semi-axis :math:`b`.
        ellipsoid_semiaxis_c_mm: Ellipsoid semi-axis :math:`c`.
        ellipsoid_position_x_mm: Ellipsoid :math:`x`-coordinate centre.
        ellipsoid_position_y_mm: Ellipsoid :math:`y`-coordinate centre.
        ellipsoid_position_z_mm: Ellipsoid :math:`z`-coordinate centre.
        description: Description of simulation.
        total_ions: Total number of ions to generate.
    """

    assert (beam_atomic_number >= 1) & (beam_atomic_number <= 92)
    assert beam_energy_eV >= 0.0
    assert (
        (ellipsoid_semiaxis_a_mm > 0.0)
        & (ellipsoid_semiaxis_b_mm > 0.0)
        & (ellipsoid_semiaxis_c_mm > 0.0)
    )
    assert (beam_fwhm_y_mm > 0.0) & (beam_fwhm_z_mm > 0.0)

    with open("TRIM.dat", "w", encoding="gbk") as fh:
        # write the (empty) header lines
        for i in range(7):
            fh.write("\n")
        fh.write(description + "\n")
        fh.write("\n")
        columns = [
            "Event Name",
            "Atomic Number",
            "Energy (eV)",
            "X (A)",
            "Y (A)",
            "Z (A)",
            "cos(X)",
            "cos(Y)",
            "cos(Z)",
        ]
        fh.write(" ".join(columns) + "\n")

        # initialize the pseudo random number generator (PRNG)
        prng = np.random.Generator(np.random.PCG64(seed=None))

        for _ in range(total_ions):
            # generate a random 5 character alphanumeric event identifier
            event = "".join(
                prng.choice(
                    [*(string.ascii_letters + string.digits)],
                    size=5,
                )
            )

            # conversion
            angstrom_per_millimeter = 1e7
            sigma_per_fwhm = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # start at the surface
            x_0_mm = 0.0

            # calculate "starting" position of implanted ion on an ellipsoid surface
            # using the target/beam dimensions
            while True:
                # random starting positions in the y-z plane
                y_0_mm = prng.normal(
                    loc=beam_position_y_mm,
                    scale=beam_fwhm_y_mm * sigma_per_fwhm,
                    size=1,
                )
                z_0_mm = prng.normal(
                    loc=beam_position_z_mm,
                    scale=beam_fwhm_z_mm * sigma_per_fwhm,
                    size=1,
                )
                with warnings.catch_warnings():
                    # ignore runtime warning likely generated by np.sqrt
                    warnings.simplefilter("ignore")
                    # calculate the corresponding x_0 position
                    x_0_mm = (
                        ellipsoid_semiaxis_a_mm
                        * np.sqrt(
                            1.0
                            - np.square(
                                (y_0_mm - ellipsoid_position_y_mm)
                                / ellipsoid_semiaxis_b_mm
                            )
                            - np.square(
                                (z_0_mm - ellipsoid_position_z_mm)
                                / ellipsoid_semiaxis_c_mm
                            )
                        )
                        + ellipsoid_position_x_mm
                    )
                # only accept sensible results (e.g., ions that actually hit the target)!
                if (
                    (np.abs(y_0_mm) <= ellipsoid_semiaxis_b_mm)
                    & (np.abs(z_0_mm) <= ellipsoid_semiaxis_c_mm)
                    & np.isfinite(x_0_mm)
                ):
                    break

            # calculate the angle of incidence
            surface_normal = _ellipsoid_unit_normal_vector(
                x_0_mm,
                y_0_mm,
                z_0_mm,
                ellipsoid_semiaxis_a_mm,
                ellipsoid_semiaxis_b_mm,
                ellipsoid_semiaxis_c_mm,
                ellipsoid_position_x_mm,
                ellipsoid_position_y_mm,
                ellipsoid_position_z_mm,
            )
            beam_direction = np.array([1.0, 0.0, 0.0])
            theta_deg = np.rad2deg(
                np.arccos(np.dot(beam_direction, surface_normal))
            )  # projectile's incident angle (relative to surface normal) in degrees

            # random polar angle
            phi_deg = prng.uniform(
                0.0,
                360.0,
            )

            # convert from degrees to radians
            theta_rad = np.deg2rad(theta_deg)
            phi_rad = np.deg2rad(phi_deg)

            # directional cosines
            cos_x = np.cos(theta_rad)  # 1.0
            cos_y = np.sin(theta_rad) * np.cos(phi_rad)  # 0.0
            cos_z = np.sin(theta_rad) * np.sin(phi_rad)  # 0.0

            # the ion starts at the target's surface
            x = 0.0
            # with a lateral spread defined by the beam position/dimensions
            y = y_0_mm * angstrom_per_millimeter
            z = z_0_mm * angstrom_per_millimeter

            # verify
            assert np.isclose(cos_x * cos_x + cos_y * cos_y + cos_z * cos_z, 1.0)

            row = [
                f"{event}",
                f"{beam_atomic_number}",
                f"{beam_energy_eV}",
                f"{x}",
                f"{y.item()}",
                f"{z.item()}",
                f"{cos_x.item()}",
                f"{cos_y.item()}",
                f"{cos_z.item()}",
            ]

            fh.write(" ".join(row) + "\n")

        # end things with an empty line
        fh.write("\n")

    # TRIM.dat needs to be windows compatible
    _unix2dos("TRIM.dat")
