"""Utilities for creating and running DPASS input files.

DPASS (``DPASS.exe``) is a Windows program that acts as a frontend for
(electronic) stopping cross section data calculated using the PASS program.
It produces tabulated output from apre-calculated database of PASS outputs
(``DPASS_DB.dat``), either through a GUI or a command line (batch) interface.

For further details, see:

https://www.sdu.dk/da/dpass

https://doi.org/10.1016/j.nimb.2018.10.047
"""

import configparser
import os


def create_job_file(
    Z_projectile: int,
    Z_target: int,
    print_header: int = 1,
    output_units: int = 0,
    target_mass: None | float = None,
    target_density: None | float = None,
    raw_data: int = 1,
    energy_min: float = 0.001,
    energy_max: float = 1000.0,
    energy_points: int = 1000,
    mesh: int = 1,
) -> None:
    """Create a new job file (``DPASS.job``) in the current directory.

    Args:
        Z_projectile: Atomic number of the projectile atom.
        Z_target: Atomic number of the target atom.
        print_header: Print header lines in ``DPASS.out`` (0 = no header; 1 = print header).
        output_units: Units for the stopping cross sections (0 = 10\ :sup:`-15` eV cm\ :sup:`2`; 1 = MeV cm\ :sup:`2` mg\ :sup:`-1`; 2 = eV nm\ :sup:`-1`).
        target_mass: Target atomic mass [for ``output_units`` = 1 or 2] (u).
        target_density: Target density [for ``output_units`` = 2] (g cm\ :sup:`-3`).
        raw_data: Use tabulated or spline interpolated points (0 =  spline interpolated points; 1 = raw tabulated points).
        energy_min: Minimum projectile energy [for ``raw_data`` = 0] (MeV u\ :sup:`-1`).
        energy_max: Maximum projectile energy [for ``raw_data`` = 0] (MeV u\ :sup:`-1`).
        energy_points: Number of energy points [for ``raw_data`` = 0].
        mesh: Sampling mesh for energy points [for ``raw_data = 0] (0 = linear; 1 = logarithmic).
    """

    # check that the inputs are sensible
    assert (Z_projectile > 0) & (Z_projectile <= 92)
    assert (Z_target > 0) & (Z_target <= 92)
    assert energy_min >= 0.001
    assert energy_max <= 1000.0

    # https://docs.python.org/3/library/configparser.html
    config = configparser.ConfigParser()
    config["Main"] = {
        "Projectile": f"{Z_projectile}",
        "Target": f"{Z_target}",
    }
    config["Options"] = {
        "Header": f"{print_header}",
        "Units": f"{output_units}",
        "A": f"{target_mass}",
        "Rho": f"{target_density}",
    }
    config["Spline"] = {
        "Raw": f"{raw_data}",
        "Emin": f"{energy_min}",
        "Emax": f"{energy_max}",
        "Epts": f"{energy_points}",
        "Log": f"{mesh}",
    }
    # write the config to a file
    with open("DPASS.job", "w") as jobfile:
        config.write(jobfile)


def run_dpass(
    path: str = "",
    batch_mode: bool = True,
) -> int:
    """Run DPASS (in batch mode) on Linux using Wine.

    Args:
        path: Location of ``DPASS.exe`` and ``DPASS_DB.dat``.
        batch_mode: Flag to run DPASS in batch mode.

    Returns:
        The return code from DPASS (0 = no error; nonzero indicates abnormal termination).
    """

    # move to the location of DPASS.exe and DPASS_DB.dat
    rc = os.system(
        "cd %s && wine DPASS.exe %s" % (path, "-batch" if batch_mode else "")
    )

    # return the return code from executing DPASS
    return rc
