import configparser
import os


def create_job_file(
    Z_projectile: int,
    Z_target: int,
    print_header: int = 1,
    output_units: int = 0,
    target_mass: str = "",
    target_density: str = "",
    raw_data: bool = True,
) -> None:
    """
    Create a new job file (DPASS.job) in the current directory.
    """

    # check that the Zs are sensible
    assert (Z_projectile > 0) & (Z_projectile <= 92)
    assert (Z_target > 0) & (Z_target <= 92)

    # https://docs.python.org/3/library/configparser.html
    config = configparser.ConfigParser()
    config["Main"] = {
        "Projectile": "%d" % Z_projectile,
        "Target": "%d" % Z_target,
    }
    config["Options"] = {
        "Header": "%d" % print_header,
        "Units": "%d" % output_units,
        "A": target_mass,
        "Rho": target_density,
    }
    config["Spline"] = {
        "Raw": "%d" % raw_data,
        "Emin": "0.001",
        "Emax": "1000.0",
        "Epts": "1000",
        "Log": "1",
    }
    # write the config to a file
    with open("DPASS.job", "w") as jobfile:
        config.write(jobfile)


def run_dpass(path: str = "", batch_mode: bool = True) -> int:
    """
    Run DPASS (in batch mode) on Linux using Wine.
    """

    # move to the location of DPASS.exe and DPASS_DB.dat
    rc = os.system(
        "cd %s && wine DPASS.exe %s" % (path, "-batch" if batch_mode else "")
    )

    # return the return code from executing DPASS
    return rc
