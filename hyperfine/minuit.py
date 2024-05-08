"""Utilities for interacting with the ``iminuit`` package.
"""

import json
import numpy as np
import iminuit
from iminuit import Minuit
from typing import Callable
from joblib import Parallel, delayed, cpu_count


def minos2dict(minuit: Minuit) -> dict:
    """Convert the MINOS errors to a dictionary.

    Convert the MINOS (i.e., asymmetric) errors determined by the MINUIT2
    minimizer to a dictionary (e.g., for serialization).

    Args:
        minuit: The ``iminuit.Minuit`` object.

    Returns:
        A dictionary containing the MINOS errors.
    """

    # empty dictionary to hold the results
    minos = {}

    # loop over all fit parameters
    for item in minuit.merrors.items():
        # extract the parameter name & MError class pairs
        par, merror = item

        # empty dictionary to hold parameter specific values
        mdict = {}

        # loop over all MError slots
        for slot in merror.__slots__:
            # fill the parameter minos dictionary
            mdict[slot] = merror.__getattribute__(slot)

        # add the parameter specific minos dictionary to the minos dictionary
        minos[par] = mdict

    # return the filled minos dictionary
    return minos


def covariance2dict(minuit: Minuit) -> dict:
    """Convert the covariance matrix to a serializable dictionary.

    Convert the covariance matrix (i.e., the MIGRAD or HESSE errors) determined
    by the MINUIT2 minimizer to a *serializable* dictionary (i.e., one that is
    compatible with JSON).

    Args:
        minuit: The ``iminuit.Minuit`` object.

    Returns:
        A serializable dictionary containing the covariance matrix (i.e., errors computed by MIGRAD or HESSE).
    """

    # empty dictionary to hold the matrix
    covariance = {}

    # check if the covariance matrix exists
    if minuit.covariance is not None:
        # loop over all fit parameters
        for par1 in minuit.parameters:
            # for each parameter, create an empty dictionary...
            covariance[par1] = {}
            # loop over all fit parameters
            for par2 in minuit.parameters:
                # ...to be filled with the covariance for each
                # parameter pair combination
                covariance[par1][par2] = minuit.covariance[par1, par2]

    # return the covariance matrix
    return covariance


def correlation2dict(minuit: Minuit) -> dict:
    """Convert the correlation matrix to a serializable dictionary.

    Convert the correlation matrix (i.e., derived from the MIGRAD or HESSE errors)
    determined by the MINUIT2 minimizer to a *serializable* dictionary
    (i.e., one that is compatible with JSON).

    Args:
        minuit: The ``iminuit.Minuit`` object.

    Returns:
        A serializable dictionary containing the correlation matrix.
    """

    # empty dictionary to hold the matrix
    correlation = {}

    # check if the covariance matrix exists
    if minuit.covariance is not None:
        # loop over all fit parameters
        for par1 in minuit.parameters:
            # for each parameter, create an empty dictionary...
            correlation[par1] = {}
            # loop over all fit parameters
            for par2 in minuit.parameters:
                # ...to be filled with the (normalized) covariance for each
                # parameter pair combination
                correlation[par1][par2] = minuit.covariance[par1, par2] / np.sqrt(
                    minuit.covariance[par1, par1] * minuit.covariance[par2, par2]
                )

    # return the covariance matrix
    return correlation


def fmin2dict(minuit: Minuit) -> dict:
    """Convert the metadata about the cost function minimum to a serializable dictionary.

    Convert the contents of ``iminuit.Minuit.fmin`` (i.e, an ``iminuit.util.FMin`` object)
    to a *serializable* dictionary (i.e., one that is compatible with JSON).
    The ``iminuit.util.FMin`` object provides detailed metadata about the function minimum
    (e.g., the value of the cost function, validity of the minimum, etc.).
    The metadata is useful for checking what happened when a fit didn't converge.

    Args:
        minuit: The ``iminuit.Minuit`` object.

    Returns:
        A serializable dictionary containing the metadata about the cost function minimum.
    """

    # create an empty dictionary
    fmin = {}

    if minuit.fmin is None:
        # return the empty dictionary if the cost function has not been minimized
        return fmin

    else:
        # populate the dictionary with all of fmin's public attributes
        for key in sorted(dir(minuit.fmin)):
            if key.startswith("_"):
                continue
            value = getattr(minuit.fmin, key)
            fmin[key] = value

        return fmin


def minuit2json(minuit: Minuit, filename: str) -> None:
    """Serialize fitting results from an ``iminuit.Minuit`` object to a JSON file.

    Serialization includes data for: ``values``, ``errors``, ``limits``,
    ``fixed``, ``covariance``, ``merrors``, and ``fmin``. Parameter correlations
    (derived from ``covariance``) are also included.

    Args:
        minuit: The ``iminuit.Minuit`` object.
        filename: Name of JSON file to save the serialized fit results.
    """

    # convert the results to a dictionary
    results = {
        "values": minuit.values.to_dict(),
        "errors": minuit.errors.to_dict(),
        "limits": minuit.limits.to_dict(),
        "fixed": minuit.fixed.to_dict(),
        "covariance": covariance2dict(minuit),
        "correlation": correlation2dict(minuit),
        "merrors": minos2dict(minuit),
        "fmin": fmin2dict(minuit),
    }

    # write the results dictionary to a file
    with open(filename, "w") as fh:
        json.dump(results, fh, indent="\t")


def json2minuit(minuit: Minuit, filename: str) -> None:
    """De-serialize fitting results from a JSON file to an ``iminuit.Minuit`` object.

    De-serialization includes data for: ``values``, ``errors``, ``limits``,
    and ``fixed`` (``covariance`` and ``merrors`` are not yet implemented).

    Args:
        minuit: The ``iminuit.Minuit`` object.
        filename: Name of JSON file containing the serialized fit results.
    """

    # read the results into a dictionary
    with open(filename, "r") as fh:
        results = json.load(fh)

    # restore the fit quantities
    for quantity in ["values", "errors", "limits", "fixed"]:
        if quantity in results:
            for key, value in results[quantity].items():
                minuit.__getattribute__(quantity)[key] = value

    # TODO: restore the covariance matrix?
    # TODO: restore the minos errors?


class GenericLeastSquares3D:
    """Generic 3D least-squares cost function with error.

    https://iminuit.readthedocs.io/en/stable/tutorial/generic_least_squares.html
    """

    # for Minuit to compute errors correctly
    errordef = iminuit.Minuit.LEAST_SQUARES

    def __init__(
        self,
        model: Callable,
        x: float,
        y: float,
        z: float,
        value: float,
        error: float,
        verbose: bool = False,
    ):
        self.model = model  # model predicts y for given x
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
        self.value = np.asarray(value)
        self.error = np.asarray(error)
        self.verbose = verbose
        # for Minuit to compute errors correctly
        self.errordef = iminuit.Minuit.LEAST_SQUARES

    def _eval_chi2(
        self, x: float, y: float, z: float, v: float, e: float, *par
    ) -> float:
        """
        Convenience method for calculating the chi2 contribution of a single set of points.
        """
        return ((v - self.model(x, y, z, *par)) / e) ** 2

    # we accept a variable number of model parameters
    def __call__(self, *par) -> float:
        """
        Functor for computing the chi2.
        """

        # serial algorithm for calculating the chi2
        """
        chi2 = 0.0
        for x, y, z, v, e in zip(self.x, self.y, self.z, self.value, self.error):
            arg = (v - self.model(x, y, z, *par)) / e
            chi2 += arg**2
        """
        # https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        chi2_list = Parallel(
            n_jobs=cpu_count(only_physical_cores=False),
            # n_jobs=-2, # all but 1 cpu
            backend="loky",
            # prefer="processes", # "threads"
            verbose=10,
            batch_size="auto",
        )(
            delayed(self._eval_chi2)(x, y, z, v, e, *par)
            for x, y, z, v, e in zip(self.x, self.y, self.z, self.value, self.error)
        )
        chi2 = sum(chi2_list)

        # optionally print the result to the terminal.
        if self.verbose:
            print(
                "\nχ² / ndof (\n"
                + ",\n".join(["\t%+.6e" % p for p in par])
                + "\n) = %.6f / max(%d - %d, 1)\n  = %.6f\n"
                % (chi2, self.x.size, len(par), chi2 / max(self.x.size - len(par), 1))
            )
        return chi2


class LeastSquares3D(GenericLeastSquares3D):
    """
    Improved 3D least-squares cost function with error that deduces the parameter
    names and the numbers of parameters from the model signature via introspection.
    This is done by generating a function signature for the LeastSquares3D class
    from the signature of the model.

    https://iminuit.readthedocs.io/en/stable/tutorial/generic_least_squares.html
    """

    def __init__(
        self,
        model: Callable,
        x: float,
        y: float,
        z: float,
        value: float,
        error: float,
        verbose: bool = False,
    ):
        super().__init__(model, x, y, z, value, error, verbose)
        # self.func_code = iminuit.util.make_func_code(iminuit.util.describe(model)[3:])

        pars = iminuit.util.describe(model, annotations=True)
        model_args = iter(pars)
        next(model_args)  # skip x
        next(model_args)  # skip y
        next(model_args)  # skip z
        self._parameters = {k: pars[k] for k in model_args}
