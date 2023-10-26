import json
import numpy as np
import iminuit
from iminuit import Minuit
from typing import Callable
from joblib import Parallel, delayed, cpu_count


def minos2dict(minuit: Minuit) -> dict:
    """
    Convert the MINOS errors to a dictionary (e.g., for serialization).
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


def minuit2json(minuit: Minuit, filename: str) -> None:
    """
    Serialize fitting results from an `iminuit.Minuit` object to a JSON file.
    """

    # convert the results to a dictionary
    results = {
        "values": minuit.values.to_dict(),
        "errors": minuit.errors.to_dict(),
        "limits": minuit.limits.to_dict(),
        "fixed": minuit.fixed.to_dict(),
        "merrors": minos2dict(minuit),
    }

    # write the results dictionary to a file
    with open(filename, "w") as fh:
        json.dump(results, fh, indent="\t")


def json2minuit(minuit: Minuit, filename: str) -> None:
    """
    De-serialize fitting results from a JSON file to an `iminuit.Minuit` object.
    """

    # read the results into a dictionary
    with open(filename, "r") as fh:
        results = json.load(fh)

    # restore the fit quantities
    for quantity in ["values", "errors", "limits", "fixed"]:
        if quantity in results:
            for key, value in results[quantity].items():
                minuit.__getattribute__(quantity)[key] = value

    # TODO: restore the minos errors


class GenericLeastSquares3D:
    """
    Generic 3D least-squares cost function with error.

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
