"""Convenience utilities."""

from typing import Sequence, TypeVar
import warnings
import numpy as np
import matplotlib


# generic type
T = TypeVar("T")


def sampling_index(
    array: Sequence[T],
    n_elements: int,
    begin_offset: int = 0,
    end_offset: int = 0,
) -> Sequence[T]:
    """Sample evenly spaced indices from an array.

    Inspired by: https://stackoverflow.com/a/50685454

    Args:
        array: The array to sample indices from.
        n_elements: Number of indices to select.
        begin_offset: Offset from ``array``'s beginning index when sampling.
        end_offset: Offset from ``array``'s ending index when sampling.

    Return:
        An array of the sampled indices.

    Example:
        .. code-block::

           import numpy as np
           from hyperfine.utilities import sampling_index

           a = np.linspace(0.0, 30.0, 100)

           n_elem = 5,
           beg_off = 0
           end_off = 0
           idx = sampling_index(a, n_elem, beg_off, end_off)
    """

    # type checking
    if type(n_elements) is not int:
        warning.warn(f"casting `n_elements` to type `int`")
        n_elements = int(n_elements)

    if type(begin_offset) is not int:
        warning.warn(f"casting `begin_offset` to type `int`")
        begin_offset = int(begin_offset)

    if type(end_offset) is not int:
        warning.warn(f"casting `end_offset` to type `int`")
        end_offset = int(end_offset)

    # return the list indexes
    return np.round(
        np.linspace(
            begin_offset,
            len(array) - 1 - end_offset,
            n_elements,
        )
    ).astype(int)


def split(
    handles_labels: Sequence[T],
    plot_type: str,
) -> Sequence[T]:
    """Split legend handles/labels based on plot type.

    Split the sequence of legend handles/labels based on the underlying plot type, which is useful for creating multiple legends on a single figure.

    Adapted from: https://stackoverflow.com/a/76485654

    Args:
        handles_labels: Sequence of plot handles/labels.
        plot_type: Type of plot to "pluck" from the sequence of plot handles/labels.

    Returns:
        The subset of the sequence of plot handles/labels corresponding to the desired plot type.

    Example:
        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from hyperfine.utilities import split

           x = np.linspace(0, 10, 10)
           y = 2 * x

           fig, ax = plt.subplots(1, 1)
           ax.plot(x, y, "-", zorder=1, label="plot")
           ax.axvspan(5, 9, color="lightgrey", zorder=0, label="axvspan")

           handles_labels = ax.get_legend_handles_labels()

           l_1 = ax.legend(*split(handles_labels, "axvspan"), loc="upper left")
           ax.add_artist(l_1)
           l_2 = ax.legend(*split(handles_labels, "plot"), loc="lower right")

           plt.show()
    """

    # map the plot type string to its underlying type
    # TODO: add more types!
    types = dict(
        plot=matplotlib.lines.Line2D,
        errorbar=matplotlib.container.ErrorbarContainer,
        scatter=matplotlib.collections.PathCollection,
        axvspan=matplotlib.patches.Polygon,
    )

    try:
        plot_type = types[plot_type]
    except KeyError:
        raise ValueError("Invalid plot type.")

    return zip(*((h, l) for h, l in zip(*handles_labels) if type(h) is plot_type))
