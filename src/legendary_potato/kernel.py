# coding: utf-8
"Generic kernels."

from calendar import monthrange
from datetime import datetime
from itertools import chain, combinations

import numpy as np

try:
    import scipy.integrate
except ImportError:
    # TODO: display warning
    scipy.integrate = None


def _subsets(set0):
    """Return all subsets of set0"""
    set0 = {ele for ele in set0}  # transform in set whatever iteratble it is
    return {
        frozenset(frozenset(ele) for ele in combinations(set0, n))
        for n in range(len(set0))
    }


def from_feature_map(phi, *args, **kwargs):
    r"""Define a kernel from a feature map.

    The mapping is the feature map function :math:`\phi` such that:

    .. math::
        K: (x_1, x_2) \mapsto <\phi(x_1), \phi(x_2)>

    define the kernel.

    Args:
        phi(fun): feature mapping.

    Returns:
        (fun) kernel function.
    """
    return lambda x1, x2: float(
        np.dot(phi(x1, *args, **kwargs).T, phi(x2, *args, **kwargs))
    )


def all_subset(set1, set2):
    """All  subset kernel for sets.

    This return the number of common subsets between set1 and set2
    """
    subsets1 = _subsets(set1)
    return len([ele for ele in _subsets(set2) if ele in subsets1])


def periodic_map(x, period=2 * np.pi):
    """Feature map for periodic features.

    Map onto a circle.
    """
    x = 2 * np.pi * x / period
    return np.array([np.cos(x), np.sin(x)])


def periodic(x1, x2, period=2 * np.pi):
    """Kernel based on periodic map.
    """
    return from_feature_map(periodic_map, period)(x1, x2)


def temporal_map(ts):
    """Feature map for mining human-based temporal behavior.

    Map onto several periodic map with periods usually used by humans.
    """
    periods = (
        1,  # second
        60,  # minute
        60 * 60,  # hour
        60 * 60 * 24,  # day
        60 * 60 * 24 * 7,  # week
        60 * 60 * 24 * 7 * 365.2425 / 12,  # month
        60 * 60 * 24 * 7 * 365.2425,  # year
    )
    datespec = datetime.utcfromtimestamp(ts)
    extraspecs = [
        (datespec.second, 60),
        (datespec.minute, 60),
        (datespec.hour, 24),
        (datespec.weekday(), 7),
        (datespec.day, 31),
        (datespec.day, monthrange(datespec.year, datespec.month)[1]),
        (datespec.month, 12),
    ]
    return np.hstack(
        chain(
            [coord for per in periods for coord in periodic_map(ts, per)],
            [
                coord
                for value, period in extraspecs
                for coord in periodic_map(value, period)
            ],
        )
    )


def temporal(ts1, ts2):
    """Kernel based on temporal map.
    """
    return from_feature_map(temporal_map)(ts1, ts2)


def l2(f1, f2, interval=(-1, 1)):
    r"""Kernel on functions square-integrable on interval.

    The result is:

    .. math::
        K(f_1, f_2) = \int_{interval} f_1(x) f_2(x) dx

    """
    return scipy.integrate.quad(
        lambda x: f1(x) * f2(x), interval[0], interval[1]
    )[0]


def matrix_weighted_factory(matrix=None):
    """Kernel modified by the symetric matrix.

    .. math::
        K(x_1, x_2) = x_1 matrix x_2^T

    Args:
        matrix (np.array): symmetric matrix

    Returns:
        (fun) kernel function

    """
    if not np.all(np.isclose(matrix - matrix.transpose(), 0)):
        raise ValueError("input matrix not symmetric")

    def kernel(x1, x2):
        """
        Args:
            x1 (np.array): a 1D array
            x2 (np.array): same shape as x1

        """
        if matrix is None:
            return x1.dot(x2)
        return x1.dot(matrix).dot(x2)

    return kernel


def matrix_weighted_example(x1, x2):
    """Example of matrix weigted.

    Args:
        x1, x2: 1x5 ndarray

    """
    matrix = np.array(
        [
            [6, 2, 5, 17, -19],
            [2, 18, 1, 16, 11],
            [5, 1, 10, -1, 2],
            [17, 16, -1, -12, 7],
            [-19, 11, 2, 7, 4],
        ]
    )
    kernel = matrix_weighted_factory(matrix)
    return kernel(x1, x2)
