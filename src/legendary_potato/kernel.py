"Generic kernels."

from itertools import combinations
import numpy as np

try:
    import scipy.integrate
except ImportError:
    # TODO: display warning
    scipy.integrate = None


def _subsets(set0):
    """Return all subsets of set0"""
    set0 = {ele for ele in set0}  # transform in set whatever iteratble it is
    return {frozenset(frozenset(ele) for ele in combinations(set0, n))
            for n in range(len(set0))}


def from_feature_map(mapping, *args, **kwargs):
    """Define a kernel from a feature map.

    The mapping is the feature map function $\phi$ such that:
    \[
    K: (x_1, x_2) \mapsto <\phi(x_1), \phi(x_2)>
    \]
    define the kernel.
    """
    return lambda x1, x2: np.dot(mapping(x1, *args, **kwargs).transpose(),
                                 mapping(x2, *args, **kwargs))[0, 0]


def all_subset(set1, set2):
    """All  subset kernel for sets.

    This return the number of common subsets between set1 and set2
    """
    subsets1 = _subsets(set1)
    return len([ele for ele in _subsets(set2) if ele in subsets1])


def periodic_map(x, period=2*np.pi):
    """Feature map for periodic features.

    Map onto a circle.
    """
    x = 2 * np.pi * x / period
    return np.array([np.cos(x), np.sin(x)])


def periodic(x1, x2, period=2*np.pi):
    """Kernel based on periodic map.
    """
    return from_feature_map(periodic_map, period)(x1, x2)


def l2(f1, f2, interval=(-1, 1)):
    """Kernel on functions square-integrable on interval.

    The result is:
    .. math:
    K(f_1, f_2) = \int_{interval} f_1(x) f_2(x) dx
    """
    return scipy.integrate.quad(lambda x: f1(x) * f2(x),
                                interval[0],
                                interval[1])[0]


def matrix_weighted(x1, x2, matrix=None):
    """Kernel modified by the symetric matrix.

    If no matrix is provided, the identity matrix is used

    .. math:
    K(x_1, x_2) = x_1 matrix x_2^T

    x1 -- a 1 dimensional ndarray
    x2 -- a 1 dimensional ndarray with the same shape as x1
    """
    if matrix is None:
        matrix = np.identity(len(x1))
    return x1.dot(matrix).dot(x2)
