"Generic kernels."

from itertools import combinations
import numpy as np


def _subsets(set0):
    """Return all subsets of set0"""
    return {frozenset(frozenset(ele) for ele in combinations(set0, n))
            for n in range(len(set0))}


def from_feature_map(mapping, *args, **kwargs):
    """Define a kernel from a feature map.

    The mapping is the feature map function $\phi$ such that:
    \[
    K: (x_1, x_2) \mapsto <\phi(x_1), \phi(x_2)>
    \]
    define the kernel.

    /!\ this function overwrite the kernel.
    """
    return lambda x1, x2: np.dot(mapping(x1, *args, **kwargs),
                                 mapping(x2, *args, **kwargs))


def all_subset(set1, set2):
    """All  subset kernel for sets.

    This return the number of common subsets between set1 and set2
    """
    subsets1 = _subsets(set1)
    return len(ele for ele in _subsets(set2) if ele in subsets1)


def periodic_map(x, period=2*np.pi):
    """Feature map for periodic features.

    Map onto a circle.
    """
    x = 2 * np.pi * x / period
    return (np.cos(x), np.sin(x))


def periodic(x1, x2, period=2*np.pi):
    """Kernel based on periodic map.
    """
    return from_feature_map(periodic_map, period)(x1, x2)
