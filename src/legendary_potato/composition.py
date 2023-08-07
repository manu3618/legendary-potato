"""
Functions to create kernels from already existing kernels.

"""

import numpy as np


def normalize(kernel=np.dot, *args, **kwargs):
    """Return the normalized version.

    This correspond to the new kernel

    .. math::
        K(x_1, x_2) = \\frac{k_0(x_1, x2)}{\\sqrt(k_0(x_1, x_1) k_0(x_2, x_2))}

    This is equivalent to normalize the feature map:

    .. math::
        \\Phi(x) = \\frac{\\phi(x)}{\\|\\phi(x)\\|}

    Returns:
        (fun) the kernel function.

    """

    def normalized_kernel(x1, x2, *args, **kwargs):
        """actual kernel to return"""
        denom = np.sqrt(
            kernel(x1, x1, *args, **kwargs) * kernel(x2, x2, *args, **kwargs)
        )
        if denom == 0:
            return 0
        else:
            return kernel(x1, x2, *args, **kwargs) / denom

    return normalized_kernel


def polynomial(kernel=np.dot, d=2, c=1, *args, **kwargs):
    """Return a polynomial kernel

    .. math::
        K(x1, x2) = (k_0(x1, x2) + c)^d

    Args:
        d (int): degree

    Returns:
        (fun) the kernel function.
    """

    def poly(x1, x2, *args, **kwargs):
        return (kernel(x1, x2, *args, **kwargs) + c) ** d

    return poly


def rbf(kernel=np.dot, gamma=1, *args, **kwargs):
    """Return a radial basis kernel

    .. math::
        K(x1, x2) = exp(-\\frac({\\|x1, x2\\|^2}{\\gamma}))

    Returns:
        (fun) the kernel function.
    """

    def radial(x1, x2, *args, **kwargs):
        return np.exp(-(kernel(x1, x1) - 2 * kernel(x1, x2) + kernel(x2, x2)) / gamma)

    return radial
