# coding: utf-8
"""
Functions to create kernels from already existing kernels.

"""

import numpy as np


def normalize(kernel, *args, **kwargs):
    """Return the normalized version.

    This correspond to the new kernel
    .. math::
        K(x_1, x_2) = \\frac{k_0(x_1, x2)}{\\sqrt(k_0(x_1, x_1) k_0(x_2, x_2))}

    This is equivalent to normalize the feature map:

    .. math::
        \\Phi(x) = \\frac{\\phi(x)}{\\|\\phi(x)\\|}

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
