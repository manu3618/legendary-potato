"Functions to create kernels from already existing kernels."

import numpy as np


def normalize(kernel, *args, **kwargs):
    """Return the normalized version.

    This correspond to the new kernel
    .. math::
    K(x_1, x_2) = \frac{kernel(x_1, x2)}\
                       {sqrt(kernel(x_1, x_1) kernel(x_2, x_2))}

    This is equivalent to normalize the feature map:
    .. math::
    \Phi(x) = \frac{\phi(x)}{\|\phi(x)\|}
    """
    return lambda x1, x2: (
        kernel(x1, x2, *args, **kwargs)
        / np.sqrt(kernel(x1, x1, *args, **kwargs)
                  * kernel(x2, x2, *args, **kwargs))
    )
