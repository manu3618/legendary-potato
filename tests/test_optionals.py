import numpy as np
import pytest

import legendary_potato.composition
import legendary_potato.kernel
import legendary_potato.utils

from .complex_samples import KERNEL_SAMPLES


@pytest.mark.parametrize(('kernel', 'sample'), KERNEL_SAMPLES.items())
def test_kernel_matrix(kernel, sample):
    """Build kernel and test if the kernel matrix is semi definite positive.
    """
    sample = [ele for ele in sample]  # consumed several times

    potato_util = legendary_potato.utils.PotatoUtils(kernel)
    mat = potato_util.matrix(sample)
    assert (
        np.all(np.linalg.eigvals(mat) > 0)
        or np.isclose([np.min(np.linalg.eigvals(mat))], [0])
    )
