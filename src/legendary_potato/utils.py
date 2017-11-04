"""Utils"""

import numpy as np


class PotatoUtils():
    """Kernel utils."""

    def __init__(self, potato, sample=None):
        """
        potato -- the kernel
        sample -- a list or tuple of sample

        the kernel must be callable by:
        >>> potato(sample[i], sample[j])
        """
        self.potato = potato
        if sample is not None:
            self.sample = sample
        else:
            self.sample = {}

    def matrix(self, sample=None, ix=None):
        """Return the kernel matrix.

        sample -- the sample to build the kernel matrix. If None, the sample
            from self are used. If self.sample is None, sample replace it.
        ix -- the indices of the matrice to be return. If None, all the matrice
            is returned as a numpy.ndarray. It must not be a generator as it
            will be consumed many times.
        """
        if sample is None and self.sample is None:
            raise RuntimeError("No sample to build the matrix")
        if sample is not None and self.sample is None:
            self.sample = sample.copy()

        sample = self.sample
        dim = len(sample)
        if ix is None:
            ix = (list(range(dim)), list(range(dim)))

        return np.array([[self.potato(sample[i], sample[j]) for i in ix[0]]
                         for j in ix[1]])
