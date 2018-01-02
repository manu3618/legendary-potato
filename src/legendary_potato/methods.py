"""Utils"""

import numpy as np


class KernelMethod():
    """Kernel utils."""

    def __init__(self, kernel, sample=None):
        """
        kernel -- the kernel
        sample -- a list or tuple of sample

        the kernel must be callable by:
        >>> kernel(sample[i], sample[j])
        """
        self.kernel = kernel
        self.sample = sample

    def matrix(self, sample=None, ix=None):
        """Return the kernel matrix.

        sample -- the sample to build the kernel matrix. If None, the sample
            from self are used. If self.sample is None, sample replace it.
        ix -- the indices of the matrice to be return. If None, all the matrice
            is returned as a numpy.ndarray. It must not be a generator as it
            will be consumed many times.
        """
        if sample is None and self.sample is None:
            raise RuntimeError("No sample to build the matrix.")
        if sample is not None and self.sample is None:
            # consume iterator/generator or copy input
            self.sample = tuple(tr_s for tr_s in sample)

        sample = self.sample
        dim = len(sample)
        if dim == 0:
            raise ValueError("Zero sample provided.")

        if ix is None:
            ix = (list(range(dim)), list(range(dim)))

        return np.array([[self.kernel(sample[i], sample[j]) for i in ix[0]]
                         for j in ix[1]])

    def _square_dist(self, s0, s1):
        """Used by distance.
        """
        return (self.kernel(s0, s0)
                + self.kernel(s1, s1)
                - 2 * self.kernel(s0, s1))

    def distance(self, sample0=None, sample1=None):
        """Compute squarenorm.

        If sample1 and sample0 are not None, then the distance between sample0
        and sample1 is returned.
        If sample1 is not None and sample0 is None, the list of distances
        between sample1 and all self.sample  are returned
        If sample1 is None, the disante matrix for self.sample is returned. If
        sample 0 is not None, self.sample is replaced by sample0.

        The distance is computed as
        .. math::
            d(x_1, x_2) = \|x_1 - x_2 \|

        """
        if sample0 is not None and sample1 is not None:
            return np.sqrt(self._square_dist(sample0, sample1))
        elif sample0 is None and sample1 is not None:
            return np.array(np.sqrt(self._square_dist(ele, sample1))
                            for ele in self._sample)
        elif sample1 is None:
            if sample0 is not None:
                self.sample = list(sample0)
            sample = self.sample
            dim = len(sample)
            return np.array([[np.sqrt(self._square_dist(sample[i], sample[j]))
                              for i in range(dim)]
                             for j in range(dim)])

    def _cosine(self, s1, s2):
        """used by cosine"""
        num = self.kernel(s1, s2)
        denom = np.sqrt(self.kernel(s1, s1)) * np.sqrt(self.kernel(s2, s2))
        if num == 0:
            return 0
        else:
            return num/denom

    def cosine(self, sample0=None, sample1=None):
        """Return the cosine between 2 samples

        If sample0 is None, samples from self are used.
        If sample1 is not None, then the cosine between sample0 and sample1 is
        returned.
        Otherwise, sample0 is considered as an iterable and all cosines are
        returned.

        The cosine is defined as
        .. math::
            cos(sample_0, sample_1) = \frac{\langle sample_0, sample_1 \rangle}
                                           {\|sample_0\| \|sample_1\|}

        """
        if sample0 is not None and sample1 is not None:
            return self._cosine(sample0, sample1)
        elif sample0 is None and sample1 is not None:
            return np.array(np._cosine(ele, sample1) for ele in self._sample)
        elif sample1 is None:
            sample = sample0 or self.sample
            dim = len(sample)
            return np.array([[self._cosine(sample[i], sample[j])
                              for i in range(dim)]
                             for j in range(dim)])

    def orthonormal(self, sample=None):
        """Return the orthonormal base from samples.

        Return a list of vector ${v_i}$. For each sample s_{n}, add to the
        list $v_{n} = s_{n} - \sum_{i<n} \langle s, v_i \rangle v_i$
        """
        if self.sample is None:
            self.sample = sample
        if sample is None:
            sample = self.sample
        if sample is None:
            raise ValueError("No valid sample to build an orthogonal base.")
        # TODO

    def fourier_serie(self, sample,  base=None):
        """Decompose  the sample on its fourier serie.

        The Fourier serie is defined  as the projection onto an orthonormal
        base. If none is provided, the base used is the one from
        self.orthonormal.
        """
        if base is None:
            base = self.orthonormal()
        # TODO
