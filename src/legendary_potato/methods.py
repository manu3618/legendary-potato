# coding: utf-8
"""Utils for kernel methods."""

from itertools import product
from inspect import isgenerator

import numpy as np


class KernelMethod:
    """Kernel utils.

    Specific to one kernel along with it taining sample.

    """

    def __init__(self, kernel=np.dot, sample=None, kernel_matrix=None):
        """
        Args:
            kernel (fun): the kernel function
            sample (list): list or tuple of sample
            kernel_matrix (np.array): 2D symmetric array.

        Either the kernel of the kernel_matrix should be provided.

        the kernel must be callable by
        .. code-block::
            kernel(sample[i], sample[j])

        """
        self.kernel = kernel
        self.kernel_matrix = kernel_matrix
        self.sample = None
        self._check_sample_arg(sample)

    def _check_sample_arg(self, sample=None):
        """Check if sample is a generator and update self.sample if needed.

        Returns:
            (list) sample
        """
        if isgenerator(sample):
            sample = list(sample)
        if self.sample is None:
            self.sample = sample
        if sample is None:
            sample = self.sample
        return sample

    def matrix(self, sample=None, ix=None):
        """Return the kernel matrix.

        Args:
            sample: the sample to build the kernel matrix. If None, the sample
                from self are used. If self.sample is None, sample replace it.
            ix: the indices of the matrice to be return. If None, all
                the matrice is returned as a numpy.ndarray. It must not be a
                generator as it will be consumed many times.

        """
        update_matrix = False  # Whether to update self.matrix
        if sample is None and self.sample is None:
            raise RuntimeError("No sample to build the matrix.")
        if sample is not None and self.sample is None:
            # consume iterator/generator or copy input
            self.sample = tuple(tr_s for tr_s in sample)
            update_matrix = True
            sample = self.sample
        if sample is None:
            sample = self.sample

        dim = len(sample)
        if dim == 0:
            raise ValueError("Zero sample provided.")

        if ix is None:
            ix = (list(range(dim)), list(range(dim)))
            if self.kernel_matrix is not None and not update_matrix:
                return self.kernel_matrix
            update_matrix = True

        kernel_matrix = np.array(
            [[self.kernel(sample[i], sample[j]) for i in ix[0]] for j in ix[1]]
        )
        if update_matrix:
            self.kernel_matrix = kernel_matrix

        return kernel_matrix

    def _square_dist(self, s0, s1):
        """Used by distance.
        """
        return (
            self.kernel(s0, s0) + self.kernel(s1, s1) - 2 * self.kernel(s0, s1)
        )

    def distance(self, sample0=None, sample1=None):
        """Compute squarenorm.

        If sample1 and sample0 are not None, then the distance between sample0
        and sample1 is returned.
        If sample1 is not None and sample0 is None, the list of distances
        between sample1 and all self.sample  are returned
        If sample1 is None, the distance matrix for self.sample is returned. If
        sample 0 is not None, self.sample is replaced by sample0.

        The distance is computed as

        .. math::
            d(x_1, x_2) = \|x_1 - x_2 \|

        Returns:
            (np.array) distance array, 1x1 array if a ingle scalar is expected.

        """
        if sample0 is not None and sample1 is not None:
            return np.array(np.sqrt(self._square_dist(sample0, sample1)))
        elif sample0 is None and sample1 is not None:
            return np.array(
                np.sqrt(self._square_dist(ele, sample1))
                for ele in self._sample
            )
        elif sample1 is None:
            if sample0 is not None:
                self.sample = list(sample0)
            sample = self.sample
            dim = len(sample)
            return np.array(
                [
                    [
                        np.sqrt(self._square_dist(sample[i], sample[j]))
                        for i in range(dim)
                    ]
                    for j in range(dim)
                ]
            )

    def distance_matrix(self, sample=None):
        """Return distance matrix between samples based on the kernel matrix.

        Args:
            sample: the sample to build the kernel matrix. If None, the sample
                from self are used. If self.sample is None, sample replace it.

        """
        gram_mat = self.matrix(sample)
        mshape = gram_mat.shape
        dist_mat = gram_mat.copy()
        for (s, t) in product(range(mshape[0]), range(mshape[1])):
            dist_mat[s, t] = np.sqrt(
                gram_mat[s, s] - 2 * gram_mat[s, t] + gram_mat[t, t]
            )
        return dist_mat

    def _cosine(self, s1, s2):
        """Used by cosine.

        """
        num = self.kernel(s1, s2)
        denom = np.sqrt(self.kernel(s1, s1)) * np.sqrt(self.kernel(s2, s2))
        if num == 0:
            return 0
        else:
            return num / denom

    def cosine(self, sample0=None, sample1=None):
        """Return the cosine between 2 samples.

        If sample0 is None, samples from self are used.
        If sample1 is not None, then the cosine between sample0 and sample1 is
        returned.
        Otherwise, sample0 is considered as an iterable and all cosines are
        returned.

        The cosine is defined as

        .. math::
            cos(sample_0, sample_1) =
            \\frac{\langle sample_0, sample_1 \\rangle}
            {\|sample_0\| \|sample_1\|}

        """
        if sample0 is not None and sample1 is not None:
            return self._cosine(sample0, sample1)
        elif sample0 is None and sample1 is not None:
            return np.array(np._cosine(ele, sample1) for ele in self._sample)
        elif sample1 is None:
            sample = sample0 or self.sample
            dim = len(sample)
            return np.array(
                [
                    [self._cosine(sample[i], sample[j]) for i in range(dim)]
                    for j in range(dim)
                ]
            )

    def orthogonal(self, sample=None):
        """Return the orthongonal base from samples.

        For each sample :math:`s_{n}`, add to the list
        :math:`v_{n} = s_{n} - \sum_{i<n} \langle s_{n}, v_i \\rangle v_i`

        Returns:
            (list) vectors :math:`\{v_i\}`.
        """
        sample = self._check_sample_arg(sample)
        if sample is None:
            raise ValueError("No valid sample to build an orthogonal base.")

        base_vect = [np.array(sample[0])]
        for n in range(1, len(sample)):
            sample_proj = self._projection(sample[n], base_vect)
            new_vect = sample[n] - sample_proj
            newvect_proj = self._projection(new_vect, base_vect)

            if all(
                [
                    np.isclose(abs(self._cosine(new_vect, newvect_proj)), 1),
                    not np.isclose(self.kernel(newvect_proj, newvect_proj), 0),
                ]
            ):
                # new vector colinear with its projection (not free)
                continue

            if any(
                [
                    np.any(np.isnan(new_vect)),
                    np.isclose(self.kernel(new_vect, new_vect), 0),
                ]
            ):
                # new vector null or invalid
                continue

            base_vect.append(new_vect)

        return base_vect

    def orthonormal(self, sample=None):
        """Return the orthongonal base from samples.

        For each sample :math:`s_{n}`, compute
        :math:`v_{n} = s_{n} - \sum_{i<n} \langle s_{n}, v_i \\rangle v_i`
        and add to the list :math:`u_{n} = \\frac{v_{n}}{\|v_{n}\|}`

        Returns:
            (list) vectors :math:`\{u_i\}`.

        """
        return [
            np.sqrt(1 / self.kernel(v, v)) * np.array(v)
            for v in self.orthogonal(sample)
        ]

    def fourier_serie(self, sample, base=None):
        """Decompose  the sample on its fourier serie.

        The Fourier serie is defined  as the projection onto an orthonormal
        base. If none is provided, the base used is the one from
        self.orthonormal.

        No check on the sabe orthonormality is performed.
        """
        if base is None:
            base = self.orthonormal(sample)
        # TODO
        raise NotImplementedError

    def projection(self, samples=None, base=None):
        """Perform projection of sample onto basis

        This is mainly done for better representation (lower dimensions).

        Args:
            samples (list of samples): list of sample to project. If None
                use self.sample
            base (list of vectors): base to expres the projection. If None,
                build a base from the first self.sample

        Returns:
            The projection coordinates

        """
        samples = self._check_sample_arg(samples)
        if base is None:
            base = self.orthonormal(samples)[0, 1]

        return [[self.kernel(sample, u) for u in base] for sample in samples]

    def _projection(self, sample, base):
        """Project one sample onto base.

        Args:
            sample: sample to project
            base (iterable): base: list of free vectors

        Returns:
            the projection.
        """
        coords = [self.kernel(sample, v) / self.kernel(v, v) for v in base]
        iter_sum = coords[0] * base[0]
        for i in range(1, len(base)):
            iter_sum = iter_sum + coords[i] * base[i]
        return iter_sum
