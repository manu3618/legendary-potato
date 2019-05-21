# coding: utf-8
import os
from contextlib import suppress
from itertools import product

import numpy as np
import pandas as pd
import pytest
import yaml

import legendary_potato.composition
import legendary_potato.kernel
from legendary_potato.methods import KernelMethod

from .common import TEST_PATH, kernel_sample_iterator, two_class_generator

GRAMMATRIX_PATH = os.path.join(TEST_PATH, "gram_matrix")
COMPOSITION_FILE = os.path.join(TEST_PATH, "composition.yaml")


def composition_iterator(filename=COMPOSITION_FILE):
    """Return an iterator over possible kernel composition from file.
    """
    with open(filename, "r") as fd:
        ret = list(yaml.load_all(fd))
    return ret[0].items()


@pytest.mark.parametrize(("kernel", "sample"), kernel_sample_iterator())
def test_matrix(kernel, sample):
    """Regression test on gram matrix.

    Construct the Gram matrix for the kernel and the samples and compare it
    to the previously calculated one.

    kernel -- the potato kernel to test
    sample -- the sample to construct the Gram matrix
    """
    kernel_name = kernel.__name__  # TODO: find a more feng shui way
    matrix_path = os.path.join(GRAMMATRIX_PATH, kernel_name + ".csv")
    potato_util = KernelMethod(kernel)
    cur_matrix = potato_util.matrix(tr_s for _, tr_s in sample)
    if os.path.exists(matrix_path):
        test_matrix = pd.read_csv(matrix_path, header=None, index_col=False)
        np.testing.assert_allclose(
            np.array(test_matrix, dtype=cur_matrix.dtype), cur_matrix
        )

    else:
        with suppress(FileExistsError):
            os.makedirs(GRAMMATRIX_PATH)
        pd.DataFrame(cur_matrix).to_csv(matrix_path, header=None, index=None)


@pytest.mark.parametrize(("kernel", "sample"), kernel_sample_iterator())
def test_matrix_properties(kernel, sample):
    """Test some properties.
    """
    util = KernelMethod(kernel)
    sample = [value for nu, value in sample]
    gram = util.matrix(sample)
    dist = util.distance_matrix()
    dim = len(sample)

    # symmetry
    for mat in gram, dist:
        for m, n in product(range(dim), repeat=2):
            if np.isnan(mat[m, n]):
                mat[m, n] = 0

        assert np.all(np.isclose(mat - mat.transpose(), 0))

    # distances are distances
    assert np.all(dist >= 0)
    for m, n, p in product(range(dim), repeat=3):
        assert any(
            [
                np.isclose(
                    dist[m, p] + dist[p, n] - dist[m, n], 0, atol=2e-08
                ),
                dist[m, n] <= dist[m, p] + dist[p, n],
            ]
        )
    assert np.all(np.isclose(np.diag(dist), 0))
    for m, n in product(range(dim), repeat=2):
        if np.isclose(dist[m, n], 0):
            # TODO: check equality method for samples
            # assert sample[m] == sample[n]
            pass


def test_empty_matrix():
    """matrix() raises expected errors.
    """
    potato_util = KernelMethod(None)
    with pytest.raises(RuntimeError):
        potato_util.matrix()
    with pytest.raises(ValueError):
        potato_util.matrix([])


@pytest.mark.parametrize(("kernel", "sample"), kernel_sample_iterator())
@pytest.mark.parametrize(("composition", "args"), composition_iterator())
def test_composition(kernel, sample, composition, args):
    """Build kernel composition and test it on the sample file and test matix.

    Test the matrix is a kernel matrix.
    """
    sample = [ele for ele in sample]  # consumed several times
    compo = legendary_potato.composition.__dict__.get(composition)

    if args:
        for arg_set in args:
            new_kern = compo(kernel, arg_set)
            potato_util = KernelMethod(new_kern)
            mat = potato_util.matrix(tr_s for _, tr_s in sample)
    else:
        new_kern = compo(kernel)
        potato_util = KernelMethod(new_kern)
        mat = potato_util.matrix(tr_s for _, tr_s in sample)

    assert np.all(np.linalg.eigvals(mat) > 0) or np.isclose(
        [np.min(np.linalg.eigvals(mat))], [0]
    )


@pytest.mark.parametrize("sample", two_class_generator())
def test_orth_base(sample):
    """Check orthonormality of built base.
    """
    potato = KernelMethod()
    base = potato.orthonormal(list(sample[0]))
    for i, j in product(range(len(base)), repeat=2):
        dot = potato.kernel(base[i], base[j])
        if i == j:
            assert np.isclose(dot, 1)
        else:
            assert np.isclose(dot, 0)


@pytest.mark.parametrize("sample", two_class_generator())
def test_proj_run(sample):
    """Run projection
    """
    potato = KernelMethod()
    base = potato.orthonormal(list(sample[0]))
    for i in range(len(base)):
        coords = potato.projection([base[i]], base)
        assert not np.isclose(coords[0][i], 0)
