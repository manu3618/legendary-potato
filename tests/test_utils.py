import os
from contextlib import suppress

import numpy as np
import pandas as pd
import pytest
import yaml

import legendary_potato.composition
import legendary_potato.kernel
import legendary_potato.utils

TEST_PATH = os.path.join(os.path.abspath(os.path.curdir))
SAMPLE_PATH = os.path.join(TEST_PATH, 'sample')
GRAMMATRIX_PATH = os.path.join(TEST_PATH, 'gram_matrix')
COMPOSITION_FILE = os.path.join(TEST_PATH, 'composition.yaml')


def kernel_sample_iterator():
    """Return an iterator over (kernels, samples).
    """
    for kern in kernel_iterator():
        kern_name = kern.__name__
        yield kern, kernel_samples(kern_name)


def kernel_iterator():
    """Return an iterator over kernels to be tested.
    """
    for kern_name in os.listdir(SAMPLE_PATH):
        yield legendary_potato.kernel.__dict__.get(kern_name)


def composition_iterator(filename=COMPOSITION_FILE):
    """Return an iterator over possible kernel composition from file.
    """
    with open(filename, 'r') as fd:
        ret = list(yaml.load_all(fd))
    return ret[0].items()


def kernel_samples(kernel_name):
    """Return iterator over samples for a kernel from specific file(s).

    The iterator generate (sample_name, sample)

    A kernel_path is generated. If it is a file, each line is considered
    to be a sample, and its name is the line number. If it is a direcctory,
    each file is considered to be a string sample and its name is the
    file name.
    """
    kernel_sample_path = os.path.join(SAMPLE_PATH, kernel_name)
    sep = ','
    if os.path.isfile(kernel_sample_path):
        # One sample per line
        with open(kernel_sample_path, 'r') as sample_file:
            line = sample_file.readline()
            try:
                if len(np.fromstring(line, sep=sep)) > 0:
                    # line composed of numbers
                    is_string = False
                else:
                    # line composed of stings
                    is_string = True
            except ValueError:
                # line composed of mix of strings and numbers. should be
                # treated as strings
                is_string = True
            sample_file.seek(0)

            for nu, line in enumerate(sample_file):
                if is_string:
                    yield (nu, [row.strip for row in line.split(sep)])
                else:
                    yield (nu, np.fromstring(line, sep=sep))
    else:
        # kernel_sample_path is a directory
        for sample_file in os.listdir(kernel_sample_path):
            file_path = os.path.join(kernel_sample_path, sample_file)
            with open(file_path, 'r') as pot:
                yield sample_file, pot.read()


@pytest.mark.parametrize(('kernel', 'sample'), kernel_sample_iterator())
def test_matrix(kernel, sample):
    """Regression test on gram matrix.

    Construct the Gram matrix for the kernel and the samples and compare it
    to the previously calculated one.

    kernel -- the potato kernel to test
    sample -- the sample to construct the Gram matrix
    """
    kernel_name = kernel.__name__  # TODO: find a more feng shui way
    matrix_path = os.path.join(GRAMMATRIX_PATH, kernel_name + '.csv')
    potato_util = legendary_potato.utils.PotatoUtils(kernel)
    cur_matrix = potato_util.matrix(tr_s for _, tr_s in sample)
    if os.path.exists(matrix_path):
        test_matrix = pd.read_csv(matrix_path, header=None, index_col=False)
        np.testing.assert_allclose(
            np.array(test_matrix, dtype=cur_matrix.dtype),
            cur_matrix
        )

    else:
        with suppress(FileExistsError):
            os.makedirs(GRAMMATRIX_PATH)
        pd.DataFrame(cur_matrix).to_csv(matrix_path, header=None, index=None)


def test_empty_matrix():
    """matrix() raises expected errors.
    """
    potato_util = legendary_potato.utils.PotatoUtils(None)
    with pytest.raises(RuntimeError):
        potato_util.matrix()
    with pytest.raises(ValueError):
        potato_util.matrix([])


@pytest.mark.parametrize(('kernel', 'sample'), kernel_sample_iterator())
@pytest.mark.parametrize(('composition', 'args'), composition_iterator())
def test_composition(kernel, sample, composition, args):
    """Build kernel composition and test it on the sample file and test matix.

    Test the matrix is a kernel matrix.
    """
    sample = [ele for ele in sample]  # consumed several times
    compo = legendary_potato.composition.__dict__.get(composition)

    if args:
        for arg_set in args:
            new_kern = compo(kernel, arg_set)
            potato_util = legendary_potato.utils.PotatoUtils(new_kern)
            mat = potato_util.matrix(tr_s for _, tr_s in sample)
    else:
        new_kern = compo(kernel)
        potato_util = legendary_potato.utils.PotatoUtils(new_kern)
        mat = potato_util.matrix(tr_s for _, tr_s in sample)

    assert (
        np.all(np.linalg.eigvals(mat) > 0)
        or np.isclose([np.min(np.linalg.eigvals(mat))], [0])
    )
