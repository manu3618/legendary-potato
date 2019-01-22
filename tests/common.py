# coding: utf-8
import os
from itertools import product

import numpy as np

import legendary_potato.kernel

TEST_PATH = os.path.dirname(__file__)
SAMPLE_PATH = os.path.join(TEST_PATH, "sample")


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


def kernel_samples(kernel_name):
    """Return iterator over samples for a kernel from specific file(s).

    The iterator generate (sample_name, sample)

    A kernel_path is generated. If it is a file, each line is considered
    to be a sample, and its name is the line number. If it is a direcctory,
    each file is considered to be a string sample and its name is the
    file name.
    """
    kernel_sample_path = os.path.join(SAMPLE_PATH, kernel_name)
    sep = ","
    if os.path.isfile(kernel_sample_path):
        # One sample per line
        with open(kernel_sample_path, "r") as sample_file:
            line = sample_file.readline()
            try:
                if len(np.fromstring(line, sep=sep)) > 0:
                    # line composed of numbers
                    is_string = False
                else:
                    # line composed of strings
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
            with open(file_path, "r") as pot:
                yield sample_file, pot.read()


def two_class_generator(random_state=1576, dims=(1, 10)):
    """Generate classic multidimensional toy set.

    For each label, a set of centers are generated following normal
    distribution. A set of samples are generated following anormal distribution
    centered on this centers.

    The generated set is a 2-class balanced dataset.

    dims -- tuple indicating smallest and highest dimension tests
    """
    np.random.seed = random_state
    for dim in range(*dims):
        orig = np.zeros(dim)
        orig[0] = -1
        a_centers = np.random.normal(orig, 2, size=(3, dim))
        orig[0] = 1
        b_centers = np.random.normal(orig, 2, size=(3, dim))
        a_data = np.vstack(
            [
                np.random.normal(center, 0.5, size=(5, dim))
                for center in a_centers
            ]
        )
        b_data = np.vstack(
            [
                np.random.normal(center, 0.5, size=(5, dim))
                for center in b_centers
            ]
        )

        labels = [1 for _ in range(15)] + [-1 for _ in range(15)]
        data = np.hstack([np.vstack([a_data, b_data]), np.transpose([labels])])
        np.random.shuffle(data)
        yield data[:, 0:-1], data[:, -1]


def multiclass_generator(
    random_state=1576, dims=(2, 3), nb_classes=(3, 5), cl_sample=5
):
    """Generate classic multiclass toyset.

    Works as the two_class_generator but alternativly with number and string
    labels.
    """
    np.random.seed = random_state
    label_list = list("abcdefghijklmnopqrstuvwxyz")
    for dim, nb_cl in product(range(*dims), range(*nb_classes)):
        centers = np.random.normal(np.zeros(dim + 1), 5, size=(nb_cl, dim + 1))
        datas = np.vstack(
            [
                np.random.normal(
                    centers[cl, :], 0.5, size=(cl_sample, dim + 1)
                )
                for cl in range(nb_cl)
            ]
        )
        datas[:, -1] = np.repeat(range(nb_cl), cl_sample)
        np.random.shuffle(datas)
        yield datas[:, 0:-1], datas[:, -1]
        yield (
            datas[:, 0:-1],
            np.array([label_list[int(i)] for i in datas[:, -1]]),
        )
