import os

import numpy as np

import legendary_potato.kernel

TEST_PATH = os.path.join(os.path.abspath(os.path.curdir))
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
            with open(file_path, "r") as pot:
                yield sample_file, pot.read()
