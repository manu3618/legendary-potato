"""Command line tool to plot classifiers decision boundary on 2D samples.

Usage:
    $ python3 -m legendary_potato.graph [classifier]
"""
import argparse
import random

import numpy as np
import pandas as pd

from legendary_potato.classifiers import SVDD, SVM


def comparision_plot(classifiers=None):
    """Plot 2D decision lines for classifiers.
    """
    if classifiers is None:
        classifiers = ("SVDD",)

    # generate sample
    random.seed("legendary")
    samp_size = 100
    X = [
        [random.normalvariate(-1, 1), random.normalvariate(1, 1)]
        for _ in range(samp_size)
    ]
    X.extend(
        [
            [random.uniform(-1, 2), random.uniform(0, 1)]
            for _ in range(samp_size)
        ]
    )
    X = np.array(X).transpose()
    Y = [1] * samp_size
    Y.extend([-1] * samp_size)
    df = pd.DataFrame({"x1": X[0], "x2": X[1], "y": Y})

    # plot sample
    fig, ax = df.plot(x="x1", y="x2", c="y", kind="scatter")

    # generate level line for each classifier
    spaces = np.linspace(-2, 2, 20)
    xv, yv = np.meshgrid(spaces, spaces)
    levels = {"SVDD": [0.8, 1, 1.2], "SVM": [-0.5, 0, 0.5]}
    for classifier in classifiers:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("classifiers", nargs=-1, default=["SVM", "SVDD"])
    args = parser.parse_args()

    comparision_plot(classifiers=args.classifiers)
