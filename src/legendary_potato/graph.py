import random

import numpy as np
import pandas as pd

from legenray_potat.classifiers import SVDD, SVM


def comparision_plot():
    """Plot 2D decision lines for classifiers.
    """
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
    for classifier in ("SVDD", "SVM"):
        pass


if __name__ == "__main__":
    comparision_plot()
