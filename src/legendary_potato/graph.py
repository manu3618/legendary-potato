"""Command line tool to plot classifiers decision boundary on 2D samples.

Usage:
    $ python3 -m legendary_potato.graph [classifier]
"""
import argparse
import random
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from legendary_potato import classifiers


def generate_dataset():
    """Generate dataset.

    Returns:

        (pandas.DataFrame) 2D set with labels
    """
    random.seed("legendary")
    samp_size = 35
    X = [
        [random.normalvariate(-1, 0.5), random.normalvariate(1, 0.5)]
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
    return df


def comparision_plot(classifs=None, *args, **kwargs):
    """Plot 2D decision lines for classifiers.
    """
    if classifs is None:
        classifs = ("SVDD", "SVM")

    df = generate_dataset()

    # plot sample
    colors = {-1: "red", 1: "blue"}
    ax = df.plot(
        x="x1", y="x2", c=df["y"].apply(lambda x: colors[x]), kind="scatter"
    )

    # generate level line for each classifier
    x_spaces = np.linspace(-3, 3, 50)
    y_spaces = np.linspace(-1, 3, 50)
    xv, yv = np.meshgrid(x_spaces, y_spaces)
    levels = {"SVDD": [0.8, 1, 1.2], "SVM": [-1, 0, 1]}
    level_colors = {"SVDD": "purple", "SVM": "orange"}
    for classifier_name in classifs:

        # train classifier
        classifier = classifiers.__dict__[classifier_name](*args, **kwargs)
        classifier.fit(
            X=df[["x1", "x2"]].to_numpy(),
            y=df["y"].to_numpy(),
            *args,
            **kwargs
        )

        # generate plot
        grid_shape = xv.shape
        zv = xv.copy()
        for row, col in product(range(grid_shape[0]), range(grid_shape[1])):
            zv[row, col] = classifier.decision_function(
                [[xv[row, col], yv[row, col]]]
            )[0]

        # TODO:: specify levels
        CS = ax.contour(
            xv,
            yv,
            zv,
            levels=levels[classifier_name],
            colors=level_colors[classifier_name],
            linestyles=["--", "-", "--"],
        )
        ax.clabel(CS, inline=1)

    return df, classifier


def curve(classif=None, *args, **kwargs):
    """Plot ROC curve.

    Args:
        classif: classifier
        *args, **kwargs: extra arguments for the fit method
    """
    if classif is None:
        classif = classifiers.SVDD(*args, **kwargs)

    df = generate_dataset()

    model = classif
    model.fit(
        X=df[["x1", "x2"]].to_numpy(), y=df["y"].to_numpy(), *args, **kwargs
    )
    y_pred = 1 - model.decision_function(model.X_)
    y_true = model.y_

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(
        fpr, tpr, color="orange", label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("classifiers", nargs=-1, default=["SVM", "SVDD"])
    args = parser.parse_args()

    # comparision_plot(classifs=args.classifiers) # TODO
    df, classifier = comparision_plot()
    curve()
