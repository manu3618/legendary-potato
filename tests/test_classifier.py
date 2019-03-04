# coding: utf-8
from itertools import chain

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

import legendary_potato.classifiers as classifiers

from .common import multiclass_generator, two_class_generator


def classifier_iterator():
    """Return an iterator over classifier.
    """
    return (classifiers.SVDD,)


@pytest.mark.parametrize("classifier", classifier_iterator())
@pytest.mark.parametrize("dataset", two_class_generator())
def test_oneclass(classifier, dataset):
    """Perform one class classification.
    """
    X, y = dataset
    if isinstance(X, list):
        X = np.array(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    y_train = np.ones(X_train.shape[0])
    classif = classifier()
    try:
        classif.fit(X_train, y_train)
    except RuntimeError as exn:
        pytest.skip("fit method did not work: %s" % exn)
    classif.predict(X_test)


@pytest.mark.parametrize("classifier", classifier_iterator())
@pytest.mark.parametrize("dataset", two_class_generator())
def test_twoclasses(classifier, dataset):
    """Perform one class classification.
    """
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    classif = classifier()
    try:
        classif.fit(X_train, y_train)
    except RuntimeError as exn:
        pytest.skip("fit method did not work: %s" % exn)
    y_pred = classif.predict(X_test)
    confusion_matrix(y_test, y_pred)


@pytest.mark.parametrize("classifier", classifier_iterator())
@pytest.mark.parametrize("dataset", multiclass_generator())
def test_multiclass(classifier, dataset):
    """Perform one class classification.
    """
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    classif = classifier()
    try:
        classif.fit(X_train, y_train)
    except RuntimeError as exn:
        pytest.skip("fit method did not work: %s" % exn)
    y_pred = classif.predict(X_test)
    confusion_matrix(y_test, y_pred)


@pytest.mark.parametrize(
    "dataset", chain(two_class_generator(), multiclass_generator())
)
def test_svdd(dataset):
    """Test SVDD specificities.
    """
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    svdd = classifiers.SVDD()
    try:
        svdd.fit(X_train, y_train)
    except RuntimeError as exn:
        pytest.skip("optimization did not work: %s" % exn)

    if svdd.hypersphere_nb == 1:
        assert np.all(svdd.alphas_ >= 0)
        assert svdd.radius_ > 0
    else:
        for sub_svdd in svdd.individual_svdd.values():
            assert np.all(sub_svdd.alphas_ >= 0)
            assert sub_svdd.radius_ > 0

    assert np.all(
        svdd.dist_center_training_sample(r) >= 0 for r in range(len(X_train))
    )

    for X in X_train, X_test, None:
        assert np.all(svdd.dist_all_centers(X) >= 0)
        assert np.all(svdd.relative_dist_all_centers(X) >= 0)


@pytest.mark.skip(reason="no 2D-array input")
@pytest.mark.parametrize("classifier", classifier_iterator())
def test_sklearn_compatibility(classifier):
    """Check the compatibility.
    """
    check_estimator(classifier)
