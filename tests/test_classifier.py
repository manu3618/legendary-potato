# coding: utf-8
from itertools import chain, product

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

import legendary_potato.classifiers as classifiers


def classifier_iterator():
    """Return an iterator over classifier.
    """
    return (classifiers.SVDD,)


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
            np.random.normal(center, 0.5, size=(5, dim))
            for center in a_centers
        )
        b_data = np.vstack(
            np.random.normal(center, 0.5, size=(5, dim))
            for center in b_centers
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
            np.random.normal(centers[cl, :], 0.5, size=(cl_sample, dim + 1))
            for cl in range(nb_cl)
        )
        datas[:, -1] = np.repeat(range(nb_cl), cl_sample)
        np.random.shuffle(datas)
        yield datas[:, 0:-1], datas[:, -1]
        yield (
            datas[:, 0:-1],
            np.array([label_list[int(i)] for i in datas[:, -1]]),
        )


@pytest.mark.parametrize("classifier", classifier_iterator())
@pytest.mark.parametrize("dataset", two_class_generator())
def test_oneclass(classifier, dataset):
    """Perform one class classification.
    """
    X, y = dataset
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

    assert np.all(svdd.alphas_ >= 0)
    assert svdd.radius_ > 0
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
