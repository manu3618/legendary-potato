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


@pytest.mark.parametrize("classifier", classifier_iterator())
@pytest.mark.parametrize("dataset", two_class_generator())
def test_oneclass(classifier, dataset):
    """Perform one class classification.
    """
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    y_train = np.ones(X_train.shape[0])
    classif = classifier()
    classif.fit(X_train, y_train)
    classif.predict(X_test)


@pytest.mark.parametrize("classifier", classifier_iterator())
@pytest.mark.parametrize("dataset", two_class_generator())
def test_twoclasses(classifier, dataset):
    """Perform one class classification.
    """
    X, y = dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    classif = classifier()
    classif.fit(X_train, y_train)
    y_pred = classif.predict(X_test)
    confusion_matrix(y_test, y_pred)


@pytest.mark.parametrize("classifier", classifier_iterator())
def test_sklearn_compatibility(classifier):
    """Check the compatibility.
    """
    check_estimator(classifier)
