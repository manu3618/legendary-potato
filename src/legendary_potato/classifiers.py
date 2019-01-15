# coding: utf-8
"""
Classifiers.
"""
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y

from .methods import KernelMethod


class SVDD(BaseEstimator, ClassifierMixin, KernelMethod):
    """Implement Support Vector DataDescription

    .. math::
        min_{r, c}  r^2 - C \sum_t  xi_t
        s.t.        y_i \| \phi(x_i) -c \| < r^2 + xi_i \forall i
                    xi_i > 0                            \forall i
    """

    def __init__(self, kernel_matrix=None, kernel=None, C=None):
        """Initialize some parameters.

        Those parameters may be overwritten by the fit() method.
        """
        self.kernel_matrix = kernel_matrix  # kernel matrix used for training
        if kernel is None:
            self.kernel = np.dot
        else:
            self.kernel = kernel
        self.C = C
        self.string_labels = False  # are labels strings or int?
        self.hypersphere_nb = 1

    def fit(
        self,
        X,
        y,
        C=None,
        kernel=None,
        is_kernel_matrix=False,
        *args,
        **kwargs
    ):
        """Fit the classifier.

        C -- contraint in the soft margin case. If None or zero, then fall back
        to hard margin case
        kernel -- kernel method to use
        is_kernel_matrix -- if true, the input is  treated as a kernel matrix
        *args, **kwargs -- extra arguments for kernel function
        """
        # X, y = check_X_y(X, y) # TODO: add check method for X
        n = len(y)
        _, y = check_X_y(np.identity(n), y)
        self.X_ = X
        if C is not None:
            self.C = C
        self.support_vectors_ = set()
        if is_kernel_matrix:
            self.kernel_matrix = X
        else:
            if self.kernel is None:
                raise ValueError(
                    "You must provide either a kernel function "
                    "or a kernel matrix."
                )
            self.sample = self.X_
            self.kernel_matrix = self.matrix()
        self.classes_ = np.unique(y)

        if np.isreal(y[0]):
            self.string_labels = False

        if len(self.classes_) > 2 or (
            len(self.classes_) == 2 and self.string_labels
        ):
            # each class has its own hypersphere (one class vs rest)
            self.hypersphere_nb = len(self.classes_)
            self.individual_svdd = {}
            for cl in self.classes_:
                # TODO: multithread/asyncio
                cl_svdd = SVDD(
                    kernel_matrix=self.kernel_matrix,
                    kernel=self.kernel,
                    C=self.C,
                )
                cl_y = [1 if elt == cl else -1 for elt in y]
                cl_svdd.fit(X, cl_y, C, kernel, is_kernel_matrix)
                self.individual_svdd[cl] = cl_svdd
            self.y_ = y
            self.alphas_ = np.array([0])
            self.radius_ = 0
        else:
            # one hypersphere
            self.y_ = np.sign(y)
            self.radius_, self.alphas_ = self._fit_one_hypersphere()
        return self

    def predict(self, X, decision_radius=1):
        """Predict classes

        Args:
            X (array like): list of test samples.
            decision_radius (numeric): modification of decision radius.
        The frontier between classes will be the computed hypersphere whose
        radius is multiply by this factor.
        """
        check_is_fitted(self, ["X_", "alphas_"])
        # X = check_array(X)

        if self.hypersphere_nb == 1:
            return self._predict_one_hypersphere(X, decision_radius)

        else:
            # check class
            dist_classes = self.relative_dist_all_centers(X)
            return np.array(dist_classes.idxmin(axis=1))

    def fit_predict(
        self,
        X,
        y,
        C=None,
        kernel=None,
        is_kernel_matrix=False,
        *args,
        **kwargs
    ):
        """Fit as the fit() methods.

        Return:
            (array) : class for each training sample.
        """
        self.fit(X, y, C, kernel, is_kernel_matrix, *args, **kwargs)
        # TODO
        raise NotImplementedError

    def _predict_one_hypersphere(self, X=None, decision_radius=1):
        """Compute results for one hypersphere

        Args:
            decision_radius (numeric): modification of decision radius.
        The frontier between classes will be the computed hypersphere whose
        radius is multiply by this factor.

        Returns:
            (np.array)
        """
        pred = self._dist_center(X) * decision_radius / self.radius_ - 1
        return np.sign(pred).reshape(-1)

    def _dist_center(self, X=None):
        """Compute ditance to class center.

        Args:
            X (array-like): list of input vectors. If None, use the train set.

        Distance to center:
        .. math::
            \| z - c \|^2 = \|z\|^2 - 2 K(z, c) + \|c\|^2

            c = \sum_t  \alpha_t \phi(X_t)
        """
        if not self.hypersphere_nb == 1:
            raise RuntimeWarning("Not available for multiclass SVDD")

        check_is_fitted(self, ["X_", "alphas_"])
        dim = len(self.alphas_)
        if X is None:
            # return distances for training set
            square_dists = [
                np.power(self.kernel_matrix[i, i], 2)
                - 2
                * sum(
                    self.alphas_[t] * self.kernel_matrix[i, t]
                    for t in range(dim)
                )
                + np.power(
                    sum(
                        self.alphas_[t]
                        * self.alphas_[s]
                        * self.kernel_matrix[s, t]
                        for s, t in product(range(dim), range(dim))
                    ),
                    2,
                )
                for i in range(dim)
            ]
        else:
            # return distances for vector X
            square_dists = [
                np.power(self.kernel(z, z), 2)
                - 2
                * sum(
                    self.alphas_[t] * self.kernel(self.X_[t], z)
                    for t in range(dim)
                )
                + np.power(
                    sum(
                        self.alphas_[s]
                        * self.alphas_[t]
                        * self.kernel(self.X_[t], self.X_[s])
                        for s, t in product(range(dim), range(dim))
                    ),
                    2,
                )
                for z in X
            ]

        return np.sqrt(square_dists)

    def _fit_one_hypersphere(self, y=None, class1=1, class2=-1):
        """Perform actual fit process

        * compute alphas
        * compute support vectors
        * recompute minimal kernel matrix
        """
        if y is None:
            y = self.y_
        dim = len(self.X_)
        alphas = [0 for _ in range(dim)]
        C = self.C

        def ell_d(al):
            """Dual function to minimize.

            function to maximize:
            .. maths::
                \alpha diag(K)^T - \alpha K \alpha^T
            """
            ay = al * y
            return ay.dot(self.kernel_matrix).dot(ay.T) - ay.dot(
                np.diag(self.kernel_matrix)
            )

        # \forall i \alpha[i] \geq 0 \leftrightdoublearrow min(\alpha) \geq 0
        cons = [
            {"type": "eq", "fun": lambda al: np.sum(al) - 1},
            {"type": "ineq", "fun": lambda al: np.min(al)},
        ]
        if C and C is not np.inf:
            # soft margin case: \forall i \alpha[i] \leq C
            cons.append(
                {"type": "ineq", "fun": lambda alphas: C - np.max(alphas)}
            )
        else:
            C = np.inf

        # TODO: asyncio
        predicted_alphas = minimize(
            ell_d, alphas, constraints=tuple(cons), options={"maxiter": 100000}
        )
        if not predicted_alphas.success:
            raise RuntimeError(predicted_alphas.message)
        alphas = predicted_alphas.x
        support_vectors = set(
            np.where(i < C and not np.isclose(i, 0) for i in alphas)[0]
        )
        self.support_vectors_ = self.support_vectors_.union(support_vectors)

        # mean distance to support vectors
        radius = np.mean(
            [
                self.dist_center_training_sample(r, alphas)
                for r in range(dim)
                if alphas[r] < C and not np.isclose(0, alphas[r])
            ]
        )
        return radius, alphas

    def dist_all_centers(self, X=None):
        """Return distance to each class center.
        """
        if self.hypersphere_nb > 1:
            dist_classes = {
                cl: svdd._dist_center(X)
                for cl, svdd in self.individual_svdd.items()
            }
        else:
            dist_classes = {1: self._dist_center(X)}
        return pd.DataFrame(dist_classes)

    def relative_dist_all_centers(self, X=None):
        """Distane to all centers divided by class radius.
        """
        if self.hypersphere_nb > 1:
            dist_classes = {
                cl: svdd._dist_center(X) / svdd.radius_
                for cl, svdd in self.individual_svdd.items()
            }
        else:
            dist_classes = {1: self._dist_center(X)/self.radius_}
        return pd.DataFrame(dist_classes)

    def dist_center_training_sample(self, r, alphas=None, cl=None):
        """Distance from vector #r to center.

        Args:
            r (int): rank of the vector
            alphas (array): list of alphas
            cl : class whose center will be used.
        """
        if cl is None:
            cl = 1
        if alphas is None:
            if len(self.classes_) > 1:
                alphas = alphas[cl]
            else:
                alphas = self.alphas
        K = self.kernel_matrix
        n = K.shape[0]
        # dist:
        # K_(r, r)
        # - 2 \sum_t \alpha_t \K_(r ,t)
        # + \sum_s\sum_t \alpha_s \alpha_t K_(r, t)
        return sum(
            [
                K[r, r],
                -2 * sum(alphas[t] * K[r, t] for t in range(n)),
                sum(
                    alphas[s] * alphas[t] * K[r, t]
                    for s, t in product(range(n), range(n))
                ),
            ]
        )
