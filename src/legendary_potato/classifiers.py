# coding: utf-8
"""
Classifiers.

Based on sklearn doc:
"http://scikit-learn.org/dev/developers/contributing.html\
#rolling-your-own-estimator"
"""
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .methods import KernelMethod


class SVDD(BaseEstimator, ClassifierMixin, KernelMethod):
    """Implement Support Vector DataDescription

    .. math::
        \\begin{cases}
            min_{r, c} & r^2 - C \\sum_t \\xi_t \\\\
            s.t        & y_i \\| \\phi(x_i) -c \\| < r^2 + xi_i \\forall i \\\\
                       & \\xi_i > 0  \\forall i \\\\
        \\end{cases}

    """

    def __init__(self, kernel_matrix=None, kernel=None, C=1):
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
        self.trained_on_sample = True  # use directly kernel matrix or sample?

    def fit(self, X, y=None, C=None, kernel=None, is_kernel_matrix=False):
        """Fit the classifier.

        Args:
            X: training samples.
            y: training labels. If None, consider all samples belongs to the
                same class (labeled "1").
            C (numeric): contraint in the soft margin case. If None or zero,
                then fall back to hard margin case.
            kernel (fun): kernel method to use. (default: linear)
            is_kernel_matrix (bool): if True, the input is treated as
                a kernel matrix.

        """
        # X, y = check_X_y(X, y) # TODO: add check method for X
        self._classifier_checks(X, y, C, kernel, is_kernel_matrix)

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
                The frontier between classes will be the computed hypersphere
                whose radius is multiply by this factor.

        """
        check_is_fitted(self, ["X_", "alphas_"])
        # X = check_array(X)

        if self.hypersphere_nb == 1:
            return self._predict_one_hypersphere(X, decision_radius)

        else:
            # check class
            dist_classes = self.relative_dist_all_centers(X)
            return np.array(dist_classes.idxmin(axis=1))

    def fit_predict(self, X, y, C=None, kernel=None, is_kernel_matrix=False):
        """Fit as the fit() methods.

        Returns:
            (array) : class for each training sample.
        """
        self.fit(X, y, C, kernel, is_kernel_matrix)
        self.predict(X)

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
        ret = np.sign(pred).reshape(-1)
        return list(map(lambda x: 1 if x == 0 else x, ret))

    def decision_function(self, X):
        """Generic decision value.

        Args:
            X (array-like): list of sample
        """
        return self._dist_center(X) / self.radius_

    def _dist_center(self, X=None):
        """Compute ditance to class center.

        Args:
            X (array-like): list of input vectors. If None, use the train set.

        Distance to center:
        .. math::
            \\| z - c \\|^2 = \\|z\\|^2 - 2 K(z, c) + \\|c\\|^2

            c = \\sum_t  \\alpha_t \\phi(X_t)
        """
        if not self.hypersphere_nb == 1:
            raise RuntimeWarning("Not available for multiclass SVDD")

        check_is_fitted(self, ["X_", "alphas_"])
        dim = len(self.alphas_)
        if X is None:
            # return distances for training set
            square_dists = [
                self.kernel_matrix[i, i]
                - 2
                * sum(
                    self.alphas_[t] * self.kernel_matrix[i, t]
                    for t in range(dim)
                )
                + sum(
                    self.alphas_[t]
                    * self.alphas_[s]
                    * self.kernel_matrix[s, t]
                    for s in range(dim)
                    for t in range(dim)
                )
                for i in range(dim)
            ]
        else:
            # return distances for vector X
            square_dists = [
                self.kernel(z, z)
                - 2
                * sum(
                    self.alphas_[t] * self.kernel(self.X_[t], z)
                    for t in range(dim)
                )
                + sum(
                    self.alphas_[s]
                    * self.alphas_[t]
                    * self.kernel(self.X_[t], self.X_[s])
                    for s in range(dim)
                    for t in range(dim)
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
        alphas = [1 / dim] * dim
        C = self.C
        upper = C * np.ones(dim)
        one = np.array([1])

        # TODO: test other solver
        # https://pypi.org/project/quadprog/
        # http://cvxopt.org/r

        def ell_d(al):
            """Dual function to minimize.

            function to maximize:
            .. maths::
                L_D = \\alpha diag(K)^T - \\alpha K \\alpha^T
                L_D = \\sum_s \\alpha_s K<x_s, x_s>
                      - \\sum_s \\sum_t \\alpha_s \\alpha_t K(x_s, x_t)
            """
            ay = al * y

            return -(
                np.mat(ay).dot(np.diag(self.kernel_matrix))
                - np.mat(ay).dot(self.kernel_matrix).dot(np.mat(ay).T)
            )

        cons = [
            # \forall i 0 \leq \alpha[i] \leq C
            LinearConstraint(A=np.identity(dim), lb=np.zeros(dim), ub=upper),
            # \sum_i \alpha[i] = 1
            LinearConstraint(A=np.ones(dim), lb=one, ub=one),
        ]

        # TODO: asyncio
        predicted_alphas = minimize(
            ell_d, alphas, constraints=cons, options={"maxiter": 10000}
        )
        if not predicted_alphas.success:
            raise RuntimeError(predicted_alphas.message)
        alphas = predicted_alphas.x

        # nullify almost null alphas:
        alphas = list(map(lambda x: 0 if np.isclose(x, 0) else x, alphas))

        # support vectors: 0 < alphas <= C
        support_vectors = set.intersection(
            set(np.where(np.less_equal(alphas, C))[0]),
            set(np.nonzero(alphas)[0]),
        )
        self.support_vectors_ = self.support_vectors_.union(support_vectors)

        if len(self.support_vectors_) < 2:
            radius = np.min(
                self.distance_matrix() + np.diag([C for _ in range(dim)])
            )
        else:
            # mean distance to support vectors
            radius = np.mean(
                [
                    self.dist_center_training_sample(r, alphas)
                    for r in self.support_vectors_
                ]
            )
        return radius, np.array(alphas)

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
            dist_classes = {1: self._dist_center(X) / self.radius_}
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

    def _center_one_class(self, mapping):
        """Compute hypersphere center.

        Args:
            mapping (fun): feature map. Default to identity (consistent with
        default kernel function)
        """
        check_is_fitted(self, ["X_", "alphas_"])
        if not self.trained_on_sample:
            raise RuntimeError("No access to initial vectors")

        center = np.sum(
            [
                self.alphas_[i] * np.array(mapping(self.X_[i]))
                for i in range(len(self.X_))
            ],
            axis=0,
        )
        return center

    def center(self, mapping=lambda x: x):
        """Compute center coordonates.

        Args:
            mapping (fun): feature map. Default to identity (consistent with
        default kernel function)
        """
        if self.hypersphere_nb > 1:
            return {
                cl: svdd._center_one_class(mapping)
                for cl, svdd in self.individual_svdd.items()
            }
        else:
            return {1: self._center_one_class(mapping)}


class SVM(BaseEstimator, ClassifierMixin, KernelMethod):
    """Implement Support Vector Machine

    .. math::
        \\begin{cases}
            min_{w, w_0, \\xi} & \\frac{1}{2} \\|w\\|^2 - C \\sum_t \\xi_t \\\\
            s.t            & y_t K(w, x_t) + w_0 > 1 -  \\xi_t  \\forall t \\\\
                           & \\xi_t > 0  \\forall t \\\\
        \\end{cases}
    """

    def __init__(self, kernel_matrix=None, kernel=np.dot, C=1):
        self.kernel_matrix = kernel_matrix  # kernel matrix used for training
        self.kernel = kernel
        self.C = C
        self.string_labels = False  # are labels strings or int?
        self.trained_on_sample = True  # use directly kernel matrix or sample?

    def fit(self, X, y=None, C=None, kernel=None, is_kernel_matrix=False):
        """Fit the classifier.

        Args:
            X: training samples.
            y: training labels. If None, consider all samples belongs to the
                same class (labeled "1").
            C (numeric): contraint in the soft margin case. If None or zero,
                then fall back to hard margin case.
            kernel (fun): kernel method to use. (default: linear)
            is_kernel_matrix (bool): if True, the input is treated as
                a kernel matrix.

        The optimisation problem is

        .. math::
            \\begin{cases}
                min_{\\alpha} & \\frac{1}{2} \\alpha^T K \\alpha - \\alpha^T ones \\\\
                s.t           & 0 \\leq \\alpha_t < C \\forall t \\\\
                              & (0 \\leq I alpha \\leq C I) \\\\
                              & \\alpha^T diag(Y) = 0
            \\end{cases}

        The generic QP problem is

        .. math::
            \\begin{cases}
                min_x   & \\frac{1}{2} x^T.P.x + q^T.x \\\\
                s.t     & G.x \\leq h \\\\
                        & A.x = b
            \\end{cases}

        In this SVM case:

        .. math::
            G = \\begin{pmatrix}
                -1 & & & 1 & & \\\\
                & \\ddots & & & \\ddots & \\\\
                & & -1 & & & 1
            \\end{pmatrix}

        .. math::
            h = \\begin{pmatrix}
                0 \\\\ \\vdots \\\\ 0 \\\\ C \\\\ \\vdots \\\\ C
            \\end{pmatrix}

        .. math::
            A = \\begin{pmatrix}
                y_1 & \\hdots & y_n
            \\end{pmatrix}

        .. math::
            b = 0

        """
        self._classifier_checks(X, y, C, kernel, is_kernel_matrix)
        C = self.C
        dim = len(self.X_)
        self.y_ = np.sign(np.array(y) - 0.1)
        alphas = [1 / dim] * dim  # warm start

        # TODO test other solver
        #
        # G = np.hstack([-1 * np.identity(dim), np.identity(dim)])
        # h = np.matrix(
        #     np.hstack([np.zeros(dim), C * np.ones(dim)])
        # ).transpose()
        # A = np.matrix(self.y)

        def ell_d(x):
            """Function to minimize.
            """
            x = np.matrix(x).transpose()
            return 1 / 2 * float(
                x.transpose().dot(self.kernel_matrix).dot(x)
            ) - float(np.ones(dim).dot(x))

        cons = [
            LinearConstraint(
                A=np.identity(dim), lb=np.zeros(dim), ub=C * np.ones(dim)
            ),
            LinearConstraint(A=self.y_, lb=np.array([0]), ub=np.array([0])),
        ]
        predicted_alphas = minimize(
            ell_d, alphas, constraints=cons, options={"maxiter": 10000}
        )
        if not predicted_alphas.success:
            raise RuntimeError(predicted_alphas.message)
        alphas = predicted_alphas.x

        # nullify almost null alphas:
        alphas = list(map(lambda x: 0 if np.isclose(x, 0) else x, alphas))
        self.alphas_ = alphas

        # support vectors: 0 < alphas <= C
        support_vectors = set.intersection(
            set(np.where(np.less_equal(alphas, C))[0]),
            set(np.nonzero(alphas)[0]),
        )
        self.support_vectors_ = self.support_vectors_.union(support_vectors)

        self.w0 = np.mean(
            [
                1
                - np.sum(
                    [
                        y[t] * alphas[t] * self.kernel_matrix[i, t]
                        for t in range(dim)
                    ]
                )
                for i in support_vectors
            ]
        )

        return np.array(alphas)

    def predict(self, X, decision_value=0):
        """Predict classes

        Args:
            X (array like): list of test samples.
            decision_value (numeric): decision value
        """
        check_is_fitted(self, ["X_", "alphas_"])
        if X is None:
            X = self.X_
        return [self._dist_hyperplane(x) for x in X]

    def _dist_hyperplane(self, x):
        check_is_fitted(self, ["X_", "alphas_"])
        return (
            sum(
                [
                    self.aplha_[i] * self.y_[i] * self.kernel(x, self.X_[i])
                    for i in len(self.alphas_)
                ]
            )
            + self.w0
        )

    def fit_predict(self, X, y, C=1, kernel=None, is_kernel_matrix=False):
        """Fit as the fit() methods.

        Returns:
            (array) : class for each training sample.
        """
        self.fit(X, y, C, kernel, is_kernel_matrix)
        self.predict(X)

    def decision_function(self, X):
        """Generic decision value.

        Args:
            X (array-like): list of samples
        """
        # TODO
        pass
