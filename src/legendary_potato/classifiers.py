'''
Classifiers.
'''
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import minimize

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
        self.string_labels = False          # are labels stings or int?

    def fit(self, X, y, C=None, kernel=None, is_kernel_matrix=False,
            *args, **kwargs):
        """Fit the classifier.

        C -- contraint in the soft margin case. If None or zero, then fall back
        to hard margin case
        kernel -- kernel method to use
        is_kernel_matrix -- if true, the input is  treated as a kernel matrix
        *args, **kwargs -- extra arguments for kernel function
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        self.C = C
        self.support_vectors_ = set()
        dim = len(y)
        if is_kernel_matrix:
            self.kernel_matrix = X
        else:
            if self.kernel is None:
                raise ValueError("You must provide either a kernel function "
                                 "or a kernel matrix.")
            self.sample = self.X_
            self.kernel_matrix = self.matrix()
        self.classes_ = np.unique(y)

        if np.isreal(y[0]):
            self.string_labels = True
        if (len(self.classes_) > 2) or (len(self.classes_) == 2
                                        and self.string_labels):
            # each class has its own hypersphere (one class vs rest)
            # TODO
            self.ys_ = {
                cl: np.array([1 if y[i] == cl else -1 for i in range(dim)])
                for cl in self.classes_
            }
            self.alphas_ = {}
            self.radius_ = {}
            for cl in self.classes_:
                alphas, radius = self._fit_two_classes(self.ys_[cl])
                self.alphas_[cl], self.radius_[cl] = alphas, radius
        else:
            # one or two classes
            self.y_ = np.sign(y)
            self.radius_, self.alphas_ = self._fit_two_classes()
        return self

    def predict(self, X, kernel=None):
        # TODO
        check_is_fitted(self,  ['X_', 'alphas_'])
        X = check_array(X)
        return np.ones(X.shape[0], dtype=int)  # TODO delete
        return np.array(
            self.utils.distance(self.center, vect) / self.radius - 1
            for vect in X
        )

    def _fit_two_classes(self, y=None, class1=1, class2=-1):
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
            return (ay.dot(self.kernel_matrix).dot(ay.T)
                    - ay.dot(np.diag(self.kernel_matrix)))

        # \forall i \alpha[i] \geq 0 \leftrightdoublearrow min(\alpha) \geq 0
        cons = [{'type': 'eq',   'fun': lambda al: np.sum(al) - 1},
                {'type': 'ineq', 'fun': lambda al: np.min(al)}]
        if C and C is not np.inf:
            # soft margin case: \forall i \alpha[i] \leq C
            cons.append({'type': 'ineq',
                         'fun': lambda alphas: C - np.max(alphas)})
        else:
            C = np.inf
        predicted_alphas = minimize(ell_d, alphas, constraints=tuple(cons))
        alphas = predicted_alphas.x
        support_vectors = set(np.where(i < C and not np.isclose(i, 0)
                                       for i in alphas)[0])
        self.support_vectors_ = self.support_vectors_.union(support_vectors)
        radius = np.mean([
            self.kernel_matrix[r, r]
            - 2 * np.sum(alphas[t] * self.kernel_matrix[t, r]
                         for t in range(dim))
            + np.sum(alphas[s] * alphas[t] * self.kernel_matrix[r, t]
                     for s in range(dim)
                     for t in range(dim))
            for r in range(dim)
            if alphas[r] < C and not np.isclose(0, alphas[r])
        ])
        return radius, alphas
