'''
Calissifiers
'''
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
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
        self.kernel = kernel
        self.C = C

    def fit(self, X, y, C=None, kernel=None, is_kernel_matrix=False,
            *args, **kwargs):
        """Fit the classifier.

        C -- contraint in the soft margin case. If None or zero, then fall back
        to hard margin case
        kernel -- kernel method to use
        is_kernel_matrix -- if true, the input is  treated as a kernel matrix
        *args, **kwargs -- extra arguments for kernel function
        """
        # TODO
        if (y > np.zeros(len(y))).all():
            # One class
            dim = X.shape[0]
            y = np.ones(dim, dtype=int)

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) > 2:
            # TODO implement multiclass
            raise ValueError("Expected at most 2 classes, "
                             "received %s" % int(len(self.classes_)))
        self.X_ = X
        self.y_ = np.sign(y)
        self.C = C
        return self

    def predict(self, X, kernel=None):
        # TODO
        check_is_fitted(self,  ['X_', 'y_'])
        X = check_array(X)
        return np.ones(X.shape[0], dtype=int)  # TODO delete
        return np.array(
            self.utils.distance(self.center, vect) / self.radius - 1
            for vect in X
        )

    def _fit(self):
        """Perform actual fit process

        * compute alphas
        * compute support vectors
        * recompute minimal kernel matrix
        """
        dim = len(self.X_)
        alphas = [0 for _ in range(dim)]

        def ell_d(alphas):
            """Dual function to minimize.

            function to maximize:
            .. maths::
            \alpha diag(K)^T - \alpha K \alpha^T
            """
            ay = alphas * self.y_
            return (ay.dot(self.kernel_matrix).dot(ay.T)
                    - ay.dot(np.diag(self.kernel_matrix)))

        # \forall i \alpha[i] \geq 0 \leftrightdoublearrow min(\alpha) \geq 0
        cons = [{'type': 'eq',   'fun': lambda alphas: np.sum(alphas) - 1},
                {'type': 'ineq', 'fun': lambda alphas: np.min(alphas)}]
        if self.C:
            # soft margin case: \forall i \alpha[i] \leq C
            cons.append({'type': 'ineq',
                         'fun': lambda alphas: self.C - np.max(alphas)})

        self.alphas_ = minimize(ell_d, alphas, contraints=tuple(cons))
