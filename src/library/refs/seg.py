"""
Subclass to evaluate quadrature of a segment/line.

Add: Gauss Radau, Vertex-based shape where quadrature points are treated as solution nodes.
"""
import numpy as np
from src.library.refs.element import RefElem as Elem


class RefSeg(Elem):

    def __init__(self, quadrature, p1):
        """
        @:brief            Main constructor
        :param p1:         Polynomial order in the xi_1 direction
        :param quadrature: Type of quadrature selected
        """
        self.P1 = p1
        self.Quadrature = quadrature
        self.Zeros, self.Weights = self.get_quadrature_zeros_weights()

    @property
    def p1(self):
        return self.P1

    @p1.setter
    def p1(self, value):
        self.P1 = value

    @property
    def quadrature(self):
        return self.Quadrature

    @property
    def zeros(self):
        return self.Zeros

    @property
    def weights(self):
        return self.Weights

    def get_quadrature_zeros_weights(self):

        if self.quadrature == "GL":  # Gauss-Legendre, or self.p1 == 1
            q_zeros, q_weights = self.get_quadrature_gl()
        elif self.quadrature == "GLL":  # Gauss-Lobatto-Legendre
            q_zeros, q_weights = self.get_quadrature_gll()
        else:
            raise NotImplementedError('Type of quadrature not implemented.')

        return q_zeros, q_weights

    def get_quadrature_gl(self):
        """
        @:brief  Get quadrature zeros and weights using numpy Gauss-Legendre library
        :return: Quadrature point coordinates and weights
        """
        # Number of integration points to obtain exact integration given polynomial order
        n_zeros = self.p1 + 1 # int((self.p1 + 3) / 2)

        q_zeros, q_weights = np.polynomial.legendre.leggauss(n_zeros)

        q_zeros.shape = -1, 1
        q_weights.shape = -1, 1

        return q_zeros, q_weights

    def get_quadrature_gll(self):
        """
        @:brief  Get Gauss-Lobatto-Legendre quadrature zeros and weights
        :return: Quadrature point coordinates and weights
        """
        # If p is even, add 1
        # if self.p1 % 2 == 0:
        #     self.p1 += 1
        # Number of integration points to obtain exact integration given polynomial order
        n_zeros = self.p1 + 1  # (self.p1 + 3) // 2

        q_zeros, q_weights = self.get_gll_zeros_weights(n_zeros, np.finfo(float).eps)
        q_zeros = q_zeros.reshape(q_zeros.shape[0], 1)
        q_weights = q_weights.reshape(q_weights.shape[0], 1)

        return q_zeros, q_weights

    @staticmethod
    def get_gll_zeros_weights(m, tol):
        """
        @:brief Get Gauss-Lobatto-Legendre zeros and weights using Newton-Raphson iteration
                https://uk.mathworks.com/matlabcentral/fileexchange/4775-legende-gauss-lobatto-nodes-and-weights
        :param m:   n_zeros - 1
        :param tol: Tolerance set as machine precision/epsilon
        :return:    Quadrature zero coordinates and weights
        """
        legendre = np.polynomial.legendre.Legendre

        # i = 0, ..., m-1
        i = np.arange(m)
        # Initialise Legendre polynomials
        tmp = np.zeros([m, m])
        # Chebyshev polynomial as initial guess
        x_im = -np.cos(((2 * i + 1) / (2 * (m - 1))) * np.pi)
        # Max no. of iterations
        niter = 1000
        # Iterative evaluation to get zeros of Legendre polynomial derivatives
        for k in range(m):

            r = x_im[k]

            if k > 0:
                r = (r + x_im[k - 1]) * 0.5

            for j in range(niter):
                s = np.sum((1 / (r - x_im))[0:k])
                # Initialise 1D Vandermonde matrix
                V = np.polynomial.legendre.legvander(r, m - 1)
                delta = - 1 / (((m * V[0][m - 1]) / (r * V[0][m - 1] - V[0][m - 2])) - s)
                r += delta

                # Check if tolerance is met
                if np.amax(np.abs(delta)) < tol:
                    break
                # Convergence error
                if j == niter - 1:
                    raise ValueError

            x_im[k] = r

        # Evaluate Legendre polynomials
        for i in range(m):
            tmp[:, i] = legendre.basis(i)(x_im)

        # Quadrature weights
        weights = 2. / (m * (m - 1) * tmp[:, m - 1] ** 2)

        return x_im, weights
