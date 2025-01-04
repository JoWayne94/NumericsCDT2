"""
Subclass to evaluate quadrature of a triangle

Author: JWT
"""
import numpy as np
from src.library.refs.element2d import RefElem2d as Elem
from src.library.refs.quad import RefQuad


class RefTri(Elem):
    """
    @:brief Subclass to represent a 2d tri element in reference space
    """
    def __init__(self, quadrature, p1, p2):
        """
        @:brief            Main constructor
        :param p1:         Polynomial order in the xi_1 direction
        :param p2:         Polynomial order in the xi_2 direction
        :param quadrature: Type of quadrature selected
        """
        self.P1 = p1
        self.P2 = p2
        self.Quadrature = quadrature
        self.Zeros, self.Weights = self.get_quadrature_zeros_weights()

    @property
    def p1(self):
        return self.P1

    @property
    def p2(self):
        return self.P2

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
        """
        @:brief Calls the quadrilateral quadrature function to get quadrature zeros and weights.
                Construct triangular shapes from quadrilateral
        :return: Quadrature point coordinates and weights
        """
        # Hard-coded for now
        q_zeros = np.array([1. / 6., 1. / 6.,
                         4. / 6., 1. / 6.,
                         1. / 6., 4. / 6.])
        # q_zeros = np.array([-2. / 3., -2. / 3.,
        #                     1. / 3., -2. / 3.,
        #                     -2. / 3., 1. / 3.])
        q_weights = np.array([1. / 6., 1. / 6., 1. / 6.])

        q_zeros.shape = -1, 2
        q_weights.shape = -1, 1

        # ref_quad = RefQuad('GL', self.P1 + 1, 0)
        # q_zeros, q_weights = ref_quad.get_quadrature_zeros_weights()
        #
        # # Apply transformation
        # q_weights[:, 0] *= 0.125 * (1. - q_zeros[:, 0])
        # q_zeros[:, 1] = 0.25 * (1. - q_zeros[:, 0]) * (1. + q_zeros[:, 1])
        # q_zeros[:, 0] = 0.5 * (1. + q_zeros[:, 0])

        return q_zeros, q_weights
