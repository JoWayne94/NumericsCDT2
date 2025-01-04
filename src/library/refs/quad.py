"""
Subclass to evaluate quadrature of a quadrilateral

Author: JWT
"""
import numpy as np
from src.library.refs.element2d import RefElem2d as Elem
from src.library.refs.seg import RefSeg as Segment


class RefQuad(Elem):
    """
    @:brief Subclass to represent a 2d quad element in reference space
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
        @:brief Calls the segment quadrature function to get quadrature zeros and weights.
                Construct quadrilateral shapes using tensor products
        :return: Quadrature point coordinates and weights
        """
        # Get quadrature zeros and weights from 1D
        segment1 = Segment(self.quadrature, self.p1)
        segment2 = Segment(self.quadrature, self.p2)
        q_zeros1, q_weights1 = segment1.get_quadrature_zeros_weights()
        q_zeros2, q_weights2 = segment2.get_quadrature_zeros_weights()

        # Weights
        q_weights = np.reshape(np.outer(q_weights1, q_weights2), (-1,), 'F').reshape(-1, 1)

        # Zeros
        q_zeros = np.zeros([q_weights.shape[0], 2])
        # https://numpy.org/doc/stable/reference/generated/numpy.tile.html
        q_zeros[:, 0] = np.tile(q_zeros1, (q_zeros2.shape[0], 1)).reshape(-1)
        q_zeros[:, 1] = np.repeat(q_zeros2, q_zeros1.shape[0], axis=0).reshape(-1)

        return q_zeros, q_weights
