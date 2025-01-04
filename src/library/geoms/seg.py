"""
Segment shape subclass. All data held constant

Author: JWT
"""
from src.library.geoms.geometry import Geometry


class Segment(Geometry):
    """
    @:brief Segment geometry class to evaluate transformation and Jacobian with methods defined
    """
    def __init__(self, nodes, x):
        """
        @:brief        Main constructor. A and B are the first and second nodes of the geometry
        :param nodes:  Node coordinates from the mesh parsed by reference
        :param x:      Arbitrary zeros coordinates in the x-direction
        """
        super().__init__()
        self.x1A = nodes[0]
        self.x1B = nodes[1]
        self.x = x

    def ref_mapping(self):
        """
        @:brief Linear mapping of a line
        :return: x-coordinates in physical space corresponding to xi_1 inputs
        """
        x1 = self.x1A * ((1 - self.x) / 2) + self.x1B * ((1 + self.x) / 2)

        return x1

    @property
    def dx1dxi1(self):
        """
        @:brief  dx/dxi derivative based on the linear map above
        :return: dx/dxi
        """
        return (self.x1B - self.x1A) / 2

    @property
    def dxi1dx1(self):

        return 1. / self.dx1dxi1

    def det_jacobian(self):
        """
        @:brief  Determinant of Jacobian getter routine
        :return: 1D Jacobian based on linear mapping
        """
        return self.dx1dxi1
