"""
Quadrilateral shape derived class containing constant data

Author: JWT
"""
import numpy as np
from src.library.geoms.geometry import Geometry


class Quadrilateral(Geometry):
    """
    @:brief Quadrilateral geometry class to evaluate transformation and Jacobian with methods defined
    """
    # x_min = [0, 1]
    # x_max = [2, 3]
    # y_min = [3, 0]
    # y_max = [1, 2]

    def __init__(self, nodes, x, y):
        """
        @:brief            Main constructor for linear quad geometry
        :param nodes:      Node coordinates from the mesh parsed by reference
        :param x:          Array of quadrature zeros xi1-coordinates
        :param y:          Array of quadrature zeros xi2-coordinates
        """
        super().__init__()
        self.A = nodes[0]
        self.B = nodes[1]
        self.C = nodes[2]
        self.D = nodes[3]
        self.x = x
        self.y = y

    def ref_mapping(self):
        """
        @:brief Linear mapping between a reference quad element and a straight-sided quad element in real space
        :return: Arrays of x and y-coordinates in physical space given arrays in reference space
        """
        coords = np.empty((2, self.x.shape[0]), dtype=np.float64)
        for i in range(2):
            coords[i] = self.A[i] * ((1 - self.x) / 2) * ((1 - self.y) / 2) +\
                        self.B[i] * ((1 + self.x) / 2) * ((1 - self.y) / 2) +\
                        self.C[i] * ((1 - self.x) / 2) * ((1 + self.y) / 2) +\
                        self.D[i] * ((1 + self.x) / 2) * ((1 + self.y) / 2)

        return coords

    @property
    def dxdxi1(self):

        dxdxi1 = np.empty((2, self.x.shape[0]), dtype=np.float64)
        for i in range(2):
            dxdxi1[i] = -((self.C[i] - self.D[i] + self.B[i] - self.A[i]) * self.y +
                          self.C[i] - self.D[i] - self.B[i] + self.A[i]) / 4.

        return dxdxi1

    @property
    def dxdxi2(self):

        dxdxi2 = np.empty((2, self.x.shape[0]), dtype=np.float64)
        for i in range(2):
            dxdxi2[i] = -((self.C[i] - self.D[i] + self.B[i] - self.A[i]) * self.x -
                          self.C[i] - self.D[i] + self.B[i] + self.A[i]) / 4.

        return dxdxi2

    @property
    def jacobian(self):
        """
        @:brief Jacobian calculation routines
        :return: 2D Jacobian based on linear mapping above
        """
        return np.array([[self.dxdxi1.T[i], self.dxdxi2.T[i]] for i in range(len(self.dxdxi1[0]))])

    def inv_jacobian(self):
        return np.linalg.inv(self.jacobian)

    def det_jacobian(self):
        return np.linalg.det(self.jacobian)
