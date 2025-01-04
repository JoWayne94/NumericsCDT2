"""
Segment element child class

Notes:

    1. Move elemental matrices routines to mats dir and derive new classes

Author: JWT
"""
from src.library.elems.element import Element as Elem
from src.library.refs.basis import *
from src.library.refs.seg import RefSeg
from src.library.geoms.seg import Segment as SegGeom


class Segment(Elem):
    """
    @:brief 1D implementation of an element
    """

    class ElemMatricesData:
        """
        @:brief Elemental matrices data subclass
        """
        def __init__(self):
            self.basis_matrix = None
            self.deriv_matrix = None

    class ElemGeometricData:
        """
        @:brief Elemental geometric data subclass
        """
        def __init__(self):
            self.geometry = None
            self.det_jacobian = None

    class ElemSolutionData:
        """
        @:brief Elemental solution data subclass
        """
        def __init__(self):
            self.u_hat = None
            self.u = None

    def __init__(self, ien, nodes, neighbour_labels, n_vars, source, p1):
        """
        @:brief                  Main constructor
        :param ien:              Node IEN that make up the element
        :param nodes:            Node coordinates list parsed in by reference
        :param neighbour_labels: Neighbour element labels
        :param n_vars:           Number of state variables
        :param source:           Source term to be incorporated in the force vector
        :param p1:               Polynomial order in the x-direction
        """
        super().__init__(ien, nodes, neighbour_labels)
        self.ref_elem = RefSeg("GLL", p1)
        self.mats = self.ElemMatricesData()
        self.geoms = self.ElemGeometricData()
        self.solution = self.ElemSolutionData()

        # Geometric data
        self.geoms.geometry = SegGeom(self.geom_data.nodes, np.array([i[0] for i in self.ref_elem.zeros]))
        self.geoms.det_jacobian = (self.geoms.geometry.det_jacobian()).reshape(-1, 1)

        # RHS
        self.source = np.array([source(i) for i in self.get_quadrature_coords]).reshape(-1, 1)

        # Matrices data
        """
        :param basis_matrix: Elemental basis matrix -> B ([Number of Gauss points, Number of polynomials])
        :param deriv_matrix: Elemental derivative matrix, d Phi/d xi -> (D B)
        """
        self.mats.basis_matrix = Lagrange1d(self.ref_elem.zeros, self.ref_elem.zeros)
        self.mats.deriv_matrix = Lagrange1dGrad(self.ref_elem.zeros, self.ref_elem.zeros)

        # Solution data
        self.solution.u_hat = np.zeros((p1 + 1, n_vars))
        self.solution.u = np.zeros((len(self.ref_elem.zeros), n_vars))

        # Matrix operators
        """
        :param mass_matrix:      Elemental mass matrix -> B^T W B
        :param stiffness_matrix: Elemental stiffness matrix -> d xi/d x (D B)^T W B
        :param laplacian_matrix: Elemental weak Laplacian matrix -> d xi/d x (D B)^T W (D B) d xi/d x
        """
        self.mass_matrix = self.get_mass_matrix()
        # self.inv_mass_matrix = np.linalg.inv(self.mass_matrix)
        self.stiffness_matrix = self.get_stiffness_matrix()
        self.laplacian_matrix = self.get_laplacian_matrix()
        self.force_vector = self.get_forcing_function()

    def calculate_elem_vol(self):
        """
        @:brief  Calculate 1D element length
        :return: Element length
        """
        return abs(self.geom_data.nodes[1] - self.geom_data.nodes[0])

    def calculate_face_normals(self):
        """
        @:brief  Define face unit normal vectors
        :return: [-1] for F0 and [+1] for F1
        """
        return np.array([-1, 1])

    @property
    def get_quadrature_coords(self):
        """
        @:brief  Quadrature coordinates getter
        :return: Quadrature coordinates in real space
        """
        return self.geoms.geometry.ref_mapping()

    def get_mass_matrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Basis matrices to assemble the mass matrix
                B^T W B = M
        :return: Elemental mass matrix ([no. of gauss points, p+1]) per element entry
        """
        mass_matrix = np.matmul(self.mats.basis_matrix.T,
                                self.mats.basis_matrix * self.ref_elem.weights * abs(self.geoms.det_jacobian))

        return mass_matrix

    def get_stiffness_matrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Basis matrices to assemble the stiffness matrix
                dxi/dx (D B)^T W B = S
        :return: Elemental stiffness matrix (non-singular) per element entry
        """
        stiffness_matrix = np.matmul(self.geoms.geometry.dxi1dx1 * self.mats.deriv_matrix[:, :, 0].T,
                                     self.mats.basis_matrix * self.ref_elem.weights * abs(self.geoms.det_jacobian))

        return [stiffness_matrix]

    def get_laplacian_matrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Derivative matrices to assemble the weak Laplacian matrix
                dxi/dx (D B)^T W dxi/dx (D B) = L
        :return: Elemental weak Laplacian matrix (symmetric) per element entry
        """
        laplacian_matrix = np.matmul(self.geoms.geometry.dxi1dx1 * self.mats.deriv_matrix[:, :, 0].T,
                                    self.geoms.geometry.dxi1dx1 * self.mats.deriv_matrix[:, :, 0] *
                                    self.ref_elem.weights * abs(self.geoms.det_jacobian))

        return laplacian_matrix

    def get_forcing_function(self):
        """
        @:brief Construct Basis transposed, Weights, and source terms to assemble the force vector
                B^T W f = F
        :return: Elemental force vector ([no. of gauss points]) per element entry
        """
        force_vector = np.matmul(self.mats.basis_matrix.T, self.source * self.ref_elem.weights *
                                 abs(self.geoms.det_jacobian)).reshape(-1,)

        # force_vector = self.mass_matrix @ self.source

        return force_vector
