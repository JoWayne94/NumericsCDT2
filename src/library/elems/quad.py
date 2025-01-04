"""
Quadrilateral element child class

Author: JWT
"""
from src.library.elems.element import Element as Elem
from src.library.elems.seg import Segment as Seg
from src.library.refs.basis import *
from src.library.refs.quad import RefQuad
from src.library.geoms.quad import Quadrilateral as QuadGeom


class Quadrilateral(Elem):
    """
    @:brief 2D implementation of a straight-sided quadrilateral
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
            self.inv_jacobian = None

    class ElemSolutionData:
        """
        @:brief Elemental solution data subclass
        """

        def __init__(self):
            self.u_hat = None
            self.u = None

    class ElemFacesObjects:
        """
        @:brief Element faces class instances
        """

        def __init__(self):
            self.F = np.empty(4, dtype=Seg)

    def __init__(self, ien, nodes, neighbour_labels, n_vars, source, p1, p2):
        """
        @:brief                  Main constructor
        :param ien:              Node IEN that make up the element
        :param nodes:            Node coordinates list parsed in by reference
        :param neighbour_labels: Neighbour element labels
        :param n_vars:           Number of state variables
        :param source:           Source term to be incorporated in the force vector
        :param p1:               Polynomial order in the x-direction
        :param p2:               Polynomial order in the y-direction
        """
        super().__init__(ien, nodes, neighbour_labels)
        self.ref_elem = RefQuad("GLL", p1, p2)
        self.mats = self.ElemMatricesData()
        self.geoms = self.ElemGeometricData()
        self.solution = self.ElemSolutionData()
        self.faces = self.ElemFacesObjects()

        # Geometric data
        self.geoms.geometry = QuadGeom(self.geom_data.nodes,
                                    np.array([i[0] for i in self.ref_elem.zeros]),
                                    np.array([i[1] for i in self.ref_elem.zeros]))
        self.geoms.det_jacobian = (self.geoms.geometry.det_jacobian()).reshape(-1, 1)
        self.geoms.inv_jacobian = self.geoms.geometry.inv_jacobian()

        # RHS
        self.source = np.array([source(i) for i in self.get_quadrature_coords.T]).reshape(-1, 1)

        # Matrices data
        """
        :param basis_matrix: Elemental basis matrix -> B ([Number of Gauss points, Number of polynomials])
        :param deriv_matrix: Elemental derivative matrix, d Phi/d xi -> (D_xi1 B) & (D_xi2 B) 
        """
        self.mats.basis_matrix, self.mats.deriv_matrix = Lagrange2dQuad(self.ref_elem.zeros,
                                                                        self.ref_elem.zeros[:p1 + 1, 0],
                                                                        self.ref_elem.zeros[:p1 + 1, 0]) # p1 = p2 for now

        # Solution data
        # self.solution.u_hat = np.zeros(((p1 + 1) * (p2 + 1), n_vars))
        # self.solution.u = np.zeros((len(self.ref_quad.zeros), n_vars))

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
        @:brief Calculates area of a polygon using the Shoelace formula
        :return: Area of the element
        """
        # https://en.wikipedia.org/wiki/Shoelace_formula
        # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        x = np.array([node[0] for node in self.geom_data.nodes])
        y = np.array([node[1] for node in self.geom_data.nodes])
        volume = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        return volume

    def calculate_face_normals(self):
        """
        @:brief Unit face normals of all faces [F0, F1, F2, F3]
        :return: Unit normal vectors
        """
        n = np.empty((4, 2))

        return n

    @property
    def get_quadrature_coords(self):
        """
        @:brief Quadrature coordinates getter
        :return: Quadrature coordinates in real space
        """
        return self.geoms.geometry.ref_mapping()

    def get_mass_matrix(self):
        """
        @:brief Construct Basis transposed, Weights, and Basis matrices to assemble the mass matrix
                B^T W B = M
        :return: Elemental mass matrix ([no. of gauss points, (p1+1) * (p2+1)]) per elem entry
        """
        mass_matrix = np.matmul(self.mats.basis_matrix.T,
                                self.mats.basis_matrix * self.ref_elem.weights * abs(self.geoms.det_jacobian))

        return mass_matrix

    def get_stiffness_matrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Basis matrices to assemble the stiffness matrix
                ((dxi1/dx1 + dxi1/dx2) (D_xi1 B)^T + (dxi2/dx1 + dxi2/dx2) (D_xi2 B)^T) W B = S
        :return: Elemental stiffness matrix per elem entry
        """
        """ Stiffness matrix seperated in x and y-direction """
        stiffness_x = np.matmul((self.geoms.inv_jacobian[:, :, 0][:, 0] * self.mats.deriv_matrix[:, :, 0] +
                                 self.geoms.inv_jacobian[:, :, 1][:, 0] * self.mats.deriv_matrix[:, :, 1]).T,
                                 self.mats.basis_matrix * self.ref_elem.weights * abs(self.geoms.det_jacobian))

        stiffness_y = np.matmul((self.geoms.inv_jacobian[:, :, 0][:, 1] * self.mats.deriv_matrix[:, :, 0] +
                                 self.geoms.inv_jacobian[:, :, 1][:, 1] * self.mats.deriv_matrix[:, :, 1]).T,
                                 self.mats.basis_matrix * self.ref_elem.weights * abs(self.geoms.det_jacobian))

        return [stiffness_x, stiffness_y]

    def get_laplacian_matrix(self):
        """
        @:brief Construct Derivative transposed, Weights, and Derivative matrices to assemble the Laplacian matrix
        :return: Elemental weak Laplacian matrix per elem entry
        """
        laplacian = np.matmul((self.geoms.inv_jacobian[:, :, 0][:, 0] * self.mats.deriv_matrix[:, :, 0] +
                               self.geoms.inv_jacobian[:, :, 1][:, 0] * self.mats.deriv_matrix[:, :, 1]).T,
                               self.ref_elem.weights * abs(self.geoms.det_jacobian) *
                              (self.geoms.inv_jacobian[:, :, 0][:, 0] * self.mats.deriv_matrix[:, :, 0] +
                               self.geoms.inv_jacobian[:, :, 1][:, 0] * self.mats.deriv_matrix[:, :, 1])) + \
                    np.matmul((self.geoms.inv_jacobian[:, :, 0][:, 1] * self.mats.deriv_matrix[:, :, 0] +
                               self.geoms.inv_jacobian[:, :, 1][:, 1] * self.mats.deriv_matrix[:, :, 1]).T,
                               self.ref_elem.weights * abs(self.geoms.det_jacobian) *
                              (self.geoms.inv_jacobian[:, :, 0][:, 1] * self.mats.deriv_matrix[:, :, 0] +
                               self.geoms.inv_jacobian[:, :, 1][:, 1] * self.mats.deriv_matrix[:, :, 1]))

        return laplacian

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
