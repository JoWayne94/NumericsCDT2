"""
Solves the Helmholtz equation

-\epsilon \Delta u + \lambda u = f

Author: JWT
"""
from src.library.mesh.mesh import *
from src.utils.helpers import *


def helmholtz_operator(mesh: Mesh, lm, n_elems, p1, p2=0, kappa=1.):
    """
    @:brief         Assemble global LHS and RHS matrices based on location mapping
    :param mesh:    Mesh class object
    :param lm:      Location mapping/global assembly matrix
    :param n_elems: Number of elements
    :param p1:      Polynomial order in the x-direction
    :param p2:      Polynomial order in the y-direction
    :param kappa:   Diffusivity coefficient
    :return:        Global matrices
    """
    # n_dofs = n_elems * p1 + 1
    n_dofs = np.max(mesh.connectivity_data.ien) + 1
    lm = extend_lm(lm, n_elems, p1)

    n_eq = np.max(lm) + 1
    # global_lhs = np.zeros((n_eq, n_eq))
    global_lhs = sparse.lil_matrix((n_eq, n_eq))
    global_force = np.zeros((n_eq,))
    u = np.zeros(n_dofs)

    dirichlet_list = []

    for e in range(n_elems):

        l_e = mesh.elements[e].laplacian_matrix
        f_e = mesh.elements[e].force_vector
        n_nodes = l_e.shape[0]  # Number of nodes (DOFs) in this element

        for a in range(n_nodes):
            A = lm[a, e]
            for b in range(n_nodes):
                B = lm[b, e]
                if (A >= 0) and (B >= 0):

                    """ Elemental Laplacian summation """
                    global_lhs[A, B] += l_e[a, b]

            if A >= 0:
                global_force[A] += f_e[a]

        """ Enforce BCs """
        for node in mesh.elements[e].geom_data.ien:
            if node in mesh.connectivity_data.boundaries:
                if mesh.connectivity_data.ids[node] == -1:
                    dirichlet_list.append(node)
                    # global_force[node] += mesh.connectivity_data.boundaries[node]
                    # global_lhs[node, node] = 1.

                    for b in range(n_nodes):  # Loop over all local DOFs of the first element
                        B = lm[b, e]
                        if B >= 0:
                            # Modify force vector for Dirichlet BC
                            global_force[B] -= mesh.connectivity_data.boundaries[node] * l_e[0, b]  # [1, 0]

                            u[B] = mesh.connectivity_data.boundaries[node]
                else:
                    # Modify force vector for Neumann BC
                    global_force[mesh.connectivity_data.ids[node]] += mesh.connectivity_data.boundaries[node]

    # Create a mask for indices not in the exclude list
    mask = np.ones(n_dofs, dtype=bool)  # Start with all True
    mask[dirichlet_list] = False  # Set the indices to exclude to False

    # Solve
    # u[np.where(mesh.connectivity_data.ids != -1)] = np.linalg.solve(global_lhs, global_force)
    global_lhs = sparse.csr_matrix(global_lhs)
    u[mask] = sparse.linalg.spsolve(kappa * global_lhs, global_force)

    return global_lhs, global_force, u
