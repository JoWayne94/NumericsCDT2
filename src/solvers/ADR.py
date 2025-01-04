"""
Produce global matrices for the ADR equation

u_t + V \cdot \grad u - \epsilon \Delta u + \lambda u = f

Author: JWT
"""
from src.library.mesh.mesh import *
from src.utils.helpers import *
from collections import Counter



def adr(mesh: Mesh, lm, n_elems, p1, p2=0, vel=None, kappa=0):
    """
    @:brief         Assemble global LHS and RHS matrices based on location mapping
    :param mesh:    Mesh class object
    :param lm:      Location mapping/global assembly matrix
    :param n_elems: Number of elements
    :param p1:      Polynomial order in the x-direction
    :param p2:      Polynomial order in the y-direction
    :param vel:     Velocity vector
    :param kappa:   Diffusion coefficient
    :return:        Global matrices
    """
    # n_dofs = n_elems * p1 + 1
    n_dofs = np.max(mesh.connectivity_data.ien) + 1
    lm = extend_lm(lm, n_elems, p1)

    n_eq = np.max(lm) + 1
    # global_lhs = np.zeros((n_eq, n_eq))
    global_lap = sparse.lil_matrix((n_eq, n_eq))
    global_stn = sparse.lil_matrix((n_eq, n_eq))

    # global_mass = np.zeros((n_eq, n_eq))
    global_mass = sparse.lil_matrix((n_eq, n_eq))

    global_force = np.zeros((n_eq,))
    u = np.zeros(n_dofs)

    dirichlet_list = []
    unique_nodes = []

    mp = Counter(mesh.connectivity_data.ids)
    # Have to find a way to account for mixed periodic and Neumann BCs
    for it in mp:
        if mp[it] == 1:
            unique_nodes.append(it)

    for e in range(n_elems):

        l_e = mesh.elements[e].laplacian_matrix
        f_e = mesh.elements[e].force_vector
        s_e = mesh.elements[e].stiffness_matrix
        m_e = mesh.elements[e].mass_matrix
        n_nodes = l_e.shape[0]  # Number of nodes (DOFs) in this element
        tmp_e = np.zeros((n_nodes, n_nodes))

        for a in range(n_nodes):
            A = lm[a, e]
            for b in range(n_nodes):
                B = lm[b, e]
                if (A >= 0) and (B >= 0):

                    """ Elemental Laplacian summation """
                    tmp_e[a, b] += kappa * l_e[a, b]
                    global_lap[A, B] += kappa * l_e[a, b]

                    """ Elemental stiffness summation """
                    for d in range(mesh.connectivity_data.n_dims):
                        tmp_e[a, b] -= vel[d] * s_e[d][a, b]
                        global_stn[A, B] -= vel[d] * s_e[d][a, b]
                    # global_lhs[A, B] -= vel[0] * s_e[a, b]

                    """ Elemental mass summation """
                    global_mass[A, B] += m_e[a, b]

            if A >= 0:
                global_force[A] += f_e[a]

        """ Enforce BCs """
        for node in mesh.elements[e].geom_data.ien:
            if node in mesh.connectivity_data.boundaries:
                if mesh.connectivity_data.ids[node] == -1:
                    dirichlet_list.append(node)

                    for b in range(n_nodes):  # Loop over all local DOFs of the first element
                        B = lm[b, e]
                        if B >= 0:
                            # Modify force vector for Dirichlet BC
                            global_force[B] -= mesh.connectivity_data.boundaries[node] * tmp_e[0, b]  # [1, 0]

                    u[node] = mesh.connectivity_data.boundaries[node]
                else:
                    # Modify force vector for Neumann BC
                    if node in unique_nodes:
                        global_force[mesh.connectivity_data.ids[node]] += mesh.connectivity_data.boundaries[node]

    """ Periodic BCs treated differently """
    # for j in mesh.connectivity_data.boundaries.values():
    #     if j not in dirichlet_list:
    #         dirichlet_list.append(j)

    # Create a mask for indices not in the exclude list
    mask = np.ones(n_dofs, dtype=bool)  # Start with all True
    mask[dirichlet_list] = False  # Set the indices to exclude to False

    # Solve
    # u[np.where(mesh.connectivity_data.ids != -1)] = np.linalg.solve(global_lhs, global_force)
    # u[mask] = np.linalg.solve(global_lhs, global_force)
    global_lap = sparse.csr_matrix(global_lap)
    global_stn = sparse.csr_matrix(global_stn)
    global_mass = sparse.csr_matrix(global_mass)

    return global_lap, global_stn, global_force, global_mass, u, mask
