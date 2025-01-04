# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
"""
Main module

Note:

    1. Serves as a template to create other test cases
    2. Computational domain only limited to rectangles or manually inputted
    3. h is constant for each spatial dimension (for default domains)

Future works:

1.

Author: JWT
"""

# Import third-party libraries
import sys, os
import matplotlib.pyplot as plt

# Import user settings
from setup import *

# Import mesh generator
from mesh.meshGen import *

# Import mesh class
from src.library.mesh.mesh import Mesh

# Import equation types
from src.solvers.ADR import *
from src.solvers.Helmholtz import *

# Configure system path
path = os.path.dirname(__file__)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def source(coords, std=10000.):
    return np.exp((-1. / (2. * std ** 2)) * ((coords[0] - soton[0]) ** 2 + (coords[1] - soton[1]) ** 2))


if __name__ == '__main__':
    """
    main()
    """

    # if len(sys.argv) < 2:
    if EQUATION_TYPE == 'Helmholtz':
        solve = helmholtz_operator
    elif EQUATION_TYPE == 'ADR':
        solve = adr
    else:
        raise NotImplementedError('Equation type is not implemented.')

    if NO_OF_DIMENSIONS == 1:
        mesh_gen = generate_1d_mesh
        ELEM_SHAPE = "S"
        NO_OF_ELEMS = NO_OF_NODES[0] - 1
    elif NO_OF_DIMENSIONS == 2:
        mesh_gen = generate_2d_mesh
        NO_OF_ELEMS = (NO_OF_NODES[0] - 1) * (NO_OF_NODES[1] - 1)
        if ELEM_SHAPE == 'T': NO_OF_ELEMS *= 2
    else:
        raise NotImplementedError('Number of spatial dimensions not supported.')

    """ Plotting parameters and visualisations """
    lw = 1.5
    ms = 5.
    line_styles = ['r.', 'gv', 'b<', 'k*', 'ro', 'g^', 'b>', 'ks']
    plt.rc('text', usetex=True)

    """ Problem definition """
    soton = np.array([442365., 115483])
    reading = np.array([473993., 171625.])

    if USER_INPUT:
        nodes = np.loadtxt(path + f'/mesh/las_grids/las_nodes_{INPUT_NAME}k.txt')
        IEN = np.loadtxt(path + f'/mesh/las_grids/las_IEN_{INPUT_NAME}k.txt', dtype=np.int64)
        boundary_nodes = np.loadtxt(path + f'/mesh/las_grids/las_bdry_{INPUT_NAME}k.txt',
                                    dtype=np.int64)

        south_boundary = np.where(nodes[boundary_nodes, 1] <= 150000.)[0]

        # Make all boundary points Dirichlet
        ID = np.zeros(len(nodes), dtype=np.int64)
        boundaries = dict()  # hold the boundary values
        n_eq = 0
        for i in range(len(nodes)):
            if i in south_boundary:
                ID[i] = -1
                boundaries[i] = 0.  # Dirichlet BC
            else:
                ID[i] = n_eq
                n_eq += 1
        mat_dim = np.max(ID) + 1
        NO_OF_ELEMS = IEN.shape[0]
        N_nodes = nodes.shape[0]
        NO_OF_DIMENSIONS = nodes.shape[1]
        # Location matrix
        lm = np.zeros_like(IEN.T)
        for e in range(NO_OF_ELEMS):
            for a in range(IEN.shape[1]):
                lm[a, e] = ID[IEN[e, a]]

        plot_uk_mesh(nodes, south_boundary, IEN, lw)

    else:
        nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                           BOUNDARY_CONDITIONS_TYPE,
                                                           BOUNDARY_CONDITIONS_VALUES, ELEM_SHAPE)

    directed_at_reading = reading - soton
    # VEL_FIELD = -10. / np.linalg.norm(directed_at_reading) * directed_at_reading

    """ Mesh construction """
    cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
    cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1, P2)

    lhs, force, _, u, mask = solve(cg, lm, NO_OF_ELEMS, P1, P2, VEL_FIELD, KAPPA)

    """ Solve linear system """
    u[mask] = sparse.linalg.spsolve(lhs, force)
    u /= np.max(u)

    """ Visualisation """
    x_left = np.min(nodes[:, 0])
    x_right = np.max(nodes[:, 0])
    y_bottom = np.min(nodes[:, 1])
    y_top = np.max(nodes[:, 1])

    plot_uk_solutions(nodes[:, 0], nodes[:, 1], u, IEN, x_left, x_right, y_bottom, y_top,
                      f"$N_e = $ {NO_OF_ELEMS}, steady advection-diffusion case", ms, soton, reading)
