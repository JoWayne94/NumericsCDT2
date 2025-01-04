"""
1D advection diffusion convergence test by varying dx

Author: JWT
"""

# Import third-party libraries
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from scipy.integrate import ode


# Import user settings
from setup import *

# Configure system path
path = os.path.dirname(__file__)

if (sys.platform[:3] == 'win') or (sys.platform[:3] == 'Win'):
    sys.path.append(os.path.abspath(os.path.join(path, '..\..\..')))
else:
    sys.path.append(os.path.abspath(os.path.join(path, '../../..')))


# Import mesh generator
from mesh.meshGen import *

# Import mesh class
from src.library.mesh.mesh import Mesh

# Import equation types
from src.solvers.Helmholtz import *
from src.solvers.ADR import *


def source1(coords):
    return 0.

def exact1(coords):
    return (np.sin(2. * np.pi * (coords - VEL_FIELD[0] * FINAL_TIME))
            * np.exp(-4. * np.pi * np.pi * KAPPA * FINAL_TIME))


if __name__ == '__main__':
    """
    main()
    """

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
    plt.rc('text', usetex=True)
    lw = 1.5
    ms = 5.
    line_colours = ['rx--', 'go--', 'bs--', 'y', 'k']

    x_left = DOMAIN_BOUNDARIES[0][0]
    x_right = DOMAIN_BOUNDARIES[1][0]
    y_bottom = -1.1
    y_top = 1.1

    """ Problem definition """
    source = source1
    exact = exact1

    """ Define number of nodes to be tested """
    nnodes_list = [9, 17, 33, 65, 129]

    """ Error measures """
    L2_errors = np.empty(len(nnodes_list))

    for nn in range(len(nnodes_list)):

        NO_OF_ELEMS = nnodes_list[nn] - 1

        """ Mesh construction """
        nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen([nnodes_list[nn]], DOMAIN_BOUNDARIES,
                                                           BOUNDARY_CONDITIONS_TYPE, BOUNDARY_CONDITIONS_VALUES)

        cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
        cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1)

        """ Solve """
        lhs, force, mass, u0, mask = solve(cg, lm, NO_OF_ELEMS, P1, 0, VEL_FIELD, KAPPA)

        """ Initial conditions """
        # Initial datum/conditions defined with a numpy array
        for n in range(nodes.shape[0]):
            u0[n] = np.sin(2. * np.pi * nodes[n])

        dt_a = CFL * cg.elements[0].calculate_elem_vol()[0] / np.abs(VEL_FIELD[0])
        dt_d = (cg.elements[0].calculate_elem_vol()[0] ** 2) / (2. * np.abs(KAPPA))
        dt = min(dt_a, dt_d)

        t = 0.  # Initial time
        # Set up the ODE solver
        r = ode(mol).set_integrator('dopri5', max_step=dt)
        r.set_initial_value(u0[mask], t)
        r.set_f_params(mass, lhs, force)  # Pass additional arguments to dudt
        r.integrate(FINAL_TIME)

        u = np.empty_like(u0)
        """ Map solution back for visualisation """
        for e in range(NO_OF_ELEMS):
            for a in range(P1 + 1):
                u[IEN[e, a]] = r.y[lm[a, e]]

        """ Analytical solution """
        u_exact = exact(nodes)

        """ Append L2 error to list """
        L2_errors[nn] = compute_l2_err(u, u_exact, cg.elements[0].calculate_elem_vol())


    nnodes_list = [9., 17., 33., 65., 129.]
    dx_list = [(DOMAIN_BOUNDARIES[1][0] - DOMAIN_BOUNDARIES[0][0]) / d for d in np.array(nnodes_list) - 1]

    """ Compute numerical order of convergence """
    # Calculate n for each unique pair of (epsilon, delta x) values
    n_values = []
    for (i, j) in combinations(range(len(L2_errors)), 2):
        epsilon_1, epsilon_2 = L2_errors[i], L2_errors[j]
        delta_x_1, delta_x_2 = dx_list[i], dx_list[j]

        # Using the formula provided to calculate n for each pair
        n = (np.log(epsilon_1) - np.log(epsilon_2)) / (np.log(delta_x_1) - np.log(delta_x_2))
        n_values.append(n)

    # Calculate the average n
    average_n = np.mean(n_values)

    # Output the result
    print(f"Calculated n values for each pair: {n_values}.")
    print(f"Average value of n {average_n}.")

    """ Plot convergence """
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(np.array(nnodes_list) - 1, L2_errors, line_colours[0],
             label='test', lw=lw, ms=ms, fillstyle='none')

    # Define the desired gradient and constant
    m1 = -1  # Desired gradient (slope) on the log-log scale
    m2 = -2
    C1 = 7  # Constant that sets the vertical position of the line
    C2 = 3

    # Calculate y based on the power law y = C * x^m
    y1 = C1 * (np.array(nnodes_list) - 1) ** m1
    y2 = C2 * (np.array(nnodes_list) - 1) ** m2

    plt.plot(np.array(nnodes_list) - 1, y2, 'k:', label=r'$A \Delta x^2$', lw=lw, ms=ms)

    plt.legend(loc='best')
    plt.xlabel(r'$\Delta x$', usetex=True)
    # plt.xticks(np.array(nnodes_list) - 1, [f"$2 \pi / {int(nx)}$" for nx in np.array(nnodes_list) - 1])
    plt.ylabel(r'$L_2$ error', usetex=True)
    plt.title(r'Convergence on a log$_{10}$-log$_{10}$ scale', usetex=True)
    plt.grid()
    # plt.tick_params(axis='both', direction='in')
    # plt.savefig(f'{path}/convergence.eps', dpi=1000)
    plt.show()
