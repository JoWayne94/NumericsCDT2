"""
Test cases for the 1d ADR equation

Author: JWT
"""

# Import third-party libraries
import sys, os
import matplotlib.pyplot as plt
import numpy as np
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
    plt.rc('text', usetex=True)

    x_left = DOMAIN_BOUNDARIES[0][0]
    x_right = DOMAIN_BOUNDARIES[1][0]
    y_bottom = -1.1
    y_top = 1.1

    """ Problem definition """
    source = source1
    exact = exact1
    nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                       BOUNDARY_CONDITIONS_TYPE, BOUNDARY_CONDITIONS_VALUES)

    # print(f"nodes:\n{nodes}")
    # print(f"IEN:\n{IEN}")
    # print(f"ID:\n{ID}")
    # print(f"boundaries:\n{boundaries}")
    # print(f"lm:\n{lm}")

    """ Mesh construction """
    cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
    cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1)

    lhs, force, mass, u0, mask = solve(cg, lm, NO_OF_ELEMS, P1, 0, VEL_FIELD, KAPPA)
    # print(f"Solutions:\n{u}")
    # print(f"Global Laplacian matrix:\n{lhs}")
    # print(f"Global force vector:\n{rhs}")

    """ Initial conditions """
    # Initial datum/conditions defined with a numpy array
    for n in range(nodes.shape[0]):
        u0[n] = np.sin(2. * np.pi * nodes[n])
        # if 0.5 <= nodes[n] <= 1.5: u[n] = np.sin(2. * np.pi * nodes[n])
        # else: u[n] = 0.

    dt_a = CFL * cg.elements[0].calculate_elem_vol()[0] / np.abs(VEL_FIELD[0])
    dt_d = (cg.elements[0].calculate_elem_vol()[0]**2) / (2. * np.abs(KAPPA))
    print(f"Time-step size limited by advection: {dt_a}")
    print(f"Time-step size due to diffusion restriction: {dt_d}")
    dt = min(dt_a, dt_d)

    # Initial conditions
    plt.plot(nodes, u0[:NO_OF_NODES[0]], 'k', label='IC')
    plt.legend(loc='best')
    plt.xlabel(r'$x$', usetex=True)
    plt.ylabel(r'$u$', usetex=True)
    # plt.axhline(H, linestyle=':', color='black')
    plt.ylim([y_bottom, y_top])
    plt.pause(0.1)

    t = 0.  # Initial time
    # Set up the ODE solver
    r = ode(mol).set_integrator('dopri5', max_step=dt)
    r.set_initial_value(u0[mask], t)
    r.set_f_params(mass, lhs, force)  # Pass additional arguments to dudt

    # Time-stepping
    while r.successful() and abs(FINAL_TIME - r.t) > 1.e-8:

        if r.t + dt >= FINAL_TIME: dt = FINAL_TIME - r.t

        r.integrate(r.t + dt)  # Integrate up to the next time step

        # Replot
        plt.cla()
        plt.plot(nodes[:-1], r.y, 'b', label=r'$CFL = $' + str(CFL), lw=lw)
        plt.title('Time = ' + '%.2f' % (r.t + dt) + ' s', usetex=True)
        plt.legend(loc='lower left')
        plt.xlabel(r'$x$', usetex=True)
        plt.ylabel(r'$u$', usetex=True)
        plt.ylim([y_bottom, y_top])
        plt.pause(0.05)

    # r.integrate(FINAL_TIME)
    print(f"Final time of {r.t} seconds reached.")
    u = np.empty_like(u0)
    """ Map solution back for visualisation """
    for e in range(NO_OF_ELEMS):
        for a in range(P1 + 1):
            u[IEN[e, a]] = r.y[lm[a, e]]

    """ Numerical solution """
    plt.cla()
    plt.plot(nodes, u[:NO_OF_NODES[0]], 'bo', label='Numerical', lw=lw, ms=ms, fillstyle='none')

    """ Analytical solution """
    x = np.linspace(DOMAIN_BOUNDARIES[0][0], DOMAIN_BOUNDARIES[1][0], num=101, endpoint=True)
    u_exact = exact(x)

    plt.plot(x, u_exact, 'k-', label='Exact', linewidth=lw)
    plt.legend(loc='best')
    plt.xlabel(r'$x$', usetex=True)
    plt.ylabel(r'$u$', usetex=True)
    plt.xlim([x_left, x_right])
    plt.ylim([y_bottom, y_top])
    plt.title(r'Numerical vs exact solution, ' + r'$T =$ ' + str(FINAL_TIME) + ' seconds')
    # plt.axhline(0, linestyle=':', color='black')
    plt.grid()
    plt.tick_params(axis='both', direction='in')
    # plt.savefig(f'{path}/gaussian.eps', dpi=1000)
    plt.show()

    """ Calculate L2-norm error """
    # dx = cg.elements[0].calculate_elem_vol()
    #
    # # l2err = sp.linalg.norm(psi_exact - psi_A, 2) / sp.linalg.norm(psi_exact, 2)
    # l2err = compute_l2_err(u, u_exact, dx)
    # print(f"L2-norm error: {l2err}")
