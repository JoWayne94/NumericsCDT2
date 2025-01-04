"""
2d unsteady ADR equation on a UK grid supplemented with wind speed and direction data

Author: JWT
"""

# Import third-party libraries
import imageio
from scipy.integrate import ode

# Import user settings
from setup import *
from velocity_field import *

# Configure system path
path = os.path.dirname(__file__)

if (sys.platform[:3] == 'win') or (sys.platform[:3] == 'Win'):
    sys.path.append(os.path.abspath(os.path.join(path, '..\..\..')))
else:
    sys.path.append(os.path.abspath(os.path.join(path, '../../..')))

# Import mesh generator
from mesh.meshGen import *

# Import equation types
from src.solvers.Helmholtz import *
from src.solvers.ADR import *


def source1(coords):
    return 1.

def source2(coords, std=1000.):
    return np.exp(-1. / (2. * std ** 2) * ((coords[0] - soton[0]) ** 2 + (coords[1] - soton[1]) ** 2))

def source_t(time, amp=1., mid=0.5 * SOTON_FIRE, spread=8. / 3.):
    std = mid / spread
    return amp * np.exp(-1. / (2. * std ** 2) * ((time - mid) ** 2))

def custom_mol(t, solution, m, lap, f, t_func=lambda time: 1.):
    """
    @:brief Function to compute the time derivative of u with variable stiffness matrix

    :param t:        Current time.
    :param solution: Current solution vector.
    :param m:        Mass matrix (numpy array).
    :param lap:      Global Laplacian matrix (numpy array).
    :param f:        Force vector (numpy array).
    :param t_func:   Source term scaling as a function of time.

    :return:      Time derivative, du/dt.
    """
    # current_time = add_seconds_to_time(int(t))
    # # Interpolate wind speed and direction at the current time-step
    # wind_s, wind_d = interpolate_wind_space_time(data, all_t, towns, nodes, current_time)
    #
    # # Convert wind direction (degrees) to vector components
    # vel_x = wind_s * np.cos(np.radians(270. - wind_d))  # x-component
    # vel_y = wind_s * np.sin(np.radians(270. - wind_d))  # y-component
    #
    # stn = sparse.lil_matrix((n_eq, n_eq))
    #
    # for elem in range(NO_OF_ELEMS):
    #
    #     s_e = cg.elements[elem].stiffness_matrix
    #     n_nodes = s_e[0].shape[0]  # Number of nodes (DOFs) in this element
    #
    #     for a in range(n_nodes):
    #         A = lm[a, elem]
    #         for b in range(n_nodes):
    #             B = lm[b, elem]
    #             if (A >= 0) and (B >= 0):
    #
    #                 """ Elemental stiffness summation """
    #                 stn[A, B] += vel_x[IEN[elem, b]] * s_e[0][a, b]
    #                 stn[A, B] += vel_y[IEN[elem, b]] * s_e[1][a, b]
    #
    # stn = sparse.csr_matrix(stn)

    l = lap + stn
    dudt = sparse.linalg.spsolve(m, t_func(t) * f - l @ solution)

    return dudt


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
    lw = 1.5
    ms = 5.
    line_styles = ['ro', 'gv', 'b<', 'cx', 'k*', 'mh',
                   'rD', 'g^', 'b>', 'c+', 'ks', 'mH',
                   'r1']
    plt.rc('text', usetex=True)

    """ Problem definition """
    town_names = ['soton', 'reading', 'middle_wallop',
                  'raf_odiham', 'heathrow', 'boscombe_down',
                  'bournemouth', 'raf_benson', 'raf_lyneham',
                  'brighton', 'raf_northolt', 'london',
                  'gatwick']
    towns = np.array([[442365., 115483.], [473993., 171625.], [429348.2, 137197.5],
                      [473907.45, 148862.27], [506947.31, 176515.46], [417792.84, 139836.18],
                      [411200.04, 97837.68], [462697.24, 191226.9], [400558.88, 178482.88],
                      [519999.5, 105384.73], [509755.61, 184981.51], [530550.75, 181534.99],
                      [526676.68, 140311.96]])
    source = source2
    soton = towns[0]
    reading = towns[1]

    if USER_INPUT:
        nodes = np.loadtxt(sys.path[-1] + f'/mesh/las_grids/las_nodes_{INPUT_NAME}k.txt')
        IEN = np.loadtxt(sys.path[-1] + f'/mesh/las_grids/las_IEN_{INPUT_NAME}k.txt', dtype=np.int64)
        boundary_nodes = np.loadtxt(sys.path[-1] + f'/mesh/las_grids/las_bdry_{INPUT_NAME}k.txt',
                                    dtype=np.int64)

        dirichlet_boundary = np.where(nodes[boundary_nodes, 1] <= 150000.)[0]

        # Make all boundary points Dirichlet
        ID = np.zeros(len(nodes), dtype=np.int64)
        boundaries = dict()  # hold the boundary values
        n_eq = 0
        for n in range(len(nodes[:, 1])):
            if n in dirichlet_boundary:
                ID[n] = -1
                boundaries[n] = 0.  # Dirichlet BC
            else:
                ID[n] = n_eq
                n_eq += 1
        mat_dim = np.max(ID) + 1
        NO_OF_ELEMS = IEN.shape[0]
        N_nodes = nodes.shape[0]
        NO_OF_DIMENSIONS = nodes.shape[1]
        # Location matrix
        lm = np.zeros_like(IEN.T)
        for e in range(NO_OF_ELEMS):
            for a in range(IEN[0].shape[0]):
                lm[a, e] = ID[IEN[e, a]]

        boundary_map = np.append(boundary_nodes, boundary_nodes[0])
        plot_uk_map(nodes, boundary_map, towns, town_names, line_styles, ms, lw, True)
        # plot_uk_mesh(nodes, dirichlet_boundary, IEN, lw)

    else:
        nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                           BOUNDARY_CONDITIONS_TYPE,
                                                           BOUNDARY_CONDITIONS_VALUES, ELEM_SHAPE)

    # print(f"nodes:\n{nodes}")
    # print(f"IEN:\n{IEN}")
    # print(f"ID:\n{ID}")
    # print(f"boundaries:\n{boundaries}")
    # print(f"lm:\n{lm}")
    files = [sys.path[-1] + f'/data/{i}/data.csv' for i in town_names]
    # Load and align wind data from CSV files
    data, all_t = load_and_align_data(files, common_time_interval="20min")

    """ Mesh construction """
    cg = Mesh(nodes, IEN, ID, boundaries, NO_OF_DIMENSIONS, ELEM_SHAPE)
    cg.construct_elements(NO_OF_ELEMS, NO_OF_VARIABLES, source, P1, P2)

    laplacian, _, force, mass, u, mask = solve(cg, lm, NO_OF_ELEMS, P1, P2, VEL_FIELD, KAPPA)
    # print(f"Solutions:\n{u}")
    # print(f"Global Laplacian matrix:\n{lhs}")
    # print(f"Global force vector:\n{rhs}")

    """ Initial conditions """
    # Initial datum/conditions defined with a numpy array
    u0 = np.zeros_like(u)

    """ Visualisation """
    x_left = np.min(nodes[:, 0])
    x_right = np.max(nodes[:, 0])
    y_bottom = np.min(nodes[:, 1])
    y_top = np.max(nodes[:, 1])
    # Directory to save frames
    frame_dir = 'frames'
    os.makedirs(frame_dir, exist_ok=True)
    # Initialize empty list to store frame filenames
    frame_files = []

    """ Time-step size """
    h = abs(nodes[1, 0] - nodes[0, 0])  # very rough estimate
    mag_v = np.sqrt(VEL_FIELD[0] ** 2 + VEL_FIELD[1] ** 2)
    dt_a = CFL * h / mag_v
    dt_d = (h ** 2) / (4. * np.abs(KAPPA))
    print(f"Time-step size limited by advection: {dt_a}")
    print(f"Time-step size due to diffusion restriction: {dt_d}")
    max_dt = min(dt_a, dt_d)

    q_time = datetime.strptime("11:30:00", "%H:%M:%S")

    # Interpolate wind speed and direction at the current time step
    wind_speed, wind_direction = interpolate_wind_space_time(data, all_t, towns, nodes, q_time)

    # Convert wind direction (degrees) to vector components
    vel_x = wind_speed * np.cos(np.radians(270. - wind_direction))  # x-component
    vel_y = wind_speed * np.sin(np.radians(270. - wind_direction))  # y-component

    stn = sparse.lil_matrix((mat_dim, mat_dim))

    for elem in range(NO_OF_ELEMS):

        s_e = cg.elements[elem].stiffness_matrix
        n_nodes = s_e[0].shape[0]  # Number of nodes (DOFs) in this element

        for a in range(n_nodes):
            A = lm[a, elem]
            for b in range(n_nodes):
                B = lm[b, elem]
                if (A >= 0) and (B >= 0):
                    """ Elemental stiffness summation """
                    stn[A, B] += vel_x[IEN[elem, b]] * s_e[0][a, b]
                    stn[A, B] += vel_y[IEN[elem, b]] * s_e[1][a, b]

    stn = sparse.csr_matrix(stn)

    t0 = 0.
    n = 0
    create_gif = False
    # Set up the ODE solver
    u = np.empty_like(u0)
    r = ode(custom_mol).set_integrator('dopri5', max_step=max_dt)
    r.set_initial_value(u0[mask], t0)
    r.set_f_params(mass, laplacian, force, source_t)  # Pass additional arguments to dudt

    dt = 900.
    print(f"Number of time intervals: {int(FINAL_TIME / dt)}.")

    # reading_elem = np.array([0], dtype=np.int32)
    # soton_elem = np.array([0], dtype=np.int32)
    # reading_local_coords = np.zeros((1, 2), dtype=np.float64)
    # soton_local_coords = np.zeros((1, 2), dtype=np.float64)
    # elem_indicator = np.array([0, 0], dtype=np.int32)
    # for elem in range(NO_OF_ELEMS):
    #     reading_local_coords = cg.elements[elem].geoms.geometry.inv_mapping(reading)
    #     soton_local_coords = cg.elements[elem].geoms.geometry.inv_mapping(soton)
    #
    #     if reading_local_coords[0][0] >= 0. and reading_local_coords[0][1] >= 0. and np.sum(
    #             reading_local_coords[0]) < np.sqrt(2.):
    #         reading_elem[0] = elem
    #         elem_indicator[0] = 1
    #     if soton_local_coords[0][0] >= 0. and soton_local_coords[0][1] >= 0. and np.sum(
    #             soton_local_coords[0]) < np.sqrt(2.):
    #         soton_elem[0] = elem
    #         elem_indicator[1] = 1
    #
    #     if elem_indicator[0] == 1 and elem_indicator[1] == 1:
    #         break

    reading_values = []
    soton_values = []

    # Time-stepping
    while r.successful() and abs(FINAL_TIME - r.t) > 1.e-8:

        if r.t + dt >= FINAL_TIME: dt = FINAL_TIME - r.t

        r.integrate(r.t + dt)  # Integrate up to the next time step
    #
    #     mapping(NO_OF_ELEMS, IEN, lm, r.y, u)
    #
    #     plot_uk_solutions(nodes[:, 0], nodes[:, 1], u, IEN, x_left, x_right, y_bottom, y_top,
    #                       r'Time = ' + '%.2f' % r.t + ' s', ms, soton, reading, show=False
    #                       )

    #     # u /= np.max(u)
    #
    #     reading_u = (u[IEN[reading_elem[0], 0]] * (1. - reading_local_coords[0][0] - reading_local_coords[0][1]) +
    #                  u[IEN[reading_elem[0], 1]] * reading_local_coords[0][0] +
    #                  u[IEN[reading_elem[0], 2]] * reading_local_coords[0][1])
    #
    #     soton_u = (u[IEN[soton_elem[0], 0]] * (1. - reading_local_coords[0][0] - reading_local_coords[0][1]) +
    #                u[IEN[soton_elem[0], 1]] * reading_local_coords[0][0] +
    #                u[IEN[soton_elem[0], 2]] * reading_local_coords[0][1])
    #
    #     reading_values.append(reading_u)
    #     soton_values.append(soton_u)
    #
    #     c_time = add_seconds_to_time(int(r.t))
    #     # Interpolate wind speed and direction at the current time step
    #     w_speed, w_direction = interpolate_wind_space_time(data, all_t, towns, nodes, c_time)
    #
    #     # Convert wind direction (degrees) to vector components
    #     v_x = w_speed * np.cos(np.radians(270. - w_direction))  # x-component
    #     v_y = w_speed * np.sin(np.radians(270. - w_direction))  # y-component
    #
    #     plot_uk_transient(nodes[:, 0], nodes[:, 1], u, IEN, x_left, x_right, y_bottom, y_top,
    #                       r'Time = ' + '%.2f' % r.t + ' s', ms,
    #                       frame_dir, frame_files, n, create_gif, soton, reading, None, None,
    #                       v_x, v_y)
    #
    #     n += 1

    # Create the GIF
    if create_gif:
        with imageio.get_writer(f'las_{INPUT_NAME}k_data.gif', mode='I', duration=0.004) as writer:
            for filename in frame_files:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Optional: Clean up the frame files after GIF creation
        for filename in frame_files:
            os.remove(filename)

    # r.integrate(FINAL_TIME)
    print(f"Final time of {r.t} seconds reached.")
    print(f"Pollutant concentration values at reading:\n{reading_values}.")
    print(f"Pollutant concentration values at soton:\n{soton_values}.")

    mapping(NO_OF_ELEMS, IEN, lm, r.y, u)

    plt.close('all')
    plot_uk_solutions(nodes[:, 0], nodes[:, 1], u, IEN, x_left, x_right, y_bottom, y_top,
                    f"$N_e = {NO_OF_ELEMS}$, unsteady advection-diffusion case", ms, soton, reading)
