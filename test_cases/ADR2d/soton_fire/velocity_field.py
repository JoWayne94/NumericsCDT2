"""
Process wind speed data

Author: JWT
"""

# Import third-party libraries
import sys, os, imageio
import pandas as pd
from scipy.interpolate import griddata, Rbf
from datetime import datetime, timedelta

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
from src.utils.helpers import *


# Function to load data from multiple CSV files
def load_and_align_data(file_paths, common_time_interval="10T"):
    """
    Load wind data from multiple CSV files and align timestamps across files.

    :param file_paths:           List of CSV file paths.
    :param common_time_interval: Time interval to standardise timestamps (e.g., '10T' for 10 minutes).

    :return: A dictionary where each key is the location index and the value is a DataFrame with aligned timestamps.
    """
    aligned_data = {}
    for i, file_path in enumerate(file_paths):
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Convert 'Time' column to datetime
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')

        # Resample data to the common time interval (linear interpolation)
        df.set_index('Time', inplace=True)
        df = df.resample(common_time_interval).interpolate(method='linear').reset_index()

        # Store aligned data
        aligned_data[i] = df

    # Extract all unique timestamps across all datasets
    all_times = sorted(set().union(*[df['Time'] for df in aligned_data.values()]))

    # Normalise timestamps in the data
    all_times = [time.replace(second=0) for time in all_times]

    return aligned_data, all_times

# Function to interpolate wind data to all map nodes
def interpolate_wind_space(aligned_data, locations, points, query_time):
    """
    Interpolates wind speed and direction to all nodes, including points outside the convex hull of locations.

    :param aligned_data: Dictionary of aligned wind data for each location.
    :param locations:    List of lists [[x1, y1], [x2, y2], ...] with coordinates of the CSV file locations.
    :param points:       List of lists [[x1, y1], [x2, y2], ...] for map node coordinates.
    :param query_time:   Specific time (datetime object) for which to interpolate data.

    :return: Interpolated wind speed and direction arrays for all nodes.
    """
    query_time = pd.to_datetime(query_time, format="%H:%M:%S")

    speeds = []
    directions = []
    valid_locations = []

    for i, df in aligned_data.items():
        # Find the row corresponding to the query time
        if query_time in df['Time'].values:
            row = df.loc[df['Time'] == query_time]
            speeds.append(row['Wind speed (m/s)'].values[0])
            directions.append(row['Wind direction (deg)'].values[0])
            valid_locations.append(locations[i])  # Only include valid locations

    if not valid_locations:
        raise ValueError("No data available for the given query time.")

    # Convert valid locations and nodes to numpy arrays
    valid_locations = np.array(valid_locations)
    # points = np.array(points)

    # Perform interpolation (linear) and handle extrapolation (nearest)
    speeds_interp = griddata(valid_locations, speeds, points, method='linear', fill_value=np.nan)
    speeds_extrap = griddata(valid_locations, speeds, points, method='nearest')
    speeds_interp = np.where(np.isnan(speeds_interp), speeds_extrap, speeds_interp)

    directions_interp = griddata(valid_locations, directions, points, method='linear', fill_value=np.nan)
    directions_extrap = griddata(valid_locations, directions, points, method='nearest')
    directions_interp = np.where(np.isnan(directions_interp), directions_extrap, directions_interp)

    return speeds_interp, directions_interp


def interpolate_wind_space_rbf(aligned_data, locations, points, query_time):
    """
    Interpolates wind speed and direction using Radial Basis Functions (RBF) to all nodes.

    :param aligned_data: Dictionary of aligned wind data for each location.
    :param locations:    List of lists [[x1, y1], [x2, y2], ...] with coordinates of the CSV file locations.
    :param points:       List of lists [[x1, y1], [x2, y2], ...] for map node coordinates.
    :param query_time:   Specific time (datetime object) for which to interpolate data.

    :return: Interpolated wind speed and direction arrays for all nodes.
    """
    query_time = pd.to_datetime(query_time, format="%H:%M:%S")

    speeds = []
    directions = []
    valid_locations = []

    for i, df in aligned_data.items():
        if query_time in df['Time'].values:
            row = df.loc[df['Time'] == query_time]
            speeds.append(row['Wind speed (m/s)'].values[0])
            directions.append(row['Wind direction (deg)'].values[0])
            valid_locations.append(locations[i])

    if not valid_locations:
        raise ValueError("No data available for the given query time.")

    # Convert valid locations and nodes to numpy arrays
    valid_locations = np.array(valid_locations)
    points = np.array(points)
    nodes_x, nodes_y = points[:, 0], points[:, 1]
    loc_x, loc_y = valid_locations[:, 0], valid_locations[:, 1]

    # Use RBF for interpolation/extrapolation
    rbf_speed = Rbf(loc_x, loc_y, speeds, function='linear')
    rbf_direction = Rbf(loc_x, loc_y, directions, function='linear')

    speeds_interp = rbf_speed(nodes_x, nodes_y)
    directions_interp = rbf_direction(nodes_x, nodes_y)

    return speeds_interp, directions_interp

# Function to interpolate wind data in space and time
def interpolate_wind_space_time(aligned_data, all_times, locations, points, query_time):
    """
    Interpolates wind speed and direction both spatially and temporally for all nodes.

    :param aligned_data: Dictionary of aligned wind data for each location.
    :param all_times:    All date-times from data
    :param locations:    List of lists [[x1, y1], [x2, y2], ...] with coordinates of the CSV file locations.
    :param points:       List of lists [[x1, y1], [x2, y2], ...] for map node coordinates.
    :param query_time:   Specific time (datetime object) for which to interpolate data.

    :return: Interpolated wind speed and direction arrays for all nodes.
    """
    # Convert query_time to pandas datetime for easy comparison
    query_time = pd.to_datetime(query_time, format="%H:%M:%S")
    # Find the two nearest timestamps around the query time
    before_time = max(time for time in all_times if time <= query_time)
    after_time = min(time for time in all_times if time >= query_time)

    if before_time == after_time:  # Exact match
        return interpolate_wind_space_rbf(aligned_data, locations, points, before_time)

    # Perform temporal interpolation
    t1 = before_time.timestamp()
    t2 = after_time.timestamp()
    tq = query_time.timestamp()

    # Interpolate wind data spatially for the two nearest timestamps
    wind_speed_before, wind_direction_before = interpolate_wind_space_rbf(aligned_data, locations, points, before_time)
    wind_speed_after, wind_direction_after = interpolate_wind_space_rbf(aligned_data, locations, points, after_time)

    # Interpolate in time
    wind_speed_interp = wind_speed_before + (wind_speed_after - wind_speed_before) * ((tq - t1) / (t2 - t1))
    wind_direction_interp = wind_direction_before + (wind_direction_after - wind_direction_before) * (
                            (tq - t1) / (t2 - t1))

    return wind_speed_interp, wind_direction_interp


def add_seconds_to_time(seconds, date_string="06:30"):
    """
    Add seconds to a time string and return the updated time as a string.

    :param date_string: A string representing the time (format: "HH:MM").
    :param seconds:     The number of seconds to add to the time.

    :return: A new time string with the seconds added (format: "HH:MM:SS").
    """
    # Parse the input time string to a datetime object
    time_obj = datetime.strptime(date_string, "%H:%M")

    # Add the seconds using timedelta
    updated_time = time_obj + timedelta(seconds=seconds)

    # Return the updated time as a string in "HH:MM:SS" format
    return updated_time.strftime("%H:%M:%S")


if __name__ == '__main__':
    """
    main()
    """

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
        for n in range(len(nodes)):
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
            for a in range(IEN.shape[1]):
                lm[a, e] = ID[IEN[e, a]]

        boundary_map = np.append(boundary_nodes, boundary_nodes[0])
        plot_uk_map(nodes, boundary_map, towns, town_names, line_styles, ms, lw, False)
        # plot_uk_mesh(nodes, dirichlet_boundary, IEN, lw)

    else:
        nodes, IEN, ID, boundaries, lm, mat_dim = mesh_gen(NO_OF_NODES, DOMAIN_BOUNDARIES,
                                                           BOUNDARY_CONDITIONS_TYPE,
                                                           BOUNDARY_CONDITIONS_VALUES, ELEM_SHAPE)

    files = [sys.path[-1] + f'/data/{i}/data.csv' for i in town_names]
    # Load and align wind data from CSV files
    data, all_t = load_and_align_data(files, common_time_interval="20min")

    # Define the time range
    t_int = 60. * 30.
    start_time = datetime.strptime("06:00:00", "%H:%M:%S")
    end_time = datetime.strptime("16:00:00", "%H:%M:%S")
    query_times = [start_time + timedelta(seconds=t_int * i) for i in
                   range(int((end_time - start_time).total_seconds() / t_int) + 1)]

    qt = datetime.strptime("11:30:00", "%H:%M:%S")

    # Interpolate wind speed and direction at the current time step
    wind_speed, wind_direction = interpolate_wind_space_time(data, all_t, towns, nodes, qt)
    # Convert wind direction (degrees) to vector components
    u = wind_speed * np.cos(np.radians(270. - wind_direction))  # x-component
    v = wind_speed * np.sin(np.radians(270. - wind_direction))  # y-component

    title = f"Wind directions at {qt.strftime('%H:%M:%S')}"

    plot_vel_field(nodes, boundary_map, towns, town_names, u, v, title, line_styles, ms, lw, show=True)

    # n = 0
    # create_gif = True
    #
    # # Directory to save frames
    # frame_dir = 'frames_vel'
    # os.makedirs(frame_dir, exist_ok=True)
    # # Initialize empty list to store frame filenames
    # frame_files = []
    #
    # # Loop over query times and create quiver plots
    # for qt in query_times:
    #     # Interpolate wind speed and direction at the current time step
    #     wind_speed, wind_direction = interpolate_wind_space_time(data, all_t, towns, nodes, qt)
    #     # Convert wind direction (degrees) to vector components
    #     u = wind_speed * np.cos(np.radians(270. - wind_direction))  # x-component
    #     v = wind_speed * np.sin(np.radians(270. - wind_direction))  # y-component
    #
    #     title = f"Wind directions at {qt.strftime('%H:%M:%S')}"
    #
    #     plot_vel_transient(nodes, boundary_map, towns, town_names, u, v, title, line_styles, ms, lw,
    #                        frame_dir, frame_files, n, gif=create_gif)
    #
    #     n += 1
    #
    # # Create the GIF
    # if create_gif:
    #     with imageio.get_writer(f'vel_field.gif', mode='I', duration=0.01) as writer:
    #         for filename in frame_files:
    #             image = imageio.imread(filename)
    #             writer.append_data(image)
    #
    #     # Optional: Clean up the frame files after GIF creation
    #     for filename in frame_files:
    #         os.remove(filename)
