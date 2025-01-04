"""
Contains methods of each basis as shape functions in reference space

Author: JWT
"""
from src.utils.helpers import *


def Lagrange1d(xi, nodes):
    """
    @:brief             Get nodal Lagrange polynomials in one-dimension
    :param xi:          Coordinates in the xi-direction, [-1, 1]. Can be evenly spaced but often defined as Gauss points
    :param nodes:       Node coordinates as solution nodes where all else but one polynomial has a non-zero value
    :return:            Lagrange polynomial values given points and nodes
    """
    xi = xi.reshape(-1, )
    nodes = nodes.reshape(-1, )
    n_nodes = nodes.shape[0]
    tmp = np.ones(n_nodes, bool)
    values = np.ones((xi.shape[0], n_nodes))
    xi.shape = -1, 1

    # q_coord_prod = np.repeat(xi.reshape(-1, 1), n_nodes - 1, axis=1)
    #
    # # Loop through number of nodes and evaluate basis values for each polynomial
    # for j in range(n_nodes):
    #     tmp[j] = False
    #     nodes_prod = np.tile(nodes[tmp].reshape(-1), (xi.shape[0], 1))
    #
    #     # np.prod should be evaluated for n_nodes - 1 no. of terms, nx no. of times,
    #     # if 5 nodes and 6 points, something like
    #     # ([[u u u u], [u u u u], [u u u u], [u u u u], [u u u u], [u u u u]])
    #     # each array is (xi - nodes[tmp]) / (nodes[j] - nodes[tmp])
    #     # https://numpy.org/doc/stable/reference/generated/numpy.prod.html
    #
    #     values[:, j] = np.prod((q_coord_prod - nodes_prod) / (nodes[j] - nodes_prod), axis=1)
    #     tmp[j] = True

    for j in range(n_nodes):
        tmp[j] = False
        values[:, j] = np.prod((xi - nodes[tmp]) / (nodes[j] - nodes[tmp]), axis=1)
        tmp[j] = True

    return values

def Lagrange1dGrad(xi, nodes):
    """
    @:brief             Get nodal Lagrange polynomial derivatives in one-dimension
    :param xi:          Coordinates in the xi-direction, [-1, 1]
    :param nodes:       Node coordinates as solution nodes where all else but one polynomial has a non-zero value
    :return:            Lagrange polynomial derivatives given nodes
    """
    xi = xi.reshape(-1, )
    nodes = nodes.reshape(-1,)
    n_nodes = nodes.shape[0]
    tmp = np.ones(n_nodes, bool)
    gradients = np.zeros((xi.shape[0], n_nodes, 1))
    xi.shape = -1, 1

    # q_coord_prod = np.repeat(xi.reshape(-1, 1), n_nodes - 1, axis=1)

    for j in range(n_nodes):

        tmp[j] = False

        for i in range(n_nodes):
            if i == j:
                continue

            tmp[i] = False
            if n_nodes > 2:
                # nodes_prod = np.tile(nodes[tmp].reshape(-1), (xi.shape[0], 1))
                #
                # gradients[:, j, :] += np.prod((q_coord_prod - nodes_prod) /
                #                               (nodes[j] - nodes_prod), axis=1).reshape(-1, 1) / (nodes[j] - nodes_prod)
                gradients[:, j, :] += (np.prod((xi - nodes[tmp])/(nodes[j] - nodes[tmp]), axis=1).reshape(-1, 1) /
                                       (nodes[j] - nodes[i]))

            else:
                gradients[:, j, :] += 1. / (nodes[j] - nodes[i])

            tmp[i] = True
        tmp[j] = True

    return gradients

def Lagrange2dQuad(xi, x_nodes, y_nodes):
    """
    @:brief         Get nodal Lagrange polynomials in two-dimensions using tensor products
    :param xi:      Coordinates in the x and y-direction within [-1, 1]x[-1, 1] space
    :param x_nodes: Node coordinates in the x-direction
    :param y_nodes: Node coordinates in the y-direction
    :return:        Lagrange tensorial base values and gradients given points and nodes
    """
    n_points = xi.shape[0]
    nx_nodes = x_nodes.shape[0]
    ny_nodes = y_nodes.shape[0]
    values = np.zeros((n_points, nx_nodes * ny_nodes))
    gradients = np.zeros((n_points, nx_nodes * ny_nodes, 2))

    # Get one-dimensional basis values first
    value_x = Lagrange1d(xi[:, 0].reshape(-1, 1), x_nodes)
    value_y = Lagrange1d(xi[:, 1].reshape(-1, 1), y_nodes)

    # Get one-dimensional gradient values first
    grad_x = Lagrange1dGrad(xi[:, 0].reshape(-1, 1), x_nodes)
    grad_y = Lagrange1dGrad(xi[:, 1].reshape(-1, 1), y_nodes)

    # Tensor products to get two-dimensional bases values
    for i in range(n_points):
        values[i, :] = np.reshape(np.outer(value_x[i, :], value_y[i, :]), (-1,), 'F')
        gradients[i, :, 0] = np.reshape(np.outer(grad_x[i, :, 0], value_y[i, :]), (-1,), 'F')
        gradients[i, :, 1] = np.reshape(np.outer(value_x[i, :], grad_y[i, :, 0]), (-1,), 'F')

    return values, gradients

def Lagrange2dTri(xi, nodes, p):
    """
    @:brief         Get nodal Lagrange polynomials in two-dimensions for reference triangle
    :param xi:      Reference coordinates in the xi_1 and xi_2-direction within reference space
    :param nodes:   Node coordinates
    :param p:       Polynomial order
    :return:        Lagrange shape functions and derivatives w.r.t. xi in reference triangle
    """
    n_points = nodes.shape[0]
    values = np.zeros((n_points, n_points))
    gradients = np.zeros((n_points, n_points, 2))
    grad_dir = np.zeros((xi.shape[0], n_points, 3))

    alpha = np.round(p * nodes)
    alpha = np.c_[(p * np.ones(n_points) - np.sum(alpha, axis=1), alpha)]
    l = np.c_[(np.ones(xi.shape[0]) - np.sum(xi, axis=1)), xi]

    for i in range(n_points):
        values[:, i] = get_tri_area_coordinates(p, alpha[i], l)
        grad_dir[:, i, :] = get_tri_grad_area_coordinates(p, alpha[i], l)

    gradients[:, :, 0] = grad_dir[:, :, 1] - grad_dir[:, :, 0]
    gradients[:, :, 1] = grad_dir[:, :, 2] - grad_dir[:, :, 0]

    return values, gradients
