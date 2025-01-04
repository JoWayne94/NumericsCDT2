"""
Global matrix assembly

Author: JWT
"""
import numpy as np
from src.library.mesh.mesh import Mesh


def force(nodes_e, s):
    """
    @:brief Computes the elemental force vector

    :param nodes_e: Nodes coordinate array
    :param s:       Source evaluation at left and right nodes
    :return:        f_e
    """
    return ((nodes_e[1] - nodes_e[0]) / 6.) * np.array([2 * s(nodes_e[0]) + s(nodes_e[1]),
                                                        s(nodes_e[0]) + 2 * s(nodes_e[1])])


