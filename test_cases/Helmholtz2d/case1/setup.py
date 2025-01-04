"""
Solver options and case setup

Author: JWT
"""
import numpy as np

"""======SPACE-TIME DOMAIN======"""
NO_OF_DIMENSIONS = 2
DOMAIN_BOUNDARIES = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
NO_OF_NODES = [5, 5]
ELEM_SHAPE = "T"
BOUNDARY_CONDITIONS_TYPE = ["Dirichlet", "Neumann", "Neumann", "Neumann"]
BOUNDARY_CONDITIONS_VALUES = [0., 0., 0., 0.]

"""=====USER-DEFINED DOMAIN====="""
USER_INPUT = False
INPUT_NAME = ""

"""=======SOLVER OPTIONS========"""
EQUATION_TYPE = "Helmholtz"
NO_OF_VARIABLES = 1
P1 = 1
P2 = 1
