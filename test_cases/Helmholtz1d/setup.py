"""
Solver options and case setup

Author: JWT
"""
import numpy as np

"""======SPACE-TIME DOMAIN======"""
NO_OF_DIMENSIONS = 1
DOMAIN_BOUNDARIES = [[0.], [1.]]
NO_OF_NODES = [11]
ELEM_SHAPE = "S"
BOUNDARY_CONDITIONS_TYPE = ["Dirichlet", "Neumann"]
BOUNDARY_CONDITIONS_VALUES = [0.1, -0.2]

"""=====USER-DEFINED DOMAIN====="""
USER_INPUT = False
INPUT_NAME = ""

"""=======SOLVER OPTIONS========"""
EQUATION_TYPE = "Helmholtz"
NO_OF_VARIABLES = 1
P1 = 1
