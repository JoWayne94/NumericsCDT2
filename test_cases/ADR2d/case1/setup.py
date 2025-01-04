"""
Solver options and case setup

Author: JWT
"""
import numpy as np


"""======SPACE-TIME DOMAIN======"""
NO_OF_DIMENSIONS = 2
DOMAIN_BOUNDARIES = [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
NO_OF_NODES = [65, 65]
ELEM_SHAPE = "T"
BOUNDARY_CONDITIONS_TYPE = ["Periodic", "Periodic", "Periodic", "Periodic"]
BOUNDARY_CONDITIONS_VALUES = [1., 1., -1., -1.]
FINAL_TIME = 1.

"""=====USER-DEFINED DOMAIN====="""
USER_INPUT = False
INPUT_NAME = ""

"""=======SOLVER OPTIONS========"""
CFL = 0.5
EQUATION_TYPE = "ADR"
VEL_FIELD = [1., 1.]
KAPPA = 0.01
NO_OF_VARIABLES = 1
P1 = 1
P2 = 1
