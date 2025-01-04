import unittest, os, sys
import numpy as np

# Configure system path
path = os.path.dirname(__file__)

if (sys.platform[:3] == 'win') or (sys.platform[:3] == 'Win'):
    sys.path.append(os.path.abspath(os.path.join(path, '..\..\..')))
else:
    sys.path.append(os.path.abspath(os.path.join(path, '../../..')))

from src.library.geoms.tri import Triangle as TriGeom
from src.library.elems.tri import Triangle as TriElem


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # Read in the geometries
        """ All geometries to be tested """
        cls.seg = None
        cls.tri = None
        cls.quad = None

    def tearDown(self) -> None:

        pass

    def assertArrayAlmostEqual(self, array1, array2, decimal=7, msg=None):
        """
        Custom assertion to check if two arrays are almost equal element-wise
        using numpy's assert_almost_equal.

        Parameters:
        - array1, array2: Numpy arrays to compare.
        - decimal: Precision (number of decimal places) for the comparison.
        """
        # Use numpy's assert_almost_equal
        try:
            np.testing.assert_almost_equal(array1, array2, decimal=decimal)
        except AssertionError as e:
            self.fail(f"Arrays are not almost equal: {e}" + msg)

    def test_geometry(self):

        standard = {
            "nodes": np.array([[0., 0.],
                               [1., 0.],
                               [0., 1.]]),
            "x": np.array([0.5]),
            "y": np.array([0.5]),
            "ans": np.eye(2)
        }

        translation = {
            "nodes": np.array([[1., 0.],
                               [2., 0.],
                               [1., 1.]]),
            "x": np.array([1.5]),
            "y": np.array([0.5]),
            "ans": np.eye(2)
        }

        scale = {
            "nodes": np.array([[0., 0.],
                               [2., 0.],
                               [0., 2.]]),
            "x": np.array([1.]),
            "y": np.array([1.]),
            "ans": 2. * np.eye(2)
        }

        rotation = {
            "nodes": np.array([[1., 1.],
                               [0., 1.],
                               [1., 0.]]),
            "x": np.array([0.5]),
            "y": np.array([0.5]),
            "ans": -1. * np.eye(2)
        }

        for t in [standard, translation, scale, rotation]:
            TestCase.tri = TriGeom(t["nodes"], np.array([0.5]), np.array([0.5]))
            # add assertion here
            self.assertEqual(TestCase.tri.jacobian.tolist(), [t["ans"].tolist()],
                             f"Jacobian of\n {t['nodes']} failed.")
            self.assertEqual(TestCase.tri.ref_mapping().tolist(), [t["x"].tolist(), t["y"].tolist()],
                             f"Local-to-global coordinates mapping of\n {t['nodes']} failed.")
            self.assertEqual(TestCase.tri.inv_mapping(np.array([t["x"][0], t["y"][0]])).tolist(), [[0.5, 0.5]],
                             f"Global-to-local coordinates mapping of\n {t['nodes']} failed.")

    def test_local_quadrature(self):

        ien = np.array([[0, 1, 2]], dtype=int)
        nodes = np.array([[0., 0.],
                          [1., 0.],
                          [0., 1.]])

        constant = {
            "s": lambda x: 1,
            "ans": 0.5,
        }

        linear_x = {
            "s": lambda x: 6. * x[0],
            "ans": 1.,
        }

        linear_y = {
            "s": lambda x: x[1],
            "ans": 1. / 6.,
        }

        product = {
            "s": lambda x: x[0] * x[1],
            "ans": 1. / 24.,
        }

        for t in [constant, linear_x, linear_y, product]:
            TestCase.tri = TriElem(ien, nodes, None, 1, t["s"], 1, 1)
            self.assertAlmostEqual((TestCase.tri.source.reshape(-1,) @ TestCase.tri.ref_elem.weights)[0], t["ans"], 7,
                             f"Source with answer\n {t['ans']} failed.")

    def test_global_quadrature(self):

        ien = np.array([[0, 1, 2]], dtype=int)

        translation = {
            "nodes": np.array([[1., 0.],
                               [2., 0.],
                               [1., 1.]]),
            "s": [lambda x: 3. * x[0], lambda x: x[0] * x[1]],
            "ans": [2., 5. / 24.],
        }

        scale = {
            "nodes": np.array([[0., 0.],
                               [2., 0.],
                               [0., 2.]]),
            "s": [lambda x: 3 * x[0], lambda x: x[0] * x[1]],
            "ans": [4., 2. / 3.],
        }

        rotation = {
            "nodes": np.array([[1., 1.],
                               [0., 1.],
                               [1., 0.]]),
            "s": [lambda x: 3 * x[0], lambda x: x[0] * x[1]],
            "ans": [1., 5. / 24.],
        }

        for t in [translation, scale, rotation]:
            for i in range(2):
                TestCase.tri = TriElem(ien, t["nodes"], None, 1, t["s"][i], 1, 1)
                global_source = (abs(TestCase.tri.geoms.det_jacobian) * TestCase.tri.mats.basis_matrix.T @
                                 TestCase.tri.source.reshape(-1,) @ TestCase.tri.ref_elem.weights)
                self.assertAlmostEqual(global_source[0], t["ans"][i], 7,
                                 f"Element\n {t['nodes']} with answer\n {t['ans'][i]} failed.")

    def test_element(self):

        ien = np.array([[0, 1, 2]], dtype=int)

        standard = {
            "nodes": np.array([[0., 0.],
                               [1., 0.],
                               [0., 1.]]),
            "S": [lambda x: 1., lambda x: x[0]],
            "f": [1. / 6. * np.array([1., 1., 1.]), 1. / 24. * np.array([1., 2., 1.])],
            "gdm": np.array([[-1., -1.],
                             [1., 0.],
                             [0., 1.]]),
            "mass": 1. / 24. * np.array([[2., 1., 1.],
                                         [1., 2., 1.],
                                         [1., 1., 2.]]),
            "lap": np.array([[1., -0.5, -0.5],
                             [-0.5, 0.5, 0.],
                             [-0.5, 0., 0.5]]),
            "stiff_x": 1. / 6. * np.array([[-1., 1., 0.],
                                           [-1., 1., 0.],
                                           [-1., 1., 0.]]),
            "stiff_y": 1. / 6. * np.array([[-1., 0., 1.],
                                           [-1., 0., 1.],
                                           [-1., 0., 1.]]),
        }

        translation = {
            "nodes": np.array([[1., 0.],
                               [2., 0.],
                               [1., 1.]]),
            "S": [lambda x: 1., lambda x: x[1]],
            "f": [1. / 6. * np.array([1., 1., 1.]), 1. / 24. * np.array([1., 1., 2.])],
            "gdm": np.array([[-1., -1.],
                             [1., 0.],
                             [0., 1.]]),
            "mass": 1. / 24. * np.array([[2., 1., 1.],
                                         [1., 2., 1.],
                                         [1., 1., 2.]]),
            "lap": np.array([[1., -0.5, -0.5],
                             [-0.5, 0.5, 0.],
                             [-0.5, 0., 0.5]]),
            "stiff_x": 1. / 6. * np.array([[-1., 1., 0.],
                                           [-1., 1., 0.],
                                           [-1., 1., 0.]]),
            "stiff_y": 1. / 6. * np.array([[-1., 0., 1.],
                                           [-1., 0., 1.],
                                           [-1., 0., 1.]]),
        }

        scale = {
            "nodes": np.array([[0., 0.],
                               [2., 0.],
                               [0., 2.]]),
            "S": [lambda x: 1., lambda x: x[0]],
            "f": [2. / 3. * np.array([1., 1., 1.]), 1. / 3. * np.array([1., 2., 1.])],
            "gdm": np.array([[-0.5, -0.5],
                             [0.5, 0.],
                             [0., 0.5]]),
            "mass": 1. / 6. * np.array([[2., 1., 1.],
                                        [1., 2., 1.],
                                        [1., 1., 2.]]),
            "lap": np.array([[1., -0.5, -0.5],
                             [-0.5, 0.5, 0.],
                             [-0.5, 0., 0.5]]),
            "stiff_x": 1. / 3. * np.array([[-1., 1., 0.],
                                           [-1., 1., 0.],
                                           [-1., 1., 0.]]),
            "stiff_y": 1. / 3. * np.array([[-1., 0., 1.],
                                           [-1., 0., 1.],
                                           [-1., 0., 1.]]),
        }

        rotation = {
            "nodes": np.array([[1., 1.],
                               [0., 1.],
                               [1., 0.]]),
            "S": [lambda x: 1., lambda x: x[1]],
            "f": [1. / 6. * np.array([1., 1., 1.]), 1. / 4. * np.array([1. / 2., 1. / 2., 1. / 3.])],
            "gdm": np.array([[1., 1.],
                             [-1., 0.],
                             [0., -1.]]),
            "mass": 1. / 24. * np.array([[2., 1., 1.],
                                         [1., 2., 1.],
                                         [1., 1., 2.]]),
            "lap": np.array([[1., -0.5, -0.5],
                             [-0.5, 0.5, 0.],
                             [-0.5, 0., 0.5]]),
            "stiff_x": 1. / 6. * np.array([[1., -1., 0.],
                                           [1., -1., 0.],
                                           [1., -1., 0.]]),
            "stiff_y": 1. / 6. * np.array([[1., 0., -1.],
                                           [1., 0., -1.],
                                           [1., 0., -1.]]),
        }

        for t in [standard, translation, scale, rotation]:
            TestCase.tri = TriElem(ien, t["nodes"], None, 1, t["S"][0], 1, 1)
            global_deriv_matrix = TestCase.tri.geoms.inv_jacobian[0][0][0] * TestCase.tri.mats.deriv_matrix[0]

            self.assertEqual(global_deriv_matrix.tolist(), t["gdm"].tolist(),
                             f"Global shape function derivatives of\n {t['nodes']} failed.")
            self.assertArrayAlmostEqual(TestCase.tri.mass_matrix, t["mass"], 7,
                             f"Mass matrix operator of\n {t['nodes']} failed.")
            self.assertEqual(TestCase.tri.laplacian_matrix.tolist(), t["lap"].tolist(),
                             f"Laplacian matrix operator of\n {t['nodes']} failed.")
            self.assertArrayAlmostEqual(TestCase.tri.stiffness_matrix[0], t["stiff_x"], 7,
                             f"Stiffness matrix in the x-direction of\n {t['nodes']} failed.")
            self.assertArrayAlmostEqual(TestCase.tri.stiffness_matrix[1], t["stiff_y"], 7,
                             f"Stiffness matrix in the y-direction of\n {t['nodes']} failed.")
            self.assertArrayAlmostEqual(TestCase.tri.force_vector, t["f"][0], 7,
                             f"Force vector (1) of\n {t['nodes']} failed.")

            TestCase.tri = TriElem(ien, t["nodes"], None, 1, t["S"][1], 1, 1)

            self.assertArrayAlmostEqual(TestCase.tri.force_vector, t["f"][1], 7,
                            f"Force vector (2) of\n {t['nodes']} failed.")


if __name__ == '__main__':
    unittest.main()
