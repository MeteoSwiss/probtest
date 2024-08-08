"""
This module contains unittests for verifying the behavior of the `perturb_array`
function and its application to numpy arrays and NetCDF4 files.
It ensures the correctness of array perturbations and validates the precision
and amplitude of the perturbations.
"""

import os
import shutil
import unittest

import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from probtest.engine.perturb import perturb_array

atype = np.float32
AMPLITUDE = atype(1e-14)
ARRAY_DIM = 100


class TestPerturb(unittest.TestCase):
    """
    Unit tests for verifying the functionality of perturbation methods applied
    to arrays and NetCDF datasets.

    This class uses the `unittest` framework to ensure the correctness of
    perturbation operations on both in-memory arrays and NetCDF4 datasets.
    It checks that perturbation functions produce the expected results and
    handle different data types correctly.
    """

    def __init__(self, methodName="runTest"):
        super().__init__(methodName=methodName)
        self.data = [None] * 2

    @classmethod
    def setUpClass(cls):
        test_path = os.path.realpath("test/tmp")
        cls.test_path = test_path
        # create test directory (remake if it exists)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
        os.mkdir(test_path)

    def test_perturb_array(self):
        # create two arrays, perturb one.
        # This is to make sure that we get a copy of the input from perturb
        x1 = np.ones((ARRAY_DIM, ARRAY_DIM), dtype=atype)
        x2 = np.ones((ARRAY_DIM, ARRAY_DIM), dtype=atype)
        x_perturbed = perturb_array(x2, 10, AMPLITUDE)

        # compute some stats and do assertions
        diff1 = np.abs(x1 - x_perturbed) / x1
        diff2 = np.abs(x2 - x_perturbed) / x2
        mean_diff1 = np.mean(diff1)
        mean_diff2 = np.mean(diff2)
        self.assertLess(
            np.max(diff2), AMPLITUDE, msg="perturbation is larger than amplitude!"
        )
        self.assertAlmostEqual(
            mean_diff1,
            AMPLITUDE * 0.5,
            msg="perturbation is most likely too small!",
            delta=AMPLITUDE * 1e-2,
        )
        self.assertAlmostEqual(
            mean_diff2,
            AMPLITUDE * 0.5,
            msg="perturbation did not return a copy of input!",
            delta=AMPLITUDE * 1e-2,
        )

    def test_perturb_nc(self):

        # create two dummy netcdf4 files, one with single, one with double precision
        for i, dt in enumerate([np.float32, np.float64]):
            self.data[i] = Dataset(f"dummy{i}.nc", "w")
            self.data[i].createDimension("x", size=ARRAY_DIM)
            self.data[i].createDimension("y", size=ARRAY_DIM)
            self.data[i].createVariable("z", dt, dimensions=("x", "y"))
            self.data[i].variables["z"][:] = np.ones((ARRAY_DIM, ARRAY_DIM))

        # perturb the double precision file
        zd = self.data[1].variables["z"][:]
        zd_perturb = perturb_array(zd, 10, AMPLITUDE)
        self.data[1].variables["z"][:] = zd_perturb

        # close the data
        for i in range(2):
            self.data[i].close()

        # reopen the files to make sure we get the values form disk
        for i in range(2):
            self.data[i] = Dataset(f"dummy{i}.nc", "r")
        zf = self.data[0].variables["z"][:]
        zd_perturb = self.data[1].variables["z"][:]

        # compute some stats and do assertions
        diff = zf - zd_perturb
        mean_diff = np.mean(np.abs(diff))
        self.assertTrue(zf.dtype == "float32", msg="zf is not float32!")
        self.assertTrue(zd.dtype == "float64", msg="zf is not float64!")
        self.assertLess(
            np.max(diff), AMPLITUDE, msg="perturbation is larger than amplitude!"
        )
        self.assertAlmostEqual(
            mean_diff,
            AMPLITUDE * 0.5,
            msg="perturbation is most likely too small!",
            delta=AMPLITUDE * 1e-2,
        )

        # to really make sure:
        # create a plot of the difference to be read out manually if you are scared
        fig, ax = plt.subplots(1, 1)
        xx, yy = np.meshgrid(np.linspace(0, 1, ARRAY_DIM), np.linspace(0, 1, ARRAY_DIM))
        cs = ax.contourf(xx, yy, diff)
        fig.colorbar(cs, ax=ax)
        fig.savefig(f"{self.test_path}/diff_figure.pdf")


if __name__ == "__main__":
    unittest.main()