import os
import unittest

import numpy as np
from netCDF4 import Dataset

from engine.stats import create_stats_dataframe

DUMMY_FILE_NAME = "test_stats.nc"

TIME_DIM_SIZE = 3
HOR_DIM_SIZE = 100
HEIGHT_DIM_SIZE = 5


def initialize_dummy_netcdf_file(name):
    data = Dataset(name, "w")

    data.createDimension("t", size=TIME_DIM_SIZE)
    data.createVariable("t", np.float64, dimensions="t")
    data.variables["t"][:] = np.arange(TIME_DIM_SIZE)

    data.createDimension("x", size=HOR_DIM_SIZE)
    data.createVariable("x", np.float64, dimensions="x")
    data.variables["x"][:] = np.arange(HOR_DIM_SIZE)

    data.createDimension("z", size=HEIGHT_DIM_SIZE)
    data.createVariable("z", np.float64, dimensions="z")
    data.variables["z"][:] = np.arange(HEIGHT_DIM_SIZE)

    return data


class TestStats(unittest.TestCase):
    def setUp(self):
        data = initialize_dummy_netcdf_file(DUMMY_FILE_NAME)

        data.createVariable("v1", np.float64, dimensions=("t", "z", "x"))
        data.variables["v1"][:] = np.ones(
            (TIME_DIM_SIZE, HEIGHT_DIM_SIZE, HOR_DIM_SIZE)
        )
        data.variables["v1"][:, :, 0] = 0
        data.variables["v1"][:, :, -1] = 2

        data.createVariable("v2", np.float64, dimensions=("t", "x"))
        data.variables["v2"][:] = np.ones((TIME_DIM_SIZE, HOR_DIM_SIZE)) * 2
        data.variables["v2"][:, 0] = 1
        data.variables["v2"][:, -1] = 3

        data.createVariable("v3", np.float64, dimensions=("t", "x"))
        data.variables["v3"][:] = np.ones((TIME_DIM_SIZE, HOR_DIM_SIZE)) * 3
        data.variables["v3"][:, 0] = 2
        data.variables["v3"][:, -1] = 4

        data.close()

    def TearDown(self):
        os.remove(DUMMY_FILE_NAME)
        os.remove(DUMMY_FILE_NAME.replace(".nc", ".csv"))

    def test_stats(self):
        horizontal_dim = "x"

        df = create_stats_dataframe(
            input_dir=".",
            file_ids=[DUMMY_FILE_NAME.replace(".nc", "")],
            time_dim="t",
            horizontal_dims=[horizontal_dim],
            stats_file_name=DUMMY_FILE_NAME.replace(".nc", ".csv"),
        )

        # check that the min/mean/max are correct
        sol = np.array(
            [
                [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
                [1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0],
                [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0],
                [3.0, 4.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 2.0],
            ]
        )

        self.assertTrue(
            np.array_equal(df.values, sol),
            "stats dataframe incorrect. Difference:\n{}".format(df.values == sol),
        )


if __name__ == "__main__":
    unittest.main()
