"""
This module contains unit tests for the `create_stats_dataframe` function from
the `engine.stats` module. It tests the functionality of creating statistical
dataframes from both NetCDF and CSV files.
"""

import os
import unittest

import eccodes
import numpy as np
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from engine.stats import create_stats_dataframe

TIME_DIM_SIZE = 3
HOR_DIM_SIZE = 100
HEIGHT_DIM_SIZE = 5

# `reduced_rotated_gg_sfc_grib2.tmpl` is a fixed-grid GRIB2 sample template
# shipped with eccodes; its grid definition fixes the number of horizontal
# points to this value.
HORIZONTAL_DIM_GRIB_SIZE = 6114


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


class TestStatsNetcdf(unittest.TestCase):
    """
    Unit test class for validating statistical calculations from NetCDF files.

    This class tests the accuracy of statistical calculations (mean, max, min)
    performed on data extracted from NetCDF files.
    It ensures that the statistics DataFrame produced from the NetCDF data
    matches expected values.
    """

    nc_file_name = "test_stats.nc"
    nc_file_glob = "test_s*.nc"
    stats_file_names = "test_stats.csv"

    def setUp(self):
        data = initialize_dummy_netcdf_file(self.nc_file_name)

        data.createVariable("v1", np.float64, dimensions=("t", "z", "x"))
        data.variables["v1"][:] = np.ones(
            (TIME_DIM_SIZE, HEIGHT_DIM_SIZE, HOR_DIM_SIZE)
        )
        data.variables["v1"][:, :, 0] = 0
        data.variables["v1"][:, :, -1] = 2

        data.createVariable("v2", np.float64, dimensions=("t", "x"), fill_value=42)
        data.variables["v2"][:] = np.ones((TIME_DIM_SIZE, HOR_DIM_SIZE)) * 2
        data.variables["v2"][:, 0] = 1
        data.variables["v2"][:, 1] = 42  # shall be ignored in max-statistic
        data.variables["v2"][:, -1] = 3

        data.createVariable("v3", np.float64, dimensions=("t", "x"))
        data.variables["v3"][:] = np.ones((TIME_DIM_SIZE, HOR_DIM_SIZE)) * 3
        data.variables["v3"][:, 0] = 2
        data.variables["v3"][:, -1] = 4

        data.close()

    def tear_down(self):
        os.remove(self.nc_file_name)
        os.remove(self.stats_file_names)

    def test_stats(self):
        file_specification = {
            "Test data": {
                "format": "netcdf",
                "time_dim": "t",
                "horizontal_dims": ["x"],
                "fill_value_key": "_FillValue",  # should be the name for fill_value
            },
        }

        df = create_stats_dataframe(
            input_dir=".",
            file_id=[["Test data", self.nc_file_glob]],
            stats_file_name=self.stats_file_names,
            file_specification=file_specification,
        )

        # check that the mean/max/min are correct
        expected = np.array(
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
            np.array_equal(df.values, expected),
            f"stats dataframe incorrect. Difference:\n{df.values == expected}",
        )


def add_variable_to_grib(filename, dict_data, type_of_level="surface", level=0):
    """
    Write one GRIB2 message per (shortName, values) pair in `dict_data` to
    `filename`, using eccodes' `reduced_rotated_gg_sfc_grib2.tmpl` sample as a
    starting point. `centre` is set to "lssw" (MeteoSwiss) so that COSMO/ICON
    local-table definitions (e.g. renaming "t" at surface level to "T_G")
    apply, exercising the same definitions path used for real ICON output.
    """
    with open(filename, "wb") as f_out:
        for short_name, values in dict_data.items():
            gid = eccodes.codes_grib_new_from_samples(
                "reduced_rotated_gg_sfc_grib2.tmpl"
            )
            eccodes.codes_set(gid, "edition", 2)
            eccodes.codes_set(gid, "centre", "lssw")
            eccodes.codes_set(gid, "dataDate", 20230913)
            eccodes.codes_set(gid, "dataTime", 0)
            eccodes.codes_set(gid, "stepRange", 0)
            eccodes.codes_set(gid, "typeOfLevel", type_of_level)
            eccodes.codes_set(gid, "level", level)
            eccodes.codes_set(gid, "shortName", short_name)
            eccodes.codes_set_values(gid, values)
            eccodes.codes_write(gid, f_out)
            eccodes.codes_release(gid)


def test_stats_grib(tmp_path):
    """
    Validates statistical calculations from a GRIB file.

    Mirrors TestStatsNetcdf, but the input is a synthetic single-level GRIB2
    file built directly via the eccodes bindings.
    """
    array_t = np.ones(HORIZONTAL_DIM_GRIB_SIZE)
    array_t[0] = 0
    array_t[-1] = 2

    array_pres = np.ones(HORIZONTAL_DIM_GRIB_SIZE) * 3
    array_pres[0] = 2
    array_pres[-1] = 4

    grib_file_name = "test_stats.grib"
    # shortName "t" resolves to "T_G" (ground temperature) via the COSMO/ICON
    # local definitions when typeOfLevel is "surface".
    add_variable_to_grib(tmp_path / grib_file_name, {"t": array_pres, "v": array_t})

    file_specification = {
        "Test data": {
            "format": "grib",
            "time_dim": "step",
            "horizontal_dims": ["values"],
            "var_excl": [],
            "fill_value_key": "_FillValue",
        },
    }

    df = create_stats_dataframe(
        input_dir=str(tmp_path),
        file_id=[["Test data", grib_file_name]],
        stats_file_name=str(tmp_path / "test_stats_grib.csv"),
        file_specification=file_specification,
    )

    # check that the mean/max/min are correct, for "T_G" then "V"
    expected = np.array(
        [
            [3.0, 4.0, 2.0],
            [1.0, 2.0, 0.0],
        ]
    )

    assert np.array_equal(
        df.values, expected
    ), f"stats dataframe incorrect. Difference:\n{df.values == expected}"


def test_stats_grib_multiple_level_types(tmp_path):
    """
    Validates that a GRIB file mixing fields that don't share one shape
    (different typeOfLevel / number of levels) is correctly split into
    per-shape groups before conversion, rather than erroring out or silently
    dropping fields.
    """
    grib_file_name = "test_stats_multilevel.grib"
    grib_path = tmp_path / grib_file_name

    surface_values = np.ones(HORIZONTAL_DIM_GRIB_SIZE)
    surface_values[0] = 0
    surface_values[-1] = 2
    add_variable_to_grib(grib_path, {"t": surface_values}, type_of_level="surface")

    # append 3 model-level fields for the same shortName to the same file
    with open(grib_path, "ab") as f_out:
        for level in (1, 2, 3):
            values = np.ones(HORIZONTAL_DIM_GRIB_SIZE) * level
            gid = eccodes.codes_grib_new_from_samples(
                "reduced_rotated_gg_sfc_grib2.tmpl"
            )
            eccodes.codes_set(gid, "edition", 2)
            eccodes.codes_set(gid, "centre", "lssw")
            eccodes.codes_set(gid, "dataDate", 20230913)
            eccodes.codes_set(gid, "dataTime", 0)
            eccodes.codes_set(gid, "stepRange", 0)
            eccodes.codes_set(gid, "typeOfLevel", "hybrid")
            eccodes.codes_set(gid, "level", level)
            eccodes.codes_set(gid, "shortName", "t")
            eccodes.codes_set_values(gid, values)
            eccodes.codes_write(gid, f_out)
            eccodes.codes_release(gid)

    file_specification = {
        "Test data": {
            "format": "grib",
            "time_dim": "step",
            "horizontal_dims": ["values"],
            "var_excl": [],
        },
    }

    df = create_stats_dataframe(
        input_dir=str(tmp_path),
        file_id=[["Test data", grib_file_name]],
        stats_file_name=str(tmp_path / "test_stats_multilevel.csv"),
        file_specification=file_specification,
    )

    # one row for the surface field (height -1) and one row per model
    # level (height 1, 2, 3)
    expected = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [1.0, 2.0, 0.0],
        ]
    )

    assert np.array_equal(
        df.values, expected
    ), f"stats dataframe incorrect. Difference:\n{df.values == expected}"


class TestStatsCsv(unittest.TestCase):
    """
    Test suite for validating statistical calculations and CSV file handling.

    This class contains unit tests for creating and validating statistics from a
    CSV file.
    The primary focus is on ensuring that the statistics calculated from the
    input data match the expected values.
    The CSV file used for testing is created and cleaned up during the test
    lifecycle.
    """

    dat_file_name = "test_stats_csv.dat"
    stats_file_name = "test_stats_csv.csv"

    def setUp(self):
        lines = (
            "time v1  v2 v3 v4 v5",
            "10   1.4 15 16 17 18",
            "20   2.4 25 26 27 28",
            "30   3.4 35 36 37 38",
        )
        with open(self.dat_file_name, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def tear_down(self):
        os.remove(self.dat_file_name)
        os.remove(self.stats_file_name)

    def test_stats(self):
        file_specification = {
            "Test data": {
                "format": "csv",
                "parser_args": {
                    "delimiter": "\\s+",
                    "header": 0,
                    "index_col": 0,
                },
            },
        }

        df = create_stats_dataframe(
            input_dir=".",
            file_id=[["Test data", self.dat_file_name]],
            stats_file_name=self.stats_file_name,
            file_specification=file_specification,
        )

        # check that the mean/max/min are correct (i.e. the same as in CSV)
        expected = np.array(
            [
                [1.4, 1.4, 1.4, 2.4, 2.4, 2.4, 3.4, 3.4, 3.4],
                [15, 15, 15, 25, 25, 25, 35, 35, 35],
                [16, 16, 16, 26, 26, 26, 36, 36, 36],
                [17, 17, 17, 27, 27, 27, 37, 37, 37],
                [18, 18, 18, 28, 28, 28, 38, 38, 38],
            ],
        )

        self.assertTrue(
            np.array_equal(df.values, expected),
            f"stats dataframe incorrect. Difference:\n{df.values == expected}",
        )


if __name__ == "__main__":
    unittest.main()
