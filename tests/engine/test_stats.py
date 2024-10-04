"""
This module contains unit tests for the `create_stats_dataframe` function from
the `engine.stats` module. It tests the functionality of creating statistical
dataframes from both NetCDF and CSV files.
"""

import eccodes
import numpy as np
import pytest
from netCDF4 import Dataset  # pylint: disable=no-name-in-module

from engine.stats import create_stats_dataframe

TIME_DIM_SIZE = 3
HOR_DIM_SIZE = 100
HEIGHT_DIM_SIZE = 5

TIME_DIM_GRIB_SIZE = 1
STEP_DIM_SIZE = 1
HEIGHT_DIM_GRIB_SIZE = 1
HORIZONTAL_DIM_GRIB_SIZE = 6114

GRIB_FILENAME = "test_stats_grib.grib"
STATS_FILE_NAMES = "test_stats.csv"
NC_FILE_NAME = "test_stats.nc"
NC_FILE_GLOB = "test_s*.nc"


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


def add_variable_to_grib(filename, dict_data):
    with open(filename, "wb") as f_out:
        for short_name in list(dict_data.keys()):
            gid = eccodes.codes_grib_new_from_samples(
                "reduced_rotated_gg_sfc_grib2.tmpl"
            )
            eccodes.codes_set(gid, "edition", 2)
            eccodes.codes_set(gid, "centre", "lssw")
            eccodes.codes_set(gid, "dataDate", 20230913)
            eccodes.codes_set(gid, "dataTime", 0)
            eccodes.codes_set(gid, "stepRange", 0)
            eccodes.codes_set(gid, "typeOfLevel", "surface")
            eccodes.codes_set(gid, "level", 0)
            eccodes.codes_set(gid, "shortName", short_name)
            eccodes.codes_set_values(gid, dict_data[short_name])
            eccodes.codes_write(gid, f_out)
            eccodes.codes_release(gid)


@pytest.fixture
def setup_grib_file(tmp_path):
    array_t = np.ones(
        (
            TIME_DIM_GRIB_SIZE,
            STEP_DIM_SIZE,
            HEIGHT_DIM_GRIB_SIZE,
            HORIZONTAL_DIM_GRIB_SIZE,
        )
    )
    array_t[:, :, :, 0] = 0
    array_t[:, :, :, -1] = 2

    array_pres = (
        np.ones(
            (
                TIME_DIM_GRIB_SIZE,
                STEP_DIM_SIZE,
                HEIGHT_DIM_GRIB_SIZE,
                HORIZONTAL_DIM_GRIB_SIZE,
            )
        )
        * 3
    )
    array_pres[:, :, :, 0] = 2
    array_pres[:, :, :, -1] = 4

    dict_data = {"t": array_pres, "v": array_t}

    # This would be where your grib file is created
    add_variable_to_grib(tmp_path / GRIB_FILENAME, dict_data)


@pytest.mark.usefixtures("setup_grib_file")
def test_stats_grib(tmp_path):
    file_specification = {
        "Test data": {
            "format": "grib",
            "time_dim": "step",
            "horizontal_dims": ["values"],
            "var_excl": [],
            "fill_value_key": "_FillValue",  # This should be the name for fill_value.
        },
    }

    df = create_stats_dataframe(
        input_dir=str(tmp_path),
        file_id=[["Test data", GRIB_FILENAME]],
        stats_file_name=tmp_path / STATS_FILE_NAMES,
        file_specification=file_specification,
    )

    # check that the mean/max/min are correct
    expected = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 4.0, 2.0],
        ]
    )

    assert np.array_equal(
        df.values, expected
    ), f"Stats dataframe incorrect. Difference:\n{df.values == expected}"


@pytest.fixture(name="setup_netcdf_file")
def fixture_setup_netcdf_file(tmp_path):
    """Fixture to create and initialize a dummy NetCDF file for testing."""

    data = initialize_dummy_netcdf_file(tmp_path / NC_FILE_NAME)

    # Creating variable "v1" with specified dimensions and setting its values
    data.createVariable("v1", np.float64, dimensions=("t", "z", "x"))
    data.variables["v1"][:] = np.ones((TIME_DIM_SIZE, HEIGHT_DIM_SIZE, HOR_DIM_SIZE))
    data.variables["v1"][:, :, 0] = 0
    data.variables["v1"][:, :, -1] = 2

    # Creating variable "v2" with fill_value, and setting its values
    data.createVariable("v2", np.float64, dimensions=("t", "x"), fill_value=42)
    data.variables["v2"][:] = np.ones((TIME_DIM_SIZE, HOR_DIM_SIZE)) * 2
    data.variables["v2"][:, 0] = 1
    data.variables["v2"][:, 1] = 42  # should be ignored in max-statistic
    data.variables["v2"][:, -1] = 3

    # Creating variable "v3" and setting its values
    data.createVariable("v3", np.float64, dimensions=("t", "x"))
    data.variables["v3"][:] = np.ones((TIME_DIM_SIZE, HOR_DIM_SIZE)) * 3
    data.variables["v3"][:, 0] = 2
    data.variables["v3"][:, -1] = 4

    data.close()

    yield


def test_stats_netcdf(setup_netcdf_file, tmp_path):  # pylint: disable=unused-argument
    """Test that the statistics generated from the NetCDF file match the
    expected values."""

    file_specification = {
        "Test data": {
            "format": "netcdf",
            "time_dim": "t",
            "horizontal_dims": ["x"],
            "fill_value_key": "_FillValue",  # should be the name for fill_value
        },
    }

    # Call the function to generate the statistics dataframe
    df = create_stats_dataframe(
        input_dir=str(tmp_path),
        file_id=[["Test data", NC_FILE_GLOB]],
        stats_file_name=tmp_path / STATS_FILE_NAMES,
        file_specification=file_specification,
    )

    # Define the expected values for comparison
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

    # Check that the dataframe matches the expected values
    assert np.array_equal(
        df.values, expected
    ), f"Stats dataframe incorrect. Difference:\n{df.values == expected}"


@pytest.fixture(name="setup_csv_file")
def fixture_setup_csv_file(tmp_path):
    """
    Fixture to set up a temporary CSV file.
    """
    dat_file_name = tmp_path / "test_stats_csv.dat"

    # Create the CSV file with the necessary content
    lines = (
        "time v1  v2 v3 v4 v5",
        "10   1.4 15 16 17 18",
        "20   2.4 25 26 27 28",
        "30   3.4 35 36 37 38",
    )
    with open(dat_file_name, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def test_stats_csv(setup_csv_file, tmp_path):  # pylint: disable=unused-argument
    """
    Test that the statistics generated from the CSV file match the expected values.
    """

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

    # Call the function that creates the stats DataFrame
    df = create_stats_dataframe(
        input_dir=str(tmp_path),
        file_id=[["Test data", "test_stats_csv.dat"]],
        stats_file_name=tmp_path / "test_stats_csv.csv",
        file_specification=file_specification,
    )

    # Expected result
    expected = np.array(
        [
            [1.4, 1.4, 1.4, 2.4, 2.4, 2.4, 3.4, 3.4, 3.4],
            [15, 15, 15, 25, 25, 25, 35, 35, 35],
            [16, 16, 16, 26, 26, 26, 36, 36, 36],
            [17, 17, 17, 27, 27, 27, 37, 37, 37],
            [18, 18, 18, 28, 28, 28, 38, 38, 38],
        ]
    )

    # Assert the DataFrame matches the expected values
    assert np.array_equal(
        df.values, expected
    ), f"Stats DataFrame incorrect. Difference:\n{df.values != expected}"
