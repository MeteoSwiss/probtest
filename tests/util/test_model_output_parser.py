"""
This module contains unit tests for the `model_output_parser.py` module.
"""

import numpy as np
import xarray as xr

from util.model_output_parser import parse_netcdf


def test_parse_netcdf_only_floats_converted(tmp_path):
    """
    Ensure parse_netcdf converts only float variables to float64 and
    does not attempt to convert string/bytes variables.
    """

    ds = xr.Dataset(
        {
            "float32_var": ("t", np.array([1.0, 2.0], dtype=np.float32)),
            "float64_var": ("t", np.array([1.0, 2.0], dtype=np.float64)),
            "int_var": ("t", np.array([1, 2], dtype=np.int32)),
            "str_var": ("t", np.array([b"A", b"B"], dtype="S1")),
        }
    )

    # Save to a temporary NetCDF file
    filename = tmp_path / "test.nc"
    ds.to_netcdf(filename)

    # Specification for parse_netcdf
    specification = {
        "time_dim": "t",
        "horizontal_dims": [],
        "fill_value_key": None,
    }

    var_dfs = parse_netcdf("test_file", str(filename), specification)

    # Check dtypes
    var_names = ["float32_var", "float64_var", "int_var", "str_var"]
    for name, df in zip(var_names, var_dfs):
        dtype = df.dtypes[0]
        if name.startswith("float"):
            assert dtype == np.float64
        elif name.startswith("int"):
            assert np.issubdtype(dtype, np.integer)
        elif name.startswith("str"):
            assert np.issubdtype(dtype, np.object_)  # pandas converts bytes -> object
