"""
This module contains unit tests for the `model_output_parser.py` module.
"""

import eccodes
import numpy as np
import xarray as xr

from util.model_output_parser import parse_grib, parse_netcdf

# `reduced_rotated_gg_sfc_grib2.tmpl` is a fixed-grid GRIB2 sample template
# shipped with eccodes; its grid definition fixes the number of horizontal
# points to this value.
GRIB_HORIZONTAL_SIZE = 6114


def _write_grib_field(f_out, short_name, values, type_of_level="surface", level=0):
    gid = eccodes.codes_grib_new_from_samples("reduced_rotated_gg_sfc_grib2.tmpl")
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


def test_parse_grib_single_hypercube(tmp_path):
    """
    A GRIB file where all fields share the same shape (one level type, one
    level) should be parsed into one DataFrame per variable.
    """
    values = np.ones(GRIB_HORIZONTAL_SIZE, dtype=np.float32)
    values[0] = 0
    values[-1] = 2

    filename = tmp_path / "test.grib"
    with open(filename, "wb") as f_out:
        _write_grib_field(f_out, "v", values)

    specification = {
        "time_dim": "step",
        "horizontal_dims": ["values"],
        "var_excl": [],
    }

    var_dfs = parse_grib("test_file", str(filename), specification)

    assert len(var_dfs) == 1
    df = var_dfs[0]
    assert df.dtypes.iloc[0] == np.float64  # float32 GRIB packing is upcast
    mean, maximum, minimum = df.iloc[0]
    assert np.isclose(mean, values.mean())
    assert np.isclose(maximum, values.max())
    assert np.isclose(minimum, values.min())


def test_parse_grib_var_excl(tmp_path):
    """
    Fields whose shortName is listed in `var_excl` must not appear in the
    output (e.g. constant grid-geometry fields like lon/lat).
    """
    filename = tmp_path / "test.grib"
    with open(filename, "wb") as f_out:
        _write_grib_field(f_out, "v", np.ones(GRIB_HORIZONTAL_SIZE))
        _write_grib_field(f_out, "u", np.ones(GRIB_HORIZONTAL_SIZE))

    specification = {
        "time_dim": "step",
        "horizontal_dims": ["values"],
        # the COSMO/ICON local definitions normalize shortName "u" to "U";
        # var_excl must match the resolved metadata shortName, not the value
        # passed to codes_set.
        "var_excl": ["U"],
    }

    var_dfs = parse_grib("test_file", str(filename), specification)

    assert len(var_dfs) == 1


def test_parse_grib_splits_incompatible_level_types(tmp_path):
    """
    A GRIB file mixing a surface field and a multi-level field for the same
    shortName does not form one hypercube; parse_grib must split them into
    separate DataFrames (one per height) instead of erroring out.
    """
    filename = tmp_path / "test.grib"
    with open(filename, "wb") as f_out:
        surface_values = np.ones(GRIB_HORIZONTAL_SIZE)
        surface_values[0] = 0
        surface_values[-1] = 2
        _write_grib_field(f_out, "t", surface_values, type_of_level="surface")

        for level in (1, 2, 3):
            _write_grib_field(
                f_out,
                "t",
                np.ones(GRIB_HORIZONTAL_SIZE) * level,
                type_of_level="hybrid",
                level=level,
            )

    specification = {
        "time_dim": "step",
        "horizontal_dims": ["values"],
        "var_excl": [],
    }

    var_dfs = parse_grib("test_file", str(filename), specification)

    # one DataFrame for the surface field, one for the 3-level hybrid field
    assert len(var_dfs) == 2
    heights = sorted(h for df in var_dfs for h in df.index.get_level_values("height"))
    assert heights == [-1, 1, 2, 3]
