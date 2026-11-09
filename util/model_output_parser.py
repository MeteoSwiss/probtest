"""
Definition of available model output data parsers

All available parsers are summarized in the dict `model_output_parser` at the
end of this module.

Each parser expects two arguments:
file_id: str
  identifier of the file type
filename: str
  full file name
specification: dict
  Dictionary with type specific settings

Each parser returns a list of Pandas DataFrame:
  index = pd.MultiIndex.from_product(
      [[file_id], [varname], height], names=("file_ID", "variable", "height")
  )
  columns = pd.MultiIndex.from_product(
      [time, compute_statistics], names=("time", "statistic")
  )

  return [pd.DataFrame(matrix, index=index, columns=columns)]
"""

import os
import re
import sys
from collections.abc import Iterable
from typing import Any, Dict, List

import earthkit.data
import eccodes
import eccodes_cosmo_resources
import numpy as np
import pandas as pd
import xarray

from util.constants import compute_statistics
from util.log_handler import logger
from util.utils import numbers
from util.xarray_ops import statistics_over_horizontal_dim

# Make eccodes aware of the COSMO/ICON local-table GRIB definitions (e.g. for
# fields using local parameter/level tables) in addition to the vendor-shipped
# ones, so GRIB files produced by COSMO/ICON decode correctly.
_cosmo_definitions_path = eccodes_cosmo_resources.get_definitions_path()
_vendor_definitions_path = eccodes.codes_definition_path()
eccodes.codes_set_definitions_path(
    f"{_cosmo_definitions_path}:{_vendor_definitions_path}"
)

# Metadata keys used to split a GRIB file into homogeneous groups before
# building a dataset out of each. A single GRIB file commonly mixes fields
# that cannot share one xarray Dataset (different level types, step types,
# grids, ...); see `parse_grib`/`__grib_dataset_from_group` for why this is
# done with plain per-field metadata reads instead of
# `FieldList.to_xarray(split_dims=...)`.
GRIB_SPLIT_KEYS = [
    "typeOfLevel",
    "stepType",
    "gridType",
    "numberOfPoints",
    "dataType",
]


def parse_netcdf(
    file_id: str, filename: str, specification: Dict[str, Any]
) -> List[pd.DataFrame]:
    """
    Parse a NetCDF file into pandas DataFrames.
    """

    logger.debug("parse NetCDF file %s", filename)
    time_dim = specification["time_dim"]
    horizontal_dims = specification["horizontal_dims"]
    fill_value_key = specification.get("fill_value_key", None)
    ds = xarray.open_dataset(filename, decode_cf=False)

    # Convert all float variables to float64
    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.floating):
            ds[v] = ds[v].astype(np.float64)

    var_tmp = __get_variables(ds, time_dim, horizontal_dims)

    var_dfs = []

    for v in var_tmp:
        sub_df = dataframe_from_ncfile(
            file_id=file_id,
            filename=filename,
            varname=v,
            time_dim=time_dim,
            horizontal_dims=horizontal_dims,
            xarray_ds=ds,
            fill_value_key=fill_value_key,
        )
        var_dfs.append(sub_df)

    ds.close()
    return var_dfs


def __grib_dataset_from_group(
    fieldlist, group: pd.DataFrame, time_dim: str
) -> xarray.Dataset:
    """
    Build a minimal xarray.Dataset directly from a homogeneous group of GRIB
    fields (same typeOfLevel/stepType/gridType/numberOfPoints/dataType),
    without going through earthkit-data's `to_xarray()`.

    This deliberately bypasses `to_xarray()`'s geometry ("gridSpec")
    resolution: for ICON's native unstructured grid, whose UUID is not
    registered with eckit-geo, that resolution fails unconditionally --
    `to_xarray()` raises `GridUnknownError`/`GribGeographyBuilder: cannot use
    unstructured grid because gridSpec is not available` even when no
    lat/lon coordinates are requested (e.g. `add_geo_coords=False`), because
    it needs geometry to determine the dataset's shape/index, not just its
    coordinates. Only the raw grid-point dimension ("values", per
    `horizontal_dims` in the ICON template) is needed for stats, so fields
    are read here via plain per-field metadata/values access, which does not
    touch geometry at all.

    Each variable gets its own uniquely-named level dimension
    (`level__<shortName>`), since different variables in the same group can
    span different sets of levels (e.g. different numbers of pressure
    levels) and xarray requires same-named dimensions to share a length.
    """
    n_points = int(group["numberOfPoints"].iloc[0])
    steps = np.sort(group["step"].unique())

    data_vars = {}
    for var_name, var_group in group.groupby("shortName", sort=False):
        levels = np.sort(var_group["level"].unique())
        if len(levels) == 1:
            # No vertical variation for this variable in this group (e.g. a
            # surface field): omit the level dimension entirely, matching
            # `dataframe_from_ncfile`'s convention of a plain (time, values)
            # array (which it reports with the height=-1 sentinel) rather
            # than a spurious size-1 level dimension.
            flat_matrix = np.full((len(steps), n_points), np.nan, dtype=np.float64)
            for _, row in var_group.iterrows():
                s_idx = np.searchsorted(steps, row["step"])
                flat_matrix[s_idx, :] = fieldlist[int(row["idx"])].to_numpy(
                    dtype=np.float64
                )
            data_vars[var_name] = xarray.Variable((time_dim, "values"), flat_matrix)
            continue

        level_dim = f"level__{var_name}"
        leveled_matrix = np.full(
            (len(steps), len(levels), n_points), np.nan, dtype=np.float64
        )
        for _, row in var_group.iterrows():
            s_idx = np.searchsorted(steps, row["step"])
            l_idx = np.searchsorted(levels, row["level"])
            leveled_matrix[s_idx, l_idx, :] = fieldlist[int(row["idx"])].to_numpy(
                dtype=np.float64
            )
        data_vars[var_name] = xarray.Variable(
            (time_dim, level_dim, "values"), leveled_matrix
        )
        data_vars[f"__coord_{level_dim}"] = xarray.Variable((level_dim,), levels)

    ds = xarray.Dataset(data_vars, coords={time_dim: steps})
    # move the per-variable level coordinates out of data_vars and into coords
    for name in list(ds.data_vars):
        if name.startswith("__coord_"):
            level_dim = name[len("__coord_") :]
            ds = ds.assign_coords({level_dim: ds[name].values})
            ds = ds.drop_vars(name)
    return ds


def __lead_time_seconds_from_filename(filename: str) -> "int | None":
    """
    Extract the forecast lead time (in seconds) from an ICON GRIB output
    filename following the `lfff<DDHHMMSS>` / `lffm<DDHHMMSS>` naming
    convention (elapsed days/hours/minutes/seconds since the start of the
    run), e.g. `lfff00000010p` -> 10s, `lfff00000100p` -> 60s. Returns None
    if `filename` doesn't look like that convention (e.g. an arbitrary/test
    file name), so the caller can fall back to the GRIB message's own
    `step`/`forecastTime` metadata.

    This is preferred over that metadata where it does apply: for some ICON
    output (e.g. sub-hourly intervals encoded with `stepUnits` in whole
    hours), that metadata is coarser than the actual output interval and
    cannot distinguish between files of the same file_id pattern at
    different lead times -- every message then reports `step=0`, e.g.
    `lfff00000000p`, `lfff00000010p` and `lfff00000020p` (leads 0s/10s/20s)
    all report `step=0`, which silently loses the actual lead time.
    Combining several such files under one file_id pattern (as `stats` does
    for a multi-file glob) then fails with "cannot reindex on an axis with
    duplicate labels" -- see `parse_grib`.
    """
    match = re.fullmatch(r"[A-Za-z_]+(\d{8})[A-Za-z0-9_]*", os.path.basename(filename))
    if not match:
        return None
    days, hours, minutes, seconds = (
        int(match.group(1)[0:2]),
        int(match.group(1)[2:4]),
        int(match.group(1)[4:6]),
        int(match.group(1)[6:8]),
    )
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def parse_grib(
    file_id: str, filename: str, specification: Dict[str, Any]
) -> List[pd.DataFrame]:
    """
    Parse a GRIB file into pandas DataFrames.
    """

    logger.debug("parse GRIB file %s", filename)
    time_dim = specification["time_dim"]
    horizontal_dims = specification["horizontal_dims"]
    fill_value_key = specification.get("fill_value_key", None)
    var_excl = specification.get("var_excl", [])

    fieldlist = earthkit.data.from_source("file", filename).to_fieldlist()

    var_dfs: List[pd.DataFrame] = []
    if len(fieldlist) == 0:
        return var_dfs

    lead_time_seconds = __lead_time_seconds_from_filename(filename)

    # Gather the per-field metadata needed to (a) filter out excluded
    # variables and (b) split the file into homogeneous groups, all via
    # plain scalar metadata keys. These are safe to read directly (unlike
    # going through `to_xarray()`, see `__grib_dataset_from_group`).
    fields = pd.DataFrame(
        {
            "idx": np.arange(len(fieldlist)),
            "shortName": fieldlist.metadata("shortName"),
            "typeOfLevel": fieldlist.metadata("typeOfLevel"),
            "stepType": fieldlist.metadata("stepType"),
            "gridType": fieldlist.metadata("gridType"),
            "numberOfPoints": fieldlist.metadata("numberOfPoints"),
            "dataType": fieldlist.metadata("dataType"),
            "level": fieldlist.metadata("level"),
            # Prefer the lead time encoded in the filename over the GRIB
            # message's own step/forecastTime metadata where it applies
            # (lead_time_seconds is None, not 0, for a non-ICON-convention
            # filename); fall back to that metadata otherwise. See
            # __lead_time_seconds_from_filename.
            "step": (
                fieldlist.metadata("step", astype=int)
                if lead_time_seconds is None
                else lead_time_seconds
            ),
        }
    )
    fields = fields[~fields["shortName"].isin(var_excl)]
    if fields.empty:
        return var_dfs

    # A GRIB file commonly mixes fields that don't share one shape (different
    # level types, step types, grids, ...), so split it into homogeneous
    # groups first; each group is then handled like a NetCDF dataset.
    for _, group in fields.groupby(GRIB_SPLIT_KEYS, sort=True):
        ds = __grib_dataset_from_group(fieldlist, group, time_dim)

        for v in __get_variables(ds, time_dim, horizontal_dims):
            sub_df = dataframe_from_ncfile(
                file_id=file_id,
                filename=filename,
                varname=v,
                time_dim=time_dim,
                horizontal_dims=horizontal_dims,
                xarray_ds=ds,
                fill_value_key=fill_value_key,
            )
            var_dfs.append(sub_df)

    return var_dfs


def __get_variables(data, time_dim, horizontal_dims):
    # return a list of variable names from the dataset data that have a time dimension
    # and horizontal dimension or in case there is no time dimension just the variables
    #  with horizontal dimension
    all_variables = data.variables.keys()
    all_variables = [
        v for v in all_variables if np.issubdtype(data.variables[v].dtype, np.number)
    ]

    variables = []

    if isinstance(horizontal_dims, Iterable):
        horizontal_dims_unpacked = [x for hd in horizontal_dims for x in hd.split(":")]

    if time_dim is not None:
        if isinstance(horizontal_dims, Iterable):
            for v in all_variables:
                if (
                    any(hd in horizontal_dims_unpacked for hd in data.variables[v].dims)
                    and time_dim in data.variables[v].dims
                ):
                    variables.append(v)
        else:
            variables = [
                v
                for v in all_variables
                if time_dim in data.variables[v].dims and v != time_dim
            ]
    else:
        if isinstance(horizontal_dims, Iterable):
            for v in all_variables:
                if any(hd in horizontal_dims_unpacked for hd in data.variables[v].dims):
                    variables.append(v)
        else:
            logger.error(
                "horizontal_dims have to be specified when no time_dim is given"
            )
            sys.exit(1)

    return variables


def dataframe_from_ncfile(
    file_id, filename, varname, time_dim, horizontal_dims, xarray_ds, fill_value_key
):  # pylint: disable=too-many-positional-arguments
    statistics = statistics_over_horizontal_dim(
        xarray_ds[varname],
        horizontal_dims,
        compute_statistics,
        fill_value_key,
    )

    first_stat = statistics[0]
    if len(first_stat.dims) == 2:
        height_name = (
            first_stat.dims[0] if first_stat.dims[0] != time_dim else first_stat.dims[1]
        )  # might be 'height', 'height_2', 'alt', 'plev', ...
        height = xarray_ds[height_name].values
        matrix = np.empty(
            (first_stat.shape[0] * len(statistics), first_stat.shape[1])
        )  # shape: (mean_max_min of each height, time)

        # weave mean max min into time dimension
        for i, stat in enumerate(statistics):
            matrix[i :: len(statistics), :] = stat.values
    elif len(first_stat.dims) == 1:
        if first_stat.dims[0] == time_dim:
            height = np.array([-1])
            # matrix needs to have 2 dimensions for DataFrame constructor
            matrix = np.empty((first_stat.size * len(statistics), 1))
            # weave mean max min into time dimension
            for i, stat in enumerate(statistics):
                matrix[i :: len(statistics), 0] = stat.values
        else:
            height = xarray_ds[first_stat.dims[0]].values
            # matrix needs to have 2 dimensions for DataFrame constructor
            matrix = np.empty((len(statistics), first_stat.size))
            # weave mean max min into time dimension
            for i, stat in enumerate(statistics):
                matrix[i, :] = stat.values
    elif len(first_stat.dims) == 0:
        height = np.array([-1])
        # matrix needs to have 2 dimensions for DataFrame constructor
        matrix = np.empty((len(statistics), 1))
        # weave mean max min into time dimension
        for i, stat in enumerate(statistics):
            matrix[i :: len(statistics), 0] = stat.values
    else:
        logger.error(
            (
                "Unknown number of dimension for first_stat of variable '%s'. "
                + "Dims: %s"
            ),
            varname,
            str(first_stat.dims),
        )
        sys.exit(1)

    if time_dim is not None:
        time = xarray_ds[time_dim].values
    else:
        # Derive a pseudo time from filename. This is required to process multiple
        # files of the same file type if the file type has not time dimension.
        time = [numbers(filename)]

    index = pd.MultiIndex.from_product(
        [[file_id], [varname], height], names=("file_ID", "variable", "height")
    )
    columns = pd.MultiIndex.from_product(
        [time, compute_statistics], names=("time", "statistic")
    )

    return pd.DataFrame(matrix.T, index=index, columns=columns)


def parse_csv(file_id, filename, specification):
    """
    file_id: str
        identifier of the file type
    filename: str
        full file name
    specification: dict(parser_args, time_dim, horizontal_dims)
        parser_args: dict
            passed directly to pandas.read_csv

    The (first) index of the read csv (i.e. usually the rows) is expected to
    represent the time dimension.
    Each column contains a individual variable
    """
    logger.debug("parse CSV file %s", filename)

    csv = pd.read_csv(filename, **specification["parser_args"])

    # transpose data such that time is along columns
    csv = csv.transpose()

    if csv.columns.nlevels == 1:
        # use the values in csv as dummy for each of the expected compute_statistics
        matrix = np.array(csv).repeat(len(compute_statistics), 1)
        columns = csv.columns
        # regular Index -> "height" information does not apply
        height = [-1]
    else:
        n_time = len(csv.columns.levels[0])
        # use "height" for independent levels of a variable
        height = np.arange(csv.columns.size / n_time, dtype=int)

        # convert to proper multidimensional array
        array = np.array(csv).reshape((csv.index.size, n_time, -1))
        # transpose such that time is in last dimension
        array = array.transpose(0, 2, 1)  # index, "height", time
        # collapse index and "height" dimensions
        array = array.reshape(-1, n_time)
        matrix = array.repeat(len(compute_statistics), 1)

        columns = csv.columns.levels[0]

    index = pd.MultiIndex.from_product(
        [[file_id], csv.index, height], names=("file_ID", "variable", "height")
    )
    columns = pd.MultiIndex.from_product(
        [columns, compute_statistics], names=("time", "statistic")
    )
    return [pd.DataFrame(matrix, index=index, columns=columns)]


model_output_parser = {  # global lookup dict
    "netcdf": parse_netcdf,
    "csv": parse_csv,
    "grib": parse_grib,
}
