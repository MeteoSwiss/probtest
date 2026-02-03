"""
This module contains functions for handling fof files
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from util.log_handler import logger


def get_report_variables(ds):
    """
    Get variable names of reports.
    """
    vars_shape_report = []
    shape_report = ds.attrs["n_hdr"]

    for var in ds.data_vars:
        if ds[var].shape[0] == shape_report:
            vars_shape_report.append(var)

    return vars_shape_report


def get_observation_variables(ds):
    """
    Get variable names of observations.
    """
    vars_shape_observation = []
    shape_observation = ds.attrs["n_body"]

    for var in ds.data_vars:
        if ds[var].shape[0] == shape_observation:
            vars_shape_observation.append(var)

    return vars_shape_observation


def split_feedback_dataset(ds):
    """
    Split feedback file according to reports and observations dimensions,
    expand lat, lon, statid and time_nomi according to l_body
    and sort them to assure unique order.
    """
    report_variables = get_report_variables(ds)
    ds_reports = ds[report_variables]

    sort_keys_reports = ["lat", "lon", "statid", "time_nomi", "codetype"]
    ds_report_sorted = ds_reports.sortby(sort_keys_reports)

    lbody = ds["l_body"].values

    for varname in filter(lambda s: s in ["lat", "lon", "statid", "time_nomi"], ds):
        values = np.repeat(ds[varname], lbody.astype(int))
        attrs = ds[varname].attrs
        ds = ds.assign({varname: xr.Variable("d_body", values, attrs=attrs)})

    ds = ds.assign_coords(dict(zip(ds.coords, [ds[coord] for coord in ds.coords])))

    observation_variables = get_observation_variables(ds)
    observation_variables.append("veri_data")

    ds_obs = ds[observation_variables]
    sort_keys_obs = ["lat", "lon", "statid", "varno", "level", "time_nomi"]
    ds_obs_sorted = ds_obs.sortby(sort_keys_obs)

    return ds_report_sorted, ds_obs_sorted


def compare_arrays(arr1, arr2, var_name):
    """
    Comparison of two arrays containing the values of the same variable.
    If not the same, it tells you in percentage terms how different they are.
    """
    total = arr1.size

    if np.array_equal(arr1, arr2):
        equal = total
        diff = np.array([])

    elif (
        np.issubdtype(arr1.dtype, np.number)
        and np.issubdtype(arr2.dtype, np.number)
        and np.array_equal(arr1, arr2, equal_nan=True)
    ):
        equal = total
        diff = np.array([])

    else:
        mask_equal = arr1 == arr2
        equal = mask_equal.sum()
        percent = (equal / total) * 100
        logger.info(
            "Differences in '%s': %.2f%% equal. %s total entries for this variable",
            var_name,
            percent,
            total,
        )
        diff = np.where(~mask_equal.ravel())[0]

    return total, equal, diff


def fill_nans_for_float32(arr):
    """
    To make sure nan values are recognised.
    """
    if arr.dtype == np.float32 and np.isnan(arr).any():
        return np.where(np.isnan(arr), -999999, arr)
    return arr


def clean_value(x):
    """
    Elimination of unnecessary spaces that ruin the
    alignment when printing the value.
    """
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace").rstrip(" '")
    return str(x).rstrip(" '")


def write_lines_log(ds1, ds2, diff, detailed_logger):
    """
    This function writes the differences detected between
    two files to a detailed log file.
    """

    da1 = ds1.to_dataframe().reset_index()
    da2 = ds2.to_dataframe().reset_index()
    col_width = 13
    index = "|".join(f"{str(x):<{col_width}}" for x in da1.columns)

    for i in diff:
        row1 = "|".join(f"{clean_value(x):<{col_width}}" for x in da1.loc[i])
        row2 = "|".join(f"{clean_value(x):<{col_width}}" for x in da2.loc[i])

        diff_vals = []
        for x, y in zip(da1.loc[i], da2.loc[i]):
            if pd.api.types.is_number(x) and pd.api.types.is_number(y):
                diff_vals.append(x - y)
            else:
                diff_vals.append("nan")

        row_diff = "|".join(f"{str(x):<{col_width}}" for x in diff_vals)

        detailed_logger.info("id  : %s", index)
        detailed_logger.info("ref  : %s", row1)
        detailed_logger.info("cur  : %s", row2)
        detailed_logger.info("diff : %s", row_diff)
        detailed_logger.info("")


def write_different_size_log(var, size1, size2, detailed_logger):
    """
    This function is triggered when the array sizes do not match and records
    in the log file that a comparison is not possible.
    """

    detailed_logger.info(
        "variable  : %s -> datasets have different lengths "
        "(%s vs. %s), comparison not possible\n",
        var,
        size1,
        size2,
    )


def compare_var_and_attr_ds(ds1, ds2, detailed_logger):
    """
    Variable by variable and attribute by attribute,
    comparison of the two datasets.
    """

    total_all, equal_all = 0, 0
    # total, equal = 0, 0
    list_to_skip = ["source", "i_body", "l_body", "veri_data"]

    for var in sorted(set(ds1.data_vars).union(ds2.data_vars)):
        if var in ds1.data_vars and var in ds2.data_vars and var not in list_to_skip:

            total, equal = process_var(ds1, ds2, var, detailed_logger)
            total_all += total
            equal_all += equal

        if var in ds1.attrs and var in ds2.attrs and var not in list_to_skip:

            total, equal = process_var(ds1, ds2, var, detailed_logger)
            total_all += total
            equal_all += equal

    return total_all, equal_all


def process_var(ds1, ds2, var, detailed_logger):
    """
    This function first checks whether two arrays have the same size.
    If they do, their values are compared.
    If they don't, the differences are written to a log file.
    The function outputs the total number of elements and the
    number of matching elements.
    """

    arr1 = fill_nans_for_float32(ds1[var].values)
    arr2 = fill_nans_for_float32(ds2[var].values)
    if arr1.size == arr2.size:
        t, e, diff = compare_arrays(arr1, arr2, var)
        if diff.size != 0:
            write_lines_log(ds1, ds2, diff, detailed_logger)

    else:
        t, e = max(arr1.size, arr2.size), 0
        write_different_size_log(var, arr1.size, arr2.size, detailed_logger)

    return t, e


def create_tolerance_csv(n_rows, tol):
    """
    This function generates a file with the same number of lines as the file being
    analyzed, where each line contains the tolerances specified when fof-compare
    is called.
    """
    tolerance_file_name = "tolerance_file.csv"
    df = pd.DataFrame({"tolerance": [tol] * n_rows})
    df.to_csv(tolerance_file_name)

    return tolerance_file_name


def get_log_file_name(file_path):
    """
    This function gives the name of the detailed log file,
    according to the file path.
    """

    core_name = os.path.basename(file_path).replace(".nc", "")
    log_file_name = f"error_{core_name}.log"
    return log_file_name


def clean_logger_file_if_only_details(file_path):
    """
    This function deletes the detailed log file if it doesn't
    contain anything.
    """
    target_line = "initialized named logger 'DETAILS'"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    stripped_lines = [line.strip() for line in lines if line.strip()]

    if 0 < len(stripped_lines) <= 2 and all(
        line == target_line for line in stripped_lines
    ):
        os.remove(file_path)
