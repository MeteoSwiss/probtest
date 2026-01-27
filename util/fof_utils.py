"""
This module contains functions for handling fof files
"""

import logging
import os

import numpy as np
import pandas as pd
import xarray as xr



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
        print(
            f"Differences in '{var_name}': {percent:.2f}% equal. "
            f"{total} total entries for this variable"
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


def write_lines_log(ds1, ds2, diff, log_path):
    if diff.size == 0:
        return

    logger = setup_logger(log_path)

    if not hasattr(write_lines_log, "_header_written"):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Differences\n\n")
        write_lines_log._header_written = True

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

        logger.info(f"id  : {index}")
        logger.info(f"ref  : {row1}")
        logger.info(f"cur  : {row2}")
        logger.info(f"diff : {row_diff}")
        logger.info("")


def write_different_size_log(var, size1, size2, log_path):
    """
    This function is triggered when the array sizes do not match and records
    in the log file that a comparison is not possible.
    """
    logger = setup_logger(log_path)
    logger.info(
        f"variable  : {var} -> datasets have different lengths "
        f"({size1} vs. {size2} ), comparison not possible" + "\n"
    )

def write_tolerance_log(err, tol, log_path):
    """
    This function is triggered when the fof-compare step fails because the
    veri_data fall outside the specified tolerance range.
    Any resulting errors are recorded in a log file.
    """
    logger = setup_logger(log_path)
    logger.info("Differences, veri_data outside of tolerance range")
    logger.info(err)
    logger.info(tol)


def setup_logger(log_path):
    """
    Sets up a logger that appends plain-text messages to the given log
    file and returns the configured logger.
    """
    logger = logging.getLogger("diff_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def compare_var_and_attr_ds(ds1, ds2, name_core):
    """
    Variable by variable and attribute by attribute,
    comparison of the two files.
    """

    total_all, equal_all = 0, 0
    list_to_skip = ["source", "i_body", "l_body", "veri_data"]
    log_path = f"error_{name_core}.log"

    for var in set(ds1.data_vars).union(ds2.data_vars):
        if var in ds1.data_vars and var in ds2.data_vars and var not in list_to_skip:

            arr1 = fill_nans_for_float32(ds1[var].values)
            arr2 = fill_nans_for_float32(ds2[var].values)

            if arr1.size == arr2.size:
                t, e, diff = compare_arrays(arr1, arr2, var)

                write_lines_log(ds1, ds2, diff, log_path)

            else:
                t, e = max(arr1.size, arr2.size), 0
                write_different_size_log(var, arr1.size, arr2.size, log_path)

            total_all += t
            equal_all += e

        if var in ds1.attrs and var in ds2.attrs and var not in list_to_skip:

            arr1 = fill_nans_for_float32(ds1[var].values)
            arr2 = fill_nans_for_float32(ds2[var].values)
            if arr1.size == arr2.size:
                t, e, diff = compare_arrays(arr1, arr2, var)

                write_lines_log(ds1, ds2, diff, log_path)
            else:
                t, e = max(arr1.size, arr2.size), 0
                write_different_size_log(var, arr1.size, arr2.size, log_path)

            total_all += t
            equal_all += e

    return total_all, equal_all


def create_tolerance_csv(n_rows, tol, tolerance_file_name):
    """
    This function generates a file with the same number of lines as the file being
    analyzed, where each line contains the tolerances specified when fof-compare
    is called.
    """
    df = pd.DataFrame({"tolerance": [tol] * n_rows})
    df.to_csv(tolerance_file_name)


def primary_check(file1, file2):
    """
    Test that the two files are of the same type.
    """
    name1 = os.path.basename(file1)
    name2 = os.path.basename(file2)

    name1_core = name1.replace("fof", "").replace(".nc", "")
    name2_core = name2.replace("fof", "").replace(".nc", "")

    return name1_core == name2_core, name1_core
