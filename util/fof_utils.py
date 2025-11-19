"""
This module contains functions for handling fof files
"""

import os
import shutil

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

    ds_obs = ds[observation_variables]
    sort_keys_obs = ["lat", "lon", "statid", "varno", "level", "time_nomi"]
    observation_variables.append("veri_data")
    ds_veri = ds[observation_variables]
    ds_obs_sorted = ds_obs.sortby(sort_keys_obs)
    ds_veri_sorted = ds_veri.sortby(sort_keys_obs)

    return ds_report_sorted, ds_obs_sorted, ds_veri_sorted


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
        diff_idx = np.where(~mask_equal.ravel())[0]
        diff = diff_idx

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
        return x.decode().rstrip(" '")
    return str(x).rstrip(" '")


def print_entire_line(ds1, ds2, diff):
    """
    If the specific option is called, this function print
    the entire line in which differences are found.
    """
    if diff.size > 0:
        da1 = ds1.to_dataframe().reset_index()
        da2 = ds2.to_dataframe().reset_index()

        for i in diff:
            col_width = 13
            row1 = "|".join(f"{clean_value(x):<{col_width}}" for x in da1.loc[i])

            row2 = "|".join(f"{clean_value(x):<{col_width}}" for x in da2.loc[i])

            diff_row = []
            for x, y in zip(da1.loc[i], da2.loc[i]):
                if pd.api.types.is_number(x) and pd.api.types.is_number(y):
                    row_diff = x - y
                else:
                    row_diff = "nan"

                diff_row.append(row_diff)

            row_diff = "|".join(f"{str(x):<{col_width}}" for x in diff_row)

            index = "|".join(f"{str(x):<{col_width}}" for x in da1.columns)

            print(f"\033[1mid\033[0m  : {index}")
            print(f"\033[1mref\033[0m : {row1}")
            print(f"\033[1mcur\033[0m : {row2}")
            print(f"\033[1mdiff\033[0m: {row_diff}")
            term_width = shutil.get_terminal_size().columns
            print("-" * term_width)


def write_lines(ds1, ds2, diff, path_name):
    """
    If the specific option is called, this function save
    the lines in which differences are found.
    """
    if diff.size > 0:
        da1 = ds1.to_dataframe().reset_index()
        da2 = ds2.to_dataframe().reset_index()
        col_width = 13
        index = "|".join(f"{str(x):<{col_width}}" for x in da1.columns)
        for i in diff:

            row1 = "|".join(f"{clean_value(x):<{col_width}}" for x in da1.loc[i])

            row2 = "|".join(f"{clean_value(x):<{col_width}}" for x in da2.loc[i])

            diff_row = []
            for x, y in zip(da1.loc[i], da2.loc[i]):
                if pd.api.types.is_number(x) and pd.api.types.is_number(y):
                    row_diff = x - y
                else:
                    row_diff = "nan"

                diff_row.append(row_diff)

            row_diff = "|".join(f"{str(x):<{col_width}}" for x in diff_row)

            with open(path_name, "a", encoding="utf-8") as f:
                f.write(f"id  : {index}" + "\n")
                f.write(f"ref  : {row1}" + "\n")
                f.write(f"cur  : {row2}" + "\n")
                f.write(f"diff : {row_diff}" + "\n")


def write_different_size(output, nl, path_name, var, sizes):
    if output:
        with open(path_name, "a", encoding="utf-8") as f:
            f.write(
                f"variable  : {var} -> datasets have different lengths "
                f"({sizes[0]} vs. {sizes[1]} ), comparison not possible" + "\n"
            )
        if nl != 0:
            print(
                f"\033[1mvar\033[0m : {var} -> datasets have different lengths "
                f"({sizes[0]} vs. {sizes[1]} ), comparison not possible"
            )


def compare_var_and_attr_ds(ds1, ds2, nl, output, location):
    """
    Variable by variable and attribute by attribute,
    comparison of the two files.
    """

    total_all, equal_all = 0, 0
    list_to_skip = ["source", "i_body", "l_body"]

    if output:
        if location:
            path_name = location
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path_name = os.path.join(script_dir, "differences.csv")

        with open(path_name, "w", encoding="utf-8") as f:
            f.write("Differences\n")

    for var in set(ds1.data_vars).union(ds2.data_vars):
        if var in ds1.data_vars and var in ds2.data_vars and var not in list_to_skip:

            arr1 = fill_nans_for_float32(ds1[var].values)
            arr2 = fill_nans_for_float32(ds2[var].values)

            if arr1.size == arr2.size:
                t, e, diff = compare_arrays(arr1, arr2, var)

                if output:
                    write_lines(ds1, ds2, diff, path_name)

                if nl != 0:
                    diff = diff[:nl]
                    print_entire_line(ds1, ds2, diff)

            else:
                t, e = max(arr1.size, arr2.size), 0

                write_different_size(output, nl, path_name, var, [arr1.size, arr2.size])

            total_all += t
            equal_all += e

        if var in ds1.attrs and var in ds2.attrs and var not in list_to_skip:
            arr1 = np.array(ds1.attrs[var], dtype=object)
            arr2 = np.array(ds2.attrs[var], dtype=object)
            if arr1.size == arr2.size:
                t, e, diff = compare_arrays(arr1, arr2, var)

                if output:
                    write_lines(ds1, ds2, diff, path_name)

                if nl != 0:
                    diff = diff[:nl]
                    print_entire_line(ds1, ds2, diff)

            else:
                t, e = max(arr1.size, arr2.size), 0
            total_all += t
            equal_all += e

    return total_all, equal_all


def primary_check(file1, file2):
    """
    Test that the two files are of the same type.
    """
    name1 = os.path.basename(file1)
    name2 = os.path.basename(file2)

    name1_core = name1.replace("fof", "").replace(".nc", "")
    name2_core = name2.replace("fof", "").replace(".nc", "")

    return name1_core == name2_core
