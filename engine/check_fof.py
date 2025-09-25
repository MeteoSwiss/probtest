"""
CLI for checking two fof files

This module provides a command-line interface (CLI) to check that
two given fof files are conistent.
"""

import os

import click
import numpy as np
import xarray as xr


def get_dims(ds):
    """
    Get dimension of reports and observations
    """
    vars_shape1 = []
    vars_shape2 = []
    shape1 = ds.attrs["n_hdr"]
    shape2 = ds.attrs["n_body"]

    for var in ds.data_vars:
        if ds[var].shape[0] == shape1:
            vars_shape1.append(var)
        elif ds[var].shape[0] == shape2:
            vars_shape2.append(var)

    return vars_shape1, vars_shape2


def split_feedback_dataset(ds):
    """
    Split feedback file according to reports and oservations dimensions,
    expand lat, lon, statid and time_nomi according to l_body
    and sort them to assure unique order
    """
    vars_shape1, _ = get_dims(ds)
    ds_reports = ds[vars_shape1]

    sort_keys_reports = ["lat", "lon", "statid", "time_nomi", "codetype"]
    ds_report_sorted = ds_reports.sortby(sort_keys_reports)

    coords = ds.coords.keys()

    dataset = ds
    lbody = ds["l_body"].values

    for varname in filter(
        lambda s: s in ["lat", "lon", "statid", "time_nomi"], dataset
    ):
        values = np.repeat(dataset[varname], lbody.astype(int))
        attrs = dataset[varname].attrs
        dataset = dataset.assign({varname: xr.Variable("d_body", values, attrs=attrs)})

    dataset = dataset.assign_coords(
        dict(zip(coords, [dataset[coord] for coord in coords]))
    )

    _, vars_shape2 = get_dims(dataset)

    ds_obs = dataset[vars_shape2]
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
        print(f"Differences in '{var_name}': {percent:.2f}% equal")
        diff_idx = np.where(~mask_equal.ravel())[0]

        diff = diff_idx[:20]
        for i in diff:
            print(f"   - ds1: {arr1.ravel()[i]}, ds2: {arr2.ravel()[i]}")

    return total, equal, diff


def prepare_array(arr):
    """
    To make sure nan values are recognised.
    """
    if arr.dtype == np.float32 and np.isnan(arr).any():
        return np.where(np.isnan(arr), -999999, arr)
    return arr


def print_entire_line(ds1, ds2, diff):
    """
    If the specific option is called, this function print
    the entire line in which differences were found.
    """
    if diff.size > 0:
        da1 = ds1.to_dataframe().reset_index()
        da2 = ds2.to_dataframe().reset_index()

        for i in diff:
            row1 = " | ".join(map(str, da1.loc[i]))
            row2 = " | ".join(map(str, da2.loc[i]))

            print(f"ds1: {row1}")
            print(f"ds2: {row2}")
            print("-" * max(len(row1), len(row2)))


def get_list_to_skip(file1):
    """
    Obtain the list of variables that we expect
    to change depending on the type of fof file.
    """
    name1 = os.path.basename(file1)

    name1_core = name1.replace("fof", "").replace(".nc", "")

    if "TEMP" in name1_core:
        list_to_skip = ["source"]

    elif "AIREP" in name1_core:
        list_to_skip = ["i_body", "record", "source"]

    elif "PILOT" in name1_core:
        list_to_skip = ["i_body", "record", "source", "e_o", "plevel"]

    elif "SYNOP" in name1_core:
        list_to_skip = [
            "i_body",
            "record",
            "source",
            "state",
            "plevel",
            "flags",
            "check",
            "sso_stdh",
        ]

    return list_to_skip


def compare_var_and_attr_ds(ds1, ds2, entireline, list_to_skip):
    """
    Variable by variable and attribute by attribute,
    comparison of the two files.
    """

    total_all, equal_all = 0, 0

    for var in set(ds1.data_vars).union(ds2.data_vars):
        if var in ds1.data_vars and var in ds2.data_vars and var not in list_to_skip:
            arr1 = prepare_array(ds1[var].values)
            arr2 = prepare_array(ds2[var].values)

            if arr1.size == arr2.size:
                t, e, diff = compare_arrays(arr1, arr2, var)
                if entireline:
                    print_entire_line(ds1, ds2, diff)

            else:
                t, e = max(arr1.size, arr2.size), 0

            total_all += t
            equal_all += e

        if var in ds1.attrs and var in ds2.attrs and var not in list_to_skip:
            arr1 = np.array(ds1.attrs[var], dtype=object)
            arr2 = np.array(ds2.attrs[var], dtype=object)
            if arr1.size == arr2.size:
                t, e, diff = compare_arrays(arr1, arr2, var)

                if entireline:
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


@click.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option(
    "--entireline", is_flag=True, help="If there are differences, printe the whole line"
)
def check_fof(file1, file2, entireline):

    if not primary_check(file1, file2):
        print("Different types of files")
        return

    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

    ds_reports1_sorted, ds_obs1_sorted = split_feedback_dataset(ds1)
    ds_reports2_sorted, ds_obs2_sorted = split_feedback_dataset(ds2)

    total_elements_all, equal_elements_all = 0, 0
    list_to_skip = get_list_to_skip(file1)

    for ds1, ds2 in [
        (ds_reports1_sorted, ds_reports2_sorted),
        (ds_obs1_sorted, ds_obs2_sorted),
    ]:
        t, e = compare_var_and_attr_ds(ds1, ds2, entireline, list_to_skip)
        total_elements_all += t
        equal_elements_all += e

    if total_elements_all > 0:
        percent_equal_all = (equal_elements_all / total_elements_all) * 100
        percent_diff_all = 100 - percent_equal_all
        print(f"Total percentage of equality: {percent_equal_all:.2f}%")
        print(f"Total percentage of difference: {percent_diff_all:.2f}%")


if __name__ == "__main__":
    check_fof()  # pylint: disable=no-value-for-parameter
