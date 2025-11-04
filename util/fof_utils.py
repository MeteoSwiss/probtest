"""
This module contains functions for handling fof files
"""

import os
import shutil
from dataclasses import dataclass
from enum import Enum

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
    Split feedback file according to reports and oservations dimensions,
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


def prepare_array(arr):
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


def check_multiple_solutions(ds1, ds2, existing_cols):

    allowed_checks = [13, 18, 32]
    allowed_states = [1, 5, 7, 9]

    df1 = ds1[existing_cols].to_dataframe().reset_index()
    df2 = ds2[existing_cols].to_dataframe().reset_index()

    total_rows = df1[existing_cols].size
    errors = 0
    diff = []

    check_cols = ["check", "r_check"]
    state_cols = ["state", "r_state"]

    for idx in df1.index:
        check_col = next((c for c in check_cols if c in df1.columns), None)
        state_col = next((c for c in state_cols if c in df1.columns), None)

        if check_col is None or state_col is None:
            raise KeyError("Colonne 'check' o 'state' non trovate nei dataset.")

        check_ref = df1.at[idx, check_col]
        check_cur = df2.at[idx, check_col]
        state_ref = df1.at[idx, state_col]
        state_cur = df2.at[idx, state_col]

        # CASE 1: check does not change
        if check_ref == check_cur:
            # If is not an accepted check, state should not change
            if check_ref not in allowed_checks:
                if state_ref != state_cur:
                    errors += 1
                    diff.append(idx)

            # If is an admitted change, state can change, but only in the admitted cases
            else:
                if state_ref != state_cur:
                    if (state_ref not in allowed_states) or (
                        state_cur not in allowed_states
                    ):
                        errors += 1
                        diff.append(idx)

        # CASE 2: check changes
        else:
            if (check_ref not in allowed_checks) and (check_cur not in allowed_checks):
                errors += 1
                diff.append(idx)

            # If check values are both admitted, also state should
            # be in the admitted values
            elif (state_ref not in allowed_states) or (state_cur not in allowed_states):
                errors += 1
                diff.append(idx)

    diff = np.array(diff)

    return total_rows, errors, diff


def compare_var_and_attr_ds(ds1, ds2, nl, output, location):  # noqa: C901
    """
    Variable by variable and attribute by attribute,
    comparison of the two files.
    """

    total_all, equal_all = 0, 0
    list_to_skip = [
        "source",
        "i_body",
        "l_body",
        "state",
        "check",
        "r_state",
        "r_check",
    ]

    if output:
        if location:
            path_name = location
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path_name = os.path.join(script_dir, "differences.csv")

        with open(path_name, "w", encoding="utf-8") as f:
            f.write("Differences\n")

    cols = ["check", "state", "r_state", "r_check"]
    existing_cols = [c for c in cols if c in ds1 and c in ds2]

    if existing_cols:
        ds1_multiple = ds1[existing_cols]
        ds2_multiple = ds2[existing_cols]

        total_rows, errors, diff = check_multiple_solutions(
            ds1_multiple, ds2_multiple, existing_cols
        )

        total_all += total_rows
        equal_all += total_rows - errors

        if output:
            write_lines(ds1, ds2, diff, path_name)

        if nl != 0:
            diff = diff[:nl]
            print_entire_line(ds1, ds2, diff)

    for var in set(ds1.data_vars).union(ds2.data_vars):
        if var in ds1.data_vars and var in ds2.data_vars and var not in list_to_skip:

            arr1 = prepare_array(ds1[var].values)
            arr2 = prepare_array(ds2[var].values)

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


def to_list(value):
    """
    This function ensures that whatever value is passed to expand_zip,
    one always gets something iterable and consistent.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def value_list(placeholder, values, placeholders):
    """
    Decide which values to use for a placeholder.
    - If the placeholder is not present, returns [""]
    - If placeholder is present but no values provided, returns [None]
    """
    if not placeholders.get(placeholder, False):
        return [""]
    return values or [None]


def expand_zip(zipped, fof_type=None, member_ids=None, member_type=None):
    """
    Expands a list of tuples (zip) by replacing the placeholders
    {fof_type}, {member_id}, and {member_type} with the values provided.
    If a placeholder is present but no corresponding value is given,
    the placeholder is left unchanged.
    """

    fof_list = to_list(fof_type)
    member_list = to_list(member_ids)
    member_type_list = to_list(member_type)

    if isinstance(zipped, zip):
        zipped = list(zipped)
    elif isinstance(zipped, (list, tuple)):
        zipped = [(z,) if not isinstance(z, tuple) else z for z in zipped]
    else:
        zipped = [(zipped,)]

    expanded = []

    for items in zipped:
        placeholders = {
            key: any(f"{{{key}}}" in str(item) for item in items)
            for key in ("fof_type", "member_id", "member_type")
        }

        fof_values = value_list("fof_type", fof_list, placeholders)
        member_values = value_list("member_id", member_list, placeholders)
        member_type_values = value_list("member_type", member_type_list, placeholders)

        for fof_val in fof_values:
            for member_val in member_values:
                for mtype_val in member_type_values:
                    formatted = [
                        item.format(
                            fof_type=fof_val or "{fof_type}",
                            member_id=member_val or "{member_id}",
                            member_type=mtype_val or "{member_type}",
                        )
                        for item in items
                    ]
                    expanded.append(
                        formatted[0] if len(formatted) == 1 else tuple(formatted)
                    )

    return expanded


class FileType(Enum):
    """
    Class that memorizes the distinction between file types
    """

    FOF = "fof"
    STATS = "csv"


@dataclass
class FileInfo:
    """
    Class that memorize the path and the type of a file.
    """

    path: str
    type: FileType = None

    def __post_init__(self):

        name = self.path.lower()

        if "fof" in name:
            self.type = FileType.FOF
            return
        if "csv" in name or "stats" in name:
            self.type = FileType.STATS
            return FileType.STATS

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if "," in first_line or ";" in first_line:
                    self.type = FileType.STATS
                    return
        except (OSError, FileNotFoundError):
            pass

        raise ValueError(f"Unknown file type for '{self.path}'")
