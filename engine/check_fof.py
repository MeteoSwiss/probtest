"""
CLI for checking two fof files

This module provides a command-line interface (CLI) to check that
two given fof files are conistent.
"""

import hashlib

import click
import numpy as np
import xarray as xr

# variable that are not supposed to change across fof files
hash_vars = [
    "lat",
    "lon",
    "statid",
    "data_category",
    "sub_category",
    "obstype",
    "d_body",
    "d_veri",
    "d_2",
    "d_3",
    "l_body",
    "n_level",
    "center",
    "sub_center",
    "codetype",
    "ident",
    "time",
    "time_nomi",
    # "time_dbase",
    "z_station",
    "z_modsurf",
    "sta_corr",
    "index_x",
    "index_y",
    "mdlsfc",
    "instype",
    "obs_id",
    #  "source",
    "subset",
    "dbkz",
    "index_d",
    # "varno",
    "bcor",
    "level_typ",
    # "qual",
    "veri_model",
    "veri_run_type",
    "veri_run_class",
    "veri_initial_date",
    "veri_forecast_time",
    "veri_resolution",
    "veri_domain_size",
    "veri_description",
    "veri_ens_member",
    "veri_exp_id",
    "veri_operator_flag",
    "n_hdr",
    "n_body",
    "n_radar",
    "plevel",
]


def compute_hash_for_vars_and_attrs(ds):

    all_bytes = b""

    for var in hash_vars:
        if var in ds.data_vars:
            var_data = ds[var].values
            all_bytes += np.sort(var_data.ravel()).tobytes()

            for attr, value in sorted(ds[var].attrs.items()):
                all_bytes += str(attr).encode("utf-8") + str(value).encode("utf-8")

    return hashlib.md5(all_bytes).hexdigest()


def compare_arrays(arr1, arr2, var_name):
    total = arr1.size

    if np.array_equal(arr1, arr2):
        equal = total

    elif np.array_equal(arr1, arr2, equal_nan=True):
        equal = total

    else:
        mask_equal = arr1 == arr2
        equal = mask_equal.sum()
        percent = (equal / total) * 100
        print(f"Differences in '{var_name}': {percent:.2f}% equal")
        diff_idx = np.where(~mask_equal.ravel())[0]
        for i in diff_idx[:20]:
            print(f"   - ds1: {arr1.ravel()[i]}, ds2: {arr2.ravel()[i]}")

    return total, equal


def compare_nc_files(file1, file2):
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

    hash1 = compute_hash_for_vars_and_attrs(ds1)
    hash2 = compute_hash_for_vars_and_attrs(ds2)

    if hash1 == hash2:
        print("fof files are consistent")

        return

    print("fof files are NOT consistent")

    index_vars = ["lat", "lon", "statid"]
    total_elements_all = 0
    equal_elements_all = 0

    ds1_sorted = ds1.sortby(index_vars)
    ds2_sorted = ds2.sortby(index_vars)

    for var in hash_vars:
        # variable comparison
        if var in ds1_sorted.data_vars and var in ds2_sorted.data_vars:
            arr1 = ds1_sorted[var].values
            arr2 = ds2_sorted[var].values

            total, equal = compare_arrays(arr1, arr2, var)
            total_elements_all += total
            equal_elements_all += equal

        # attribute comparison
        if var in ds1_sorted.attrs and var in ds2_sorted.attrs:
            arr1 = np.array(ds1_sorted.attrs[var], dtype=object)
            arr2 = np.array(ds2_sorted.attrs[var], dtype=object)
            t, e = compare_arrays(arr1, arr2, var)
            total_elements_all += t
            equal_elements_all += e

    if total_elements_all > 0:
        percent_equal_all = (equal_elements_all / total_elements_all) * 100
        percent_diff_all = 100 - percent_equal_all
        print(f"Total percentage of equality: {percent_equal_all:.2f}%")
        print(f"Total percentage of difference: {percent_diff_all:.2f}%")


@click.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
def check_fof(file1, file2):

    if "SYNOP" in file1:
        ad = ["phase"]
        hash_vars.extend(ad)
    elif "AIREP" in file1:
        ad = ["tracking", "phase"]
        hash_vars.extend(ad)
    elif "PILOT" in file1:
        ad = ["tracking", "meas_type"]
        hash_vars.extend(ad)
    elif "TEMP" in file1:
        ad = ["tracking", "rad_corr", "meas_type", "dlat", "dlon"]
        hash_vars.extend(ad)

    compare_nc_files(file1, file2)


if __name__ == "__main__":
    check_fof()  # pylint: disable=no-value-for-parameter
