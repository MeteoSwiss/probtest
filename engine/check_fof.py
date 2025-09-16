"""
CLI for checking two fof files

This module provides a command-line interface (CLI) to check that
two given fof files are conistent.
"""
import xarray as xr
import click
import numpy as np
import hashlib


# variable that are not supposed to change across fof files
hash_vars = ["lat", "lon", "statid", "data_category", "sub_category", "obstype", "d_body", "d_veri", "d_2", "d_3",
               "l_body", "n_level", "center", "sub_center", "codetype", "ident", "time", "time_nomi", "time_dbase",
               "z_station", "z_modsurf", "r_state", "r_flags", "r_check", "sta_corr", "index_x", "index_y", "mdlsfc",
               "instype", "obs_id", "source", "subset", "dbkz", "index_d", "varno", "bcor",
               "level_typ", "qual","veri_model", "veri_run_type", "veri_run_class", "veri_initial_date",
                "veri_forecast_time", "veri_resolution", "veri_domain_size", "veri_description", "veri_ens_member",
                 "veri_exp_id", "veri_operator_flag", "n_hdr", "n_body", "n_radar"]




def compute_hash_for_vars_and_attrs(ds, hash_vars):

    all_bytes = b""

    for var in hash_vars:
        if var in ds.data_vars:
            var_data = ds[var].values
            all_bytes += np.sort(var_data.ravel()).tobytes()

            for attr, value in sorted(ds[var].attrs.items()):
                all_bytes += str(attr).encode("utf-8") + str(value).encode("utf-8")

    return hashlib.md5(all_bytes).hexdigest()


def compare_nc_files(file1, file2, hash_vars):
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

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

    hash1 = compute_hash_for_vars_and_attrs(ds1, hash_vars)
    hash2 = compute_hash_for_vars_and_attrs(ds2, hash_vars)


    if hash1 == hash2:
        print("fof files are consistent")
    else:
        print("fof files are NOT consistent")




@click.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))

def check_fof(file1, file2):

    compare_nc_files(file1, file2, hash_vars)

if __name__ == "__main__":
    check_fof()
