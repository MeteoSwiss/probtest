"""
CLI for computing tolerance values from fof files.

This module computes relative differences and determines
the tolerance levels for various ensemble members.
"""

import os
import sys

import click
import numpy as np
import pandas as pd
import xarray as xr

from engine.check_fof import get_dims
from util.click_util import cli_help
from util.file_system import file_names_from_pattern
from util.log_handler import logger


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

    vars_shape2.append("veri_data")
    ds_obs = dataset[vars_shape2]
    sort_keys_obs = ["lat", "lon", "statid", "varno", "level", "time_nomi"]
    ds_obs_sorted = ds_obs.sortby(sort_keys_obs)

    vars_to_drop = [
        "obs",
        "bcor",
        "level_typ",
        "level_sig",
        "state",
        "flags",
        "check",
        "e_o",
        "qual",
        "plevel",
    ]

    ds_veri = ds_obs_sorted.drop_vars(vars_to_drop)

    ds_veri = ds_veri.to_dataframe().reset_index()

    return ds_report_sorted, ds_veri


@click.command()
@click.option(
    "--file-id",
    nargs=2,
    type=str,
    multiple=True,
    metavar="FILE_TYPE FILE_PATTERN",
    help=cli_help["file_id"],
)
@click.option(
    "--tolerance-file-name",
    help=cli_help["tolerance_file_name"],
)
# @click.option(
#     "--member-ids",
#     type=CommaSeperatedInts(),
#     default="1,2,3,4,5,6,7,8,9",
#     help=cli_help["member_ids"],
# )
@click.option(
    "--model_output_dir",
    help=cli_help["model_output_dir"],
)
# @click.option(
#     "--member-type",
#     type=str,
#     default="",
#     help=cli_help["member_type"],
# )
# tolerance, member_ids,
def tolerance_fof(file_id, model_output_dir, tolerance_file_name):
    for _, file_pattern in file_id:
        input_files, err = file_names_from_pattern(model_output_dir, file_pattern)
        if err > 0:
            logger.info(
                "Can not find any files for file_pattern %s. Continue.", file_pattern
            )
            continue

    ref_file = next((f for f in input_files if "ref" in f), None)
    ndata = len(input_files)
    if ndata < 1:
        logger.critical(
            "not enough data to compute tolerance, got %s dataset. Abort.", ndata
        )
        sys.exit(1)

    logger.info("computing tolerance from %s ensemble members!", ndata)

    ref_ds = xr.open_dataset(model_output_dir + "/" + ref_file)
    _, ds_veri_ref = split_feedback_dataset(ref_ds)
    rdiff = []
    var_name = "veri_data"

    for i in input_files:
        ds = xr.open_dataset(model_output_dir + "/" + i)
        _, ds_veri = split_feedback_dataset(ds)
        ds_veri[var_name] = (ds_veri_ref[var_name] - ds_veri[var_name]) / (
            1.0 + ds_veri_ref[var_name].abs()
        )
        rdiff.append(ds_veri)

    concatenated = pd.concat([df[var_name] for df in rdiff], axis=1)

    max_values = concatenated.max(axis=1)

    tolerance_dir = os.path.dirname(tolerance_file_name)

    if tolerance_dir != "" and not os.path.exists(tolerance_dir):
        os.makedirs(tolerance_dir)
    logger.info("writing tolerance file to %s", tolerance_file_name)
    max_values.to_csv(tolerance_file_name)
