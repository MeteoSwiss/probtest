"""
CLI for computing tolerance values from statistical datasets and from fof files.

This module reads statistical data from CSV files and fof data from netCDF files,
computes relative differences, and determines the tolerance levels for various
ensemble members.
"""

import os

import click
import pandas as pd

from util.click_util import CommaSeperatedInts, CommaSeperatedStrings, cli_help
from util.dataframe_ops import (
    compute_rel_diff_dataframe,
    file_name_parser,
    force_monotonic,
    has_enough_data,
)
from util.fof_utils import FileType, expand_zip, get_file_type
from util.log_handler import logger


@click.command()
@click.option(
    "--ensemble_files",
    type=CommaSeperatedStrings(),
    default=None,
    help=cli_help["ensemble_files"],
)
@click.option(
    "--tolerance-files",
    type=CommaSeperatedStrings(),
    default=None,
    help=cli_help["tolerance_files_output"],
)
@click.option(
    "--member-ids",
    type=CommaSeperatedInts(),
    default="1,2,3,4,5,6,7,8,9,10",
    help=cli_help["member_ids"],
)
@click.option(
    "--member-type",
    type=str,
    default="",
    help=cli_help["member_type"],
)
@click.option(
    "--fof-types",
    type=CommaSeperatedStrings(),
    default="",
    help=cli_help["fof_types"],
)
def tolerance(
    ensemble_files,
    tolerance_files,
    member_ids,
    member_type,
    fof_types,
):

    files_list = zip(ensemble_files, tolerance_files)
    expanded_zip = expand_zip(files_list, fof_types)

    for mem, tol in expanded_zip:

        ensemble_files = expand_zip(mem, member_ids=member_ids, member_type=member_type)

        file_type = get_file_type(mem)

        dfs = [file_name_parser[file_type](file) for file in ensemble_files]
        df_ref = file_name_parser[file_type](mem.format(member_id="ref"))

        has_enough_data(dfs)
        df_ref = df_ref["veri_data"] if file_type is FileType.FOF else df_ref
        dfs = [df["veri_data"] for df in dfs] if file_type is FileType.FOF else dfs

        rdiff = [compute_rel_diff_dataframe(df_ref, df) for df in dfs]

        if file_type is FileType.STATS:
            rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff]
            df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()
            force_monotonic(df_max)

        elif file_type is FileType.FOF:
            df_max = pd.concat(rdiff, axis=1).max(axis=1)

        tolerance_dir = os.path.dirname(tol)
        print(tolerance_dir)
        if tolerance_dir and not os.path.exists(tol):
            os.makedirs(tol)
        logger.info("writing tolerance file to %s", tol)
        df_max.to_csv(tol)
