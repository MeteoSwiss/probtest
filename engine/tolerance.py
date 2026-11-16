"""
CLI for computing tolerance values from statistical datasets and from fof files.

This module reads statistical data from CSV files and fof data from netCDF files,
computes relative differences, and determines the tolerance levels for various
ensemble members.
"""

import os
import sys

import click

from util.click_util import CommaSeparatedInts, CommaSeparatedStrings, cli_help
from util.dataframe_ops import file_name_parser, has_enough_data
from util.log_handler import logger
from util.tolerance_methods import (
    METHOD_EXTREME_VALUES,
    METHOD_RELATIVE,
    TOLERANCE_METHODS,
    build_extreme_values_statistics,
    build_relative_tolerance,
)
from util.utils import FileInfo, FileType, expand_fof, expand_members


@click.command()
@click.option(
    "--ensemble-files",
    type=CommaSeparatedStrings(),
    default=None,
    help=cli_help["ensemble_files"],
)
@click.option(
    "--tolerance-files",
    type=CommaSeparatedStrings(),
    default=None,
    help=cli_help["tolerance_files_output"],
)
@click.option(
    "--member-ids",
    type=CommaSeparatedInts(),
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
    type=CommaSeparatedStrings(),
    default="",
    help=cli_help["fof_types"],
)
@click.option(
    "--minimum-tolerance",
    type=float,
    default="0.0",
    help=cli_help["minimum_tolerance"],
)
@click.option(
    "--method",
    type=click.Choice(TOLERANCE_METHODS),
    default=METHOD_RELATIVE,
    help=cli_help["tolerance_method"],
)
def tolerance(
    ensemble_files,
    tolerance_files,
    member_ids,
    member_type,
    fof_types,
    minimum_tolerance,
    method,
):  # pylint: disable=too-many-positional-arguments

    files_list = zip(ensemble_files, tolerance_files)
    expanded_zip = expand_fof(files_list, fof_types)

    for mem, tol in expanded_zip:

        ensemble_files = expand_members(
            mem, member_ids=member_ids, member_type=member_type
        )
        dfs = [
            file_name_parser[info.file_type](info.path)
            for file in ensemble_files
            for info in [FileInfo(file)]
        ]

        ref_info = FileInfo(mem.format(member_id="ref", member_type=""))
        if ref_info.file_type is FileType.FOF:
            ref_info.path = ref_info.path.replace("ref", "")
        df_ref = file_name_parser[ref_info.file_type](ref_info.path)

        has_enough_data(dfs)
        df_ref = df_ref["veri_data"] if ref_info.file_type is FileType.FOF else df_ref
        dfs = (
            [df["veri_data"] for df in dfs]
            if ref_info.file_type is FileType.FOF
            else dfs
        )

        if method == METHOD_EXTREME_VALUES:
            if ref_info.file_type is not FileType.STATS:
                logger.critical(
                    "The %s method only supports stats files. Abort.", method
                )
                sys.exit(1)
            df_max = build_extreme_values_statistics(df_ref, dfs)
        else:
            df_max = build_relative_tolerance(
                df_ref, dfs, ref_info.file_type, minimum_tolerance
            )

        tolerance_dir = os.path.dirname(tol)

        if tolerance_dir and not os.path.exists(tol):
            os.makedirs(tolerance_dir, exist_ok=True)

        logger.info("writing tolerance file to %s", tol)
        df_max.to_csv(tol)
