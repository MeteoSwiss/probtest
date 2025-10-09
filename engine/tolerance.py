"""
CLI for computing tolerance values from statistical datasets and from fof files.

This module reads statistical data from CSV files, computes relative differences,
and determines the tolerance levels for various ensemble members. Same is done
for fof files.
"""

import os

import click
import pandas as pd

from util.click_util import CommaSeperatedInts, CommaSeperatedStrings, cli_help
from util.dataframe_ops import (
    FileType,
    compute_rel_diff_dataframe,
    force_monotonic,
    has_enough_data,
    parse_probtest_csv,
    parse_probtest_fof,
)
from util.fof_utils import expand_zip
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

    expanded = expand_zip(ensemble_files, fof_types, member_ids="{member_id}")
    expanded_tol = expand_zip(tolerance_files, fof_types)

    files_list = zip(expanded, expanded_tol)

    for item in files_list:
        if any(FileType.STATS.value in i for i in item[0]):

            stats_files = expand_zip(
                item[0], member_ids=member_ids, member_type=member_type
            )

            dfs = [
                parse_probtest_csv(file[0], index_col=[0, 1, 2]) for file in stats_files
            ]

            df_ref = parse_probtest_csv(
                item[0][0].format(member_id="ref"), index_col=[0, 1, 2]
            )

            has_enough_data(dfs)

            rdiff = [compute_rel_diff_dataframe(df_ref, df) for df in dfs]
            rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff]

            df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()
            force_monotonic(df_max)

        elif any(FileType.FOF.value in f for f in item[0]):

            fof_files = expand_zip(item[0], member_ids=member_ids)

            dfs_fof = [parse_probtest_fof(file[0]) for file in fof_files]

            has_enough_data(dfs_fof)

            df_ref_fof = parse_probtest_fof(item[0][0].format(member_id="ref"))

            rdiff = [
                compute_rel_diff_dataframe(df_ref_fof["veri_data"], df["veri_data"])
                for df in dfs_fof
            ]
            df_max = pd.concat(rdiff, axis=1).max(axis=1)

        tolerance_file = repr(item[1][0])
        tolerance_dir = os.path.dirname(tolerance_file)
        print(tolerance_dir)
        if tolerance_dir and not os.path.exists(tolerance_dir):
            os.makedirs(tolerance_dir)
        logger.info("writing tolerance file to %s", tolerance_file)
        df_max.to_csv(tolerance_file)
