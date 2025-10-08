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
    compute_rel_diff_dataframe,
    enough_data,
    force_monotonic,
    parse_probtest_csv,
    parse_probtest_fof,
)
from util.fof_utils import expand_zip, order_stat_fof
from util.log_handler import logger


@click.command()
@click.option(
    "--files_name",
    type=CommaSeperatedStrings(),
    default=None,
    help=cli_help["files_name"],
)
@click.option(
    "--tolerance-files-name",
    type=CommaSeperatedStrings(),
    default=None,
    help=cli_help["tolerance_files_name"],
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
    "--fof-type",
    default="",
    help=cli_help["fof_type"],
)
def tolerance(
    files_name,
    tolerance_files_name,
    member_ids,
    member_type,
    fof_type,
):

    expanded = expand_zip(files_name, fof_type, member_ids, member_type)
    results = []
    tolerance_files_name = order_stat_fof(tolerance_files_name)

    if any("stat" in name.lower() for name in files_name):
        dfs = [
            parse_probtest_csv(stat[0], index_col=[0, 1, 2])
            for stat in expanded
            if "stat" in stat[0].lower()
        ]
        df_ref = parse_probtest_csv(
            files_name[0].format(member_id="ref"), index_col=[0, 1, 2]
        )
        enough_data(dfs)

        rdiff = [compute_rel_diff_dataframe(df_ref, df) for df in dfs]
        rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff]

        df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()
        force_monotonic(df_max)

        results.append([df_max, tolerance_files_name[0]])

    if any("fof" in name.lower() for name in files_name):
        fof_list = fof_type.split(",") if fof_type else []
        dfs_fof, dfs_ref_fof = {}, {}

        for tfof in fof_list:
            dfs_fof[tfof] = [
                parse_probtest_fof(stat[0])
                for stat in expanded
                if "fof" in stat[0].lower() and tfof in stat[0]
            ]
            enough_data(dfs_fof[tfof])

            dfs_ref_fof[tfof] = parse_probtest_fof(
                files_name[1].format(member_id="ref", fof_type=tfof)
            )

            rdiff = [
                compute_rel_diff_dataframe(
                    dfs_ref_fof[tfof]["veri_data"], df["veri_data"]
                )
                for df in dfs_fof[tfof]
            ]
            df_max = pd.concat(rdiff, axis=1).max(axis=1)

            results.append([df_max, tolerance_files_name[1].format(fof_type=tfof)])

    for df_max, tol_file in results:
        tolerance_dir = os.path.dirname(tol_file)
        if tolerance_dir and not os.path.exists(tolerance_dir):
            os.makedirs(tolerance_dir)
        logger.info("writing tolerance file to %s", tol_file)
        df_max.to_csv(tol_file)
