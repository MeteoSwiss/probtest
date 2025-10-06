"""
CLI for computing tolerance values from statistical datasets and from fof files.

This module reads statistical data from CSV files, computes relative differences,
and determines the tolerance levels for various ensemble members. Same is done
for fof files.
"""

import os
import sys

import click
import pandas as pd

from util.click_util import CommaSeperatedInts, cli_help
from util.dataframe_ops import (
    compute_rel_diff_dataframe,
    parse_probtest_csv,
    parse_probtest_fof,
)
from util.log_handler import logger


@click.command()
@click.option(
    "--stats-file-name",
    default=None,
    help=cli_help["stats_file_name"],
)
@click.option(
    "--tolerance-file-name",
    default=None,
    help=cli_help["tolerance_file_name"],
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
@click.option("--fof-file-name", help=cli_help["fof_file_name"], default=None)
@click.option(
    "--tolerance-file-fof-name", help=cli_help["tolerance_file_fof_name"], default=None
)
@click.option(
    "--fof-type",
    default="",
    help=cli_help["fof_type"],
)
def tolerance(
    stats_file_name,
    tolerance_file_name,
    member_ids,
    member_type,
    fof_file_name,
    tolerance_file_fof_name,
    fof_type,
):  # pylint: disable=too-many-positional-arguments

    jobs = []
    if fof_file_name:
        fof_list = fof_type.split(",") if fof_type else []
        for ftypes in fof_list:
            files = [
                fof_file_name.format(member_id=str(m_id), fof_type=ftypes)
                for m_id in member_ids
            ]
            ref_file = fof_file_name.format(member_id="ref", fof_type=ftypes)
            out_file = tolerance_file_fof_name.format(fof_type=ftypes)
            jobs.append(("fof", files, ref_file, out_file))

    if stats_file_name:

        files = [
            stats_file_name.format(member_id=m_id)
            for m_id in (
                (str(m_id) if not member_type else member_type + "_" + str(m_id))
                for m_id in member_ids
            )
        ]
        print(files)

        ref_file = stats_file_name.format(member_id="ref", member_type="")
        out_file = tolerance_file_name
        jobs.append(("stats", files, ref_file, out_file))

    for job_type, files, ref_file, out_file in jobs:
        if job_type == "fof":
            dfs = [parse_probtest_fof(f) for f in files]
            df_ref = parse_probtest_fof(ref_file)
            rdiff = [
                compute_rel_diff_dataframe(df_ref["veri_data"], df["veri_data"])
                for df in dfs
            ]
            df_max = pd.concat(rdiff, axis=1).max(axis=1)

        else:
            dfs = [parse_probtest_csv(f, index_col=[0, 1, 2]) for f in files]
            df_ref = parse_probtest_csv(ref_file, index_col=[0, 1, 2])
            rdiff = [compute_rel_diff_dataframe(df_ref, df) for df in dfs]
            rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff]
            df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()

        ndata = len(dfs)
        if ndata < 1:
            logger.critical(
                "not enough data to compute tolerance, got %s dataset. Abort.", ndata
            )
            sys.exit(1)

        logger.info("computing tolerance from %s ensemble members!", ndata)

        tolerance_dir = os.path.dirname(out_file)
        if tolerance_dir != "" and not os.path.exists(tolerance_dir):
            os.makedirs(tolerance_dir)
        logger.info("writing tolerance file to %s", out_file)
        df_max.to_csv(out_file)

    # if fof_file_name:
    #     fof_list = fof_type.split(",") if fof_type else []
    #     for ftypes in fof_list:
    #         dfs_fof = ([parse_probtest_fof(fof_file_name.format(member_id=m_id,
    # fof_type=ftypes))
    #         for m_id in (str(m_id) for m_id in member_ids)])

    #         df_ref_fof = parse_probtest_fof(fof_file_name.format(member_id="ref",
    # fof_type=ftypes))

    #         ndata = len(dfs_fof)
    #         if ndata < 1:
    #             logger.critical(
    #                 "not enough data to compute tolerance, got %s dataset.
    # Abort.", ndata
    #             )
    #             sys.exit(1)

    #         logger.info("computing tolerance from %s ensemble members!", ndata)

    #         rdiff_fof = [compute_rel_diff_dataframe(df_ref_fof['veri_data'],
    # df['veri_data']) for df in dfs_fof]

    #         df_max = pd.concat(rdiff_fof, axis=1).max(axis=1)
    #         tol_file = tolerance_file_fof_name.format(fof_type=ftypes)
    #         tolerance_dir = os.path.dirname(tol_file)
    #         if tolerance_dir != "" and not os.path.exists(tolerance_dir):
    #             os.makedirs(tolerance_dir)
    #         logger.info("writing tolerance file to %s", tol_file)
    #         df_max.to_csv(tol_file)

    # if stats_file_name:
    #     # read in stats files
    #     dfs = [
    #         parse_probtest_csv(stats_file_name.format(member_id=m_id),
    # index_col=[0, 1, 2])
    #         for m_id in (
    #             (str(m_id) if not member_type else member_type + "_" + str(m_id))
    #             for m_id in member_ids
    #         )
    #     ]
    #     df_ref = parse_probtest_csv(
    #         stats_file_name.format(member_id="ref"), index_col=[0, 1, 2]
    #     )

    #     ndata = len(dfs)
    #     if ndata < 1:
    #         logger.critical(
    #             "not enough data to compute tolerance, got %s dataset.
    # Abort.", ndata)
    #         sys.exit(1)

    #     logger.info("computing tolerance from %s ensemble members!", ndata)
    #     # compute relative differences for all combinations
    #     rdiff = [compute_rel_diff_dataframe(df_ref, df) for df in dfs]
    #     rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff]
    #     # max over all combinations
    #     df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()
    #     tolerance_dir = os.path.dirname(tolerance_file_name)
    #     if tolerance_dir != "" and not os.path.exists(tolerance_dir):
    #         os.makedirs(tolerance_dir)
    #     logger.info("writing tolerance file to %s", tolerance_file_name)
    #     df_max.to_csv(tolerance_file_name)
