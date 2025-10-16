"""
CLI for computing tolerance values from statistical datasets

This module reads statistical data from CSV files, computes relative differences,
and determines the tolerance levels for various ensemble members.
"""

import os
import sys

import click
import pandas as pd

from util.click_util import CommaSeparatedInts, cli_help
from util.dataframe_ops import (
    compute_rel_diff_dataframe,
    force_monotonic,
    parse_probtest_csv,
)
from util.log_handler import logger


@click.command()
@click.option(
    "--stats-file-name",
    help=cli_help["stats_file_name"],
)
@click.option(
    "--tolerance-file-name",
    help=cli_help["tolerance_file_name"],
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
    "--minimum-tolerance",
    type=float,
    default="0.0",
    help=cli_help["minimum_tolerance"],
)
def tolerance(
    stats_file_name, tolerance_file_name, member_ids, member_type, minimum_tolerance
):

    # read in stats files
    dfs = [
        parse_probtest_csv(stats_file_name.format(member_id=m_id), index_col=[0, 1, 2])
        for m_id in (
            (str(m_id) if not member_type else member_type + "_" + str(m_id))
            for m_id in member_ids
        )
    ]
    df_ref = parse_probtest_csv(
        stats_file_name.format(member_id="ref"), index_col=[0, 1, 2]
    )

    ndata = len(dfs)
    if ndata < 1:
        logger.critical(
            "not enough data to compute tolerance, got %s dataset. Abort.", ndata
        )
        sys.exit(1)

    logger.info("computing tolerance from %s ensemble members!", ndata)
    # compute relative differences for all combinations
    rdiff = [compute_rel_diff_dataframe(df_ref, df) for df in dfs]
    # max-scan over height - drop height index
    rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff]
    # max over all combinations
    df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()

    # Take the cumulative maximum to only allow monotonically growing tolerances
    force_monotonic(df_max)

    # Set a fixed tolerance value in case these became exactly zero
    df_max = df_max.map(lambda x: minimum_tolerance if x < minimum_tolerance else x)

    tolerance_dir = os.path.dirname(tolerance_file_name)
    if tolerance_dir != "" and not os.path.exists(tolerance_dir):
        os.makedirs(tolerance_dir)
    logger.info("writing tolerance file to %s", tolerance_file_name)
    df_max.to_csv(tolerance_file_name)
