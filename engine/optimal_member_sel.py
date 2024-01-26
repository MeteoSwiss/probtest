import itertools
import os
import sys

import click
import pandas as pd

from util.click_util import CommaSeperatedInts, cli_help
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
    "--member-num",
    type=CommaSeperatedInts(),
    default="10",
    help=cli_help["member_num"],
)
@click.option(
    "--member-type",
    type=str,
    default="",
    help=cli_help["member_type"],
)
@click.option(
    "--total-member-num",
    type=int,
    default=100,
    help=cli_help["total_member_num"],
)
def optimal_member_sel(stats_file_name, tolerance_file_name, member_num, member_type,total_member_num):
    if len(member_num) == 1:
        member_num = [i for i in range(1, member_num[0] + 1)]
    # read in stats files
    dfs = [
        parse_probtest_csv(stats_file_name.format(member_id=m_id), index_col=[0, 1, 2])
        for m_id in (
            (str(m_num) if not member_type else member_type + "_" + str(m_num))
            for m_num in range(1,total_member_num)
        )
    ]
    dfs.append(
        parse_probtest_csv(stats_file_name.format(member_id="ref"), index_col=[0, 1, 2])
    )

    ndata = len(dfs)
    if ndata < 2:
        logger.critical(
            "not enough ensemble members to find optimal spread, got {} dataset. Abort.".format(ndata)
        )
        sys.exit(1)

    return
