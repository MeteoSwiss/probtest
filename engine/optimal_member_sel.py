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
from scipy.stats import rankdata


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
            for m_num in range(1,total_member_num+1)
        )
    ]
    ndata = len(dfs)
    if ndata < 2:
        logger.critical(
            "not enough ensemble members to find optimal spread, got {} dataset. Abort.".format(ndata)
        )
        sys.exit(1)

    df_ref = parse_probtest_csv(stats_file_name.format(member_id="ref"), index_col=[0, 1, 2])
    vars=set(index[1] for index in df_ref.index)

    # Initialize a dictionary to store maximum values for each variable
    max_values_per_var = {var: [] for var in vars}

    # Iterate through each df_diff
    for i in range(total_member_num):
        df_diff = abs(df_ref - dfs[i])

        # Iterate through the variable names
        for var in vars:
            # Extract maximum value for the current variable in the current df_diff
            max_value = df_diff.loc[(slice(None), var), :].max().max()
            # Append the maximum value to the array corresponding to the variable
            max_values_per_var[var].append(max_value)

    # Initialize a dictionary to store rankings for each variable
    rankings_per_var = {}

    # Iterate through each variable
    for var, max_values in max_values_per_var.items():
        # Use rankdata to rank max values in descending order
        rankings = rankdata([-value for value in max_values], method='min')
        # Convert ranks to integers and store them for the current variable
        rankings_per_var[var] = [int(rank) for rank in rankings]

    return
