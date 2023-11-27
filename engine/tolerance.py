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
def tolerance(stats_file_name, tolerance_file_name, member_num, member_type):
    if len(member_num) == 1:
        member_num = [i for i in range(1, member_num[0] + 1)]
    # read in stats files
    dfs = [
        parse_probtest_csv(stats_file_name.format(member_id=m_id), index_col=[0, 1, 2])
        for m_id in (
            (str(m_num) if not member_type else member_type + "_" + str(m_num))
            for m_num in member_num
        )
    ]
    dfs.append(
        parse_probtest_csv(stats_file_name.format(member_id="ref"), index_col=[0, 1, 2])
    )

    ndata = len(dfs)
    if ndata < 2:
        logger.critical(
            "not enough data to compute tolerance, got {} dataset. Abort.".format(ndata)
        )
        sys.exit(1)
    # get all possible combinations of the input data
    combs = list(itertools.product(range(ndata), range(ndata)))

    # do not use the i==j combinations
    combs = [(i, j) for i, j in combs if j < i]
    logger.info("computing tolerance from {} input combinations!".format(len(combs)))
    # compute relative differences for all combinations
    rdiff = [compute_rel_diff_dataframe(dfs[i], dfs[j]) for i, j in combs]
    # max-scan over height - drop height index
    rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff]
    # max over all combinations
    df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()

    # Take the cumulative maximum to only allow monotonically growing tolerances
    force_monotonic(df_max)

    tolerance_dir = os.path.dirname(tolerance_file_name)
    if tolerance_dir != "" and not os.path.exists(tolerance_dir):
        os.makedirs(tolerance_dir)
    logger.info("writing tolerance file to {}".format(tolerance_file_name))
    df_max.to_csv(tolerance_file_name)

    return
