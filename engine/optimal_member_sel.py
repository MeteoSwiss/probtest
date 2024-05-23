import sys

import click
import numpy as np
import random
import os

from engine.check import check_intersection, check_variable
from engine.tolerance import tolerance
from util.click_util import CommaSeperatedInts, cli_help
from util.dataframe_ops import compute_rel_diff_dataframe, parse_probtest_csv
from util.log_handler import logger


def select_members(stats_file_name, member_num, member_type, total_member_num, factor):

    for i in range(1,101):
        random_numbers = random.sample(range(1,total_member_num+1),member_num)
        print('Randomly selected the following members: ', random_numbers)
        # Create tolerances from random members
        context = click.Context(tolerance)
        context.invoke(
            tolerance,
            stats_file_name=stats_file_name,
            tolerance_file_name='random_tolerance.csv',
            member_num=random_numbers,
            member_type=member_type,
            )
        # Test selection
        passed = test_selection(
            stats_file_name, 'random_tolerance.csv', total_member_num, member_type, factor
        )
        if (passed == total_member_num):
            break

    return random_numbers


def test_selection(
    stats_file_name, tolerance_file_name, total_member_num, member_type, factor
):


    input_file_ref = stats_file_name.format(member_id="ref")
    passed = 0
    df_tol = parse_probtest_csv(tolerance_file_name, index_col=[0, 1])
    df_tol *= factor
    df_ref = parse_probtest_csv(input_file_ref, index_col=[0, 1, 2])

    for m_num in range(1,total_member_num+1):
        m_id = str(m_num) if not member_type else member_type + "_" + str(m_num)

        df_cur = parse_probtest_csv(
            stats_file_name.format(member_id=m_id), index_col=[0, 1, 2]
        )

        # check if variables are available in reference file
        skip_test, df_ref, df_cur = check_intersection(df_ref, df_cur)
        if skip_test:  # No intersection
            logger.info(
                "ERROR: No intersection between variables in input and reference file."
            )
            exit(1)

        # compute relative difference
        diff_df = compute_rel_diff_dataframe(df_ref, df_cur)
        # take maximum over height
        diff_df = diff_df.groupby(["file_ID", "variable"]).max()

        out, err, tol = check_variable(diff_df, df_tol)

        if out:
            passed = passed + 1

    logger.info(
        "The tolerance test passed for {} out of {} references.".format(
            passed, total_member_num
        )
    )
    return passed


@click.command()
@click.option(
    "--test-tolerance/--no-test-tolerance",
    is_flag=True,
    help=cli_help["test_tolerance"],
)
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
    default="15",
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
@click.option(
    "--factor",
    type=float,
    help=cli_help["factor"],
)
def optimal_member_sel(
    test_tolerance,
    stats_file_name,
    tolerance_file_name,
    member_num,
    member_type,
    total_member_num,
    factor,
):

    if len(member_num) != 1:
        logger.info(
            "ERROR: The optimal member selection needs a single value for member_num."
        )
        exit(1)
    else:
        member_num = member_num[0]

    if test_tolerance:
        # Test selection
        test_selection(
            stats_file_name, tolerance_file_name, total_member_num, member_type, factor
        )
    else:
        selection = select_members(
            stats_file_name, member_num, member_type, total_member_num, factor
        )
        # The last created file was successful
        os.rename('random_tolerance.csv',tolerance_file_name)

    return
