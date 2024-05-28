import os
import random
from collections import Counter
from datetime import datetime

import click

from engine.check import check_intersection, check_variable
from engine.tolerance import tolerance
from util.click_util import CommaSeperatedInts, cli_help
from util.dataframe_ops import compute_rel_diff_dataframe, parse_probtest_csv
from util.log_handler import logger


def select_members(stats_file_name, member_num, member_type, total_member_num, factor):

    members = [i for i in range(1, total_member_num + 1)]

    max_members = 15  # Selection should not have more than 15 members
    for f in range(
        int(factor), 51, 5
    ):  # Try with bigger factor if 15 members are not enough
        logger.info("Set factor to {}".format(f))
        for mem_num in range(member_num, max_members + 1):
            logger.info("Try with {} members.".format(mem_num))
            max_passed = 1
            vars = []
            for i in range(1, 51):
                random_numbers = random.sample(range(1, total_member_num + 1), mem_num)
                logger.info(
                    "Test {} with {} randomly selected members and factor {}.".format(
                        i, mem_num, f
                    )
                )
                # Create tolerances from random members
                context = click.Context(tolerance)
                context.invoke(
                    tolerance,
                    stats_file_name=stats_file_name,
                    tolerance_file_name="random_tolerance.csv",
                    member_num=random_numbers,
                    member_type=member_type,
                )
                # Test selection (not randomly selected members)
                valid_members = [item for item in members if item not in random_numbers]
                passed, new_vars = test_selection(
                    stats_file_name,
                    "random_tolerance.csv",
                    valid_members,
                    member_type,
                    f,
                )
                vars.extend(new_vars)

                duplicates = {item: count for item, count in Counter(vars).items()}
                sorted_duplicates = dict(
                    sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
                )
                # The following is to save computing time
                if i < 33:
                    if max_passed < passed:
                        max_passed = passed
                    # The more combs were tested
                    # the higher should the success rate be to continue
                    tested_stats = total_member_num - mem_num
                    if max_passed < i * 0.03 * tested_stats:
                        break

                if passed == total_member_num - mem_num:
                    return random_numbers

    max_count = max(sorted_duplicates.values())
    most_common_vars = [
        item for item, count in sorted_duplicates.items() if count == max_count
    ]
    logger.info(
        (
            "ERROR: Could not find {} random members, which pass for all stat files. "
            + "The most sensitive variable(s) is/are {}, which failed for {} out of {} "
            + "random selections. Consider removing this/those variable(s) from the "
            + "experiment and run again."
        ).format(max_members, most_common_vars, max_count, i)
    )
    exit(1)


def test_selection(
    stats_file_name, tolerance_file_name, total_member_num, member_type, factor
):

    if type(total_member_num) is int:
        total_member_num = [i for i in range(1, total_member_num + 1)]

    input_file_ref = stats_file_name.format(member_id="ref")
    passed = 0
    df_tol = parse_probtest_csv(tolerance_file_name, index_col=[0, 1])
    df_tol *= factor
    df_ref = parse_probtest_csv(input_file_ref, index_col=[0, 1, 2])

    vars = []
    for m_num in total_member_num:
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

        if not out:
            var = set(index[1] for index in err[0].index)
            var = list(var)
            vars.extend(var)
        else:
            passed = passed + 1

    vars = list(set(vars))

    logger.info(
        "The tolerance test passed for {} out of {} references.".format(
            passed, len(total_member_num)
        )
    )
    return passed, vars


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
        start_time = datetime.now()
        selection = select_members(
            stats_file_name, member_num, member_type, total_member_num, factor
        )
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        logger.info(
            "The optimal member selection took {}s.".format(
                elapsed_time.total_seconds()
            )
        )
        logger.info("Selected members: {}".format(selection))
        # The last created file was successful
        os.rename("random_tolerance.csv", tolerance_file_name)

    return
