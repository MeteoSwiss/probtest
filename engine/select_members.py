"""
CLI for member selection

This script provides a command line interface for selecting members and testing
tolerance factors for statistical data.
It can find members and a corresponding tolerance factor that validate for all
stats files or test the tolerance of a given selection.
"""

import logging
import os
import sys
from datetime import datetime

import click

from engine.tolerance import tolerance
from util.click_util import cli_help
from util.dataframe_ops import test_stats_file_with_tolerances
from util.log_handler import logger


def find_members_and_factor_validating_for_all_stats_files(
    random_tolerance_file_name,
    stats_file_name,
    member_type,
    max_member_count,
    total_member_count,
    min_factor,
    max_factor,
):  # pylint: disable=too-many-positional-arguments
    """
    Find a minimal subset of ensemble members and a tolerance factor such that
    all statistics files are validated.

    Given a set of ensemble members, the algorithm attempts to identify a subset
    that satisfies specified tolerances.
    Starting with each ensemble member individually, the algorithm checks how
    many of the remaining ensemble members fall within the defined tolerances
    when compared to the current member.
    The ensemble member that results in the largest number of other members
    satisfying the tolerances is selected as the reference.

    This step is repeated, progressively validating additional members, until
    all ensemble members are validated or a predefined maximum number of
    iterations is reached. If the latter occurs, the tolerance factor is
    increased and the process restarts.
    """

    members_not_validating = set(range(1, total_member_count + 1))
    member_selection = set()

    for _ in range(max_member_count):

        member_with_minmal_fails = -1
        minimal_fails = set()

        for mem in members_not_validating:
            logger.info("checking member selection with additional member %s ...", mem)

            temp_member_selection = member_selection.union({mem})

            # creating tolerances
            context = click.Context(tolerance)
            context.invoke(
                tolerance,
                stats_file_name=stats_file_name,
                tolerance_file_name=random_tolerance_file_name,
                member_ids=list(temp_member_selection),
                member_type=member_type,
            )

            # Test selection (exclude random selection)
            validation_members = [
                m for m in members_not_validating if m not in temp_member_selection
            ]
            _, failed, _ = test_selection(
                stats_file_name,
                random_tolerance_file_name,
                validation_members,
                member_type,
                factor=min_factor,
            )

            if member_with_minmal_fails == -1:
                member_with_minmal_fails = mem
                minimal_fails = failed
            elif len(failed) < len(minimal_fails):
                member_with_minmal_fails = mem
                minimal_fails = failed

        if member_with_minmal_fails != -1:
            member_selection.add(member_with_minmal_fails)
            members_not_validating = minimal_fails
            logger.info(
                "Current member selection size %s, fails %s%%.\n",
                len(member_selection),
                int(len(minimal_fails) / total_member_count * 100),
            )

    if members_not_validating:
        # re-create tolerances with member selection
        context = click.Context(tolerance)
        context.invoke(
            tolerance,
            stats_file_name=stats_file_name,
            tolerance_file_name=random_tolerance_file_name,
            member_ids=member_selection,
            member_type=member_type,
        )
        for f in range(
            int(min_factor), int(max_factor) + 1, 5
        ):  # Try with bigger factor if max_member_count is not enough
            logger.info("Set factor to %s", f)

            # Test selection (exclude random selection)
            _, failed, most_common_vars = test_selection(
                stats_file_name,
                random_tolerance_file_name,
                members_not_validating,
                member_type,
                f,
            )
            if not failed:
                return sorted(member_selection), f
    else:
        return sorted(member_selection), min_factor

    logger.error(
        "ERROR: Could not find %s members, which pass for all stat files. "
        + "The most sensitive variable(s) is/are %s, which failed with the factor %s"
        + ". Consider removing this/those variable(s) from the "
        + "experiment and run again.",
        max_member_count,
        most_common_vars,
        max_factor,
    )
    sys.exit(1)


def test_selection(
    stats_file_name, tolerance_file_name, total_member_count, member_type, factor
):
    """
    Tests how many stats files pass the tolerance test for the selected members
    Returns the number of passed stats files and the variables which failed
    """

    if isinstance(total_member_count, int):
        members = list(range(1, total_member_count + 1))
    else:
        members = total_member_count
        total_member_count = len(members)

    total_member_count = len(members)

    passed = set()
    failed = set()

    # Change level to not get whole output from test_stats_file_with_tolerances
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.ERROR)

    variables = set()

    for mem in members:
        m_id = str(mem) if not member_type else member_type + "_" + str(mem)

        out, err, _ = test_stats_file_with_tolerances(
            tolerance_file_name,
            stats_file_name.format(member_id="ref"),
            stats_file_name.format(member_id=m_id),
            factor,
        )

        if out:
            passed.add(mem)
        else:
            failed.add(mem)
            var = set(index[1] for index in err[0].index)
            variables.update(var)

    variables = list(variables)

    # Reset logger level
    logging.getLogger().setLevel(original_level)
    logger.info(
        "... passing for %s out of %s members.\n",
        len(passed),
        total_member_count,
    )
    return passed, failed, variables


@click.command()
@click.option(
    "--experiment-name",
    help=cli_help["experiment_name"],
)
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
    "--selected-members-file-name",
    help=cli_help["selected_members_file_name"],
)
@click.option(
    "--tolerance-file-name",
    help=cli_help["tolerance_file_name"],
)
@click.option(
    "--member-type",
    type=str,
    default="",
    help=cli_help["member_type"],
)
@click.option(
    "--max-member-count",
    type=int,
    default=15,
    help=cli_help["max_member_count"],
)
@click.option(
    "--total-member-count",
    type=int,
    default=50,
    help=cli_help["total_member_count"],
)
@click.option(
    "--factor",
    type=float,
    default=5.0,
    help=cli_help["factor"],
)
@click.option(
    "--min-factor",
    type=float,
    default=5.0,
    help=cli_help["min_factor"],
)
@click.option(
    "--max-factor",
    type=float,
    default=50.0,
    help=cli_help["max_factor"],
)
# Selects members and writes them to a file together with the tolerance factor
def select_members(
    experiment_name,
    test_tolerance,
    stats_file_name,
    selected_members_file_name,
    tolerance_file_name,
    member_type,
    max_member_count,
    total_member_count,
    factor,
    min_factor,
    max_factor,
):  # pylint: disable=unused-argument, too-many-positional-arguments

    if max_member_count >= total_member_count:
        logger.error("ERROR: max_member_count must be smaller than total_member_count")
        sys.exit(1)

    if test_tolerance:
        # Test selection
        test_selection(
            stats_file_name,
            tolerance_file_name,
            total_member_count,
            member_type,
            factor,
        )
    else:
        random_tolerance_file_name = f"random_tolerance_{experiment_name}.csv"
        start_time = datetime.now()
        selection, factor = find_members_and_factor_validating_for_all_stats_files(
            random_tolerance_file_name,
            stats_file_name,
            member_type,
            max_member_count,
            total_member_count,
            min_factor,
            max_factor,
        )
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        logger.info("The member selection took %ss.", elapsed_time.total_seconds())

        # Write selection into a file
        selection = ",".join(map(str, selection))
        logger.info(
            "Writing selected members %s with tolerance factor %s to file %s",
            selection,
            int(factor),
            selected_members_file_name,
        )
        with open(selected_members_file_name, "w", encoding="utf-8") as file:
            file.write(selection + "\n")
            file.write("export FACTOR=" + str(int(factor)))

        # The last created file was successful
        logger.info("Writing tolerance file to %s", tolerance_file_name)
        os.rename(random_tolerance_file_name, tolerance_file_name)
