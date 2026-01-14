"""
CLI for performance checks

This module provides a command-line interface (CLI) tool for evaluating the
performance of a current experiment by comparing its timing data with a
reference.
It assesses whether the current experiment's runtime is within acceptable limits
based on specified parameters and provides feedback on whether the performance
check has passed or failed.
"""

import sys
from typing import List

import click

from util.click_util import CommaSeparatedStrings, cli_help
from util.log_handler import logger
from util.tree import TimingTree


@click.command()
@click.option(
    "--timing-current",
    help=cli_help["timing_current"],
)
@click.option(
    "--timing-reference",
    help=cli_help["timing_reference"],
)
@click.option(
    "--measurement-uncertainty",
    help=cli_help["measurement_uncertainty"],
    type=float,
    default=2,
)
@click.option(
    "--tolerance-factor", help=cli_help["tolerance_factor"], type=float, default=1.1
)
@click.option(
    "--new-reference-threshold",
    help=cli_help["new_reference_threshold"],
    type=float,
    default=0.95,
)
@click.option(
    "--timer-sections",
    type=CommaSeparatedStrings(),
    help=(
        "Comma-separated list of timer sections to evaluate "
        "(e.g. 'total,physics,nh_solve')."
    ),
    default="total",
)
@click.option("--i-table", type=int, help=cli_help["i_table"], default=-1)
def performance_check(
    timing_current: str,
    timing_reference: str,
    i_table: int,
    measurement_uncertainty: float,
    tolerance_factor: float,
    new_reference_threshold: float,
    timer_sections: List[str],
):  # pylint: disable=too-many-positional-arguments

    if measurement_uncertainty < 0:
        logger.error("measurement_uncertainty needs to be positive")
    if tolerance_factor < 1:
        logger.error("tolerance_factor needs to be greater than 1")
    if new_reference_threshold < 0 or new_reference_threshold > 1:
        logger.error("new_reference_threshold needs to be between 0 and 1")

    cur_tt = TimingTree.from_json(timing_current)
    ref_tt = TimingTree.from_json(timing_reference)

    cur_times = cur_tt.extract_timings(i_table=i_table, timer_sections=timer_sections)
    ref_times = ref_tt.extract_timings(i_table=i_table, timer_sections=timer_sections)

    allowed_times = (ref_times + measurement_uncertainty) * tolerance_factor

    # Logging per region
    logger.info("Timing comparison per region:")
    for section in timer_sections:
        logger.info(
            "%-12s current=%8.3f  reference=%8.3f  allowed=%8.3f",
            section,
            cur_times[section],
            ref_times[section],
            allowed_times[section],
        )

    # Determine which regions passed
    passed_mask = cur_times <= allowed_times
    passed = passed_mask.all()  # overall pass only if all regions pass

    if passed:
        logger.info("RESULT: performance_check PASSED!")

        # Check for regions that are significantly faster than reference
        faster_mask = cur_times < ref_times * new_reference_threshold
        if faster_mask.any():
            fast_regions = list(faster_mask[faster_mask].index)
            logger.info(
                "The following regions are significantly faster than reference: %s. "
                "Consider updating the reference.",
                ", ".join(fast_regions),
            )
        sys.exit(0)
    else:
        # Identify regions that failed
        failed_regions = list(passed_mask[~passed_mask].index)
        logger.info(
            "RESULT: performance_check FAILED (regions: %s)",
            ", ".join(failed_regions),
        )
        sys.exit(1)
