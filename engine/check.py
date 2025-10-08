"""
CLI for Checking Data Files with Tolerances

This module defines a CLI to compare two data files (reference and current)
against specified tolerances.
It utilizes utility functions for testing statistical files with tolerances and
computing divergence between DataFrames.
"""

import sys

import click

from util.click_util import CommaSeperatedStrings, cli_help
from util.dataframe_ops import check_stats_file_with_tolerances, compute_division
from util.fof_utils import expand_zip
from util.log_handler import logger


@click.command()
@click.option(
    "--input-files-ref",
    type=CommaSeperatedStrings(),
    help=cli_help["input_files_ref"],
    default=None,
)
@click.option(
    "--input-files-cur",
    type=CommaSeperatedStrings(),
    help=cli_help["input_files_cur"],
    default=None,
)
@click.option(
    "--tolerance-files-name",
    type=CommaSeperatedStrings(),
    help=cli_help["tolerance_files_name"],
    default=None,
)
@click.option("--factor", type=float, help=cli_help["factor"])
@click.option(
    "--fof-type",
    default="",
    help=cli_help["fof_type"],
)
def check(
    input_files_ref,
    input_files_cur,
    tolerance_files_name,
    factor,
    fof_type,
):

    zipped = zip(input_files_ref, input_files_cur, tolerance_files_name)

    expanded_zip = expand_zip(zipped, fof_type)

    output_list = []

    for ref, cur, tol in expanded_zip:
        input_file_ref = ref
        input_file_cur = cur
        tolerance_file_name = tol

        out, err, tol = check_stats_file_with_tolerances(
            tolerance_file_name, input_file_ref, input_file_cur, factor
        )
        out = out[0]
        err = out[1]
        tol = out[2]
        file = out[3]

        if out:
            logger.info("RESULT: check PASSED for %s", file)

        else:
            logger.info("RESULT: check FAILED for %s", file)
            logger.info("Differences")
            logger.info(err)
            logger.info("\nTolerance")
            logger.info(tol)
            logger.info("\nError relative to tolerance")
            logger.info(compute_division(err, tol))

    sys.exit(0 if out else 1)
