"""
CLI for Checking Data Files with Tolerances

This module defines a CLI to compare two data files (reference and current)
against specified tolerances.
It utilizes utility functions for testing statistical and fof files with
tolerances and computing divergence between DataFrames.
"""

import sys

import click

from util.click_util import CommaSeparatedStrings, cli_help
from util.dataframe_ops import check_file_with_tolerances, compute_division
from util.fof_utils import expand_zip
from util.log_handler import logger


@click.command()
@click.option(
    "--reference_files",
    type=CommaSeparatedStrings(),
    help=cli_help["reference_files"],
    default=None,
)
@click.option(
    "--current_files",
    type=CommaSeparatedStrings(),
    help=cli_help["current_files"],
    default=None,
)
@click.option(
    "--tolerance-files",
    type=CommaSeparatedStrings(),
    help=cli_help["tolerance_files_input"],
    default=None,
)
@click.option("--factor", type=float, help=cli_help["factor"])
@click.option(
    "--fof-types",
    type=CommaSeparatedStrings(),
    default="",
    help=cli_help["fof_types"],
)
def check(
    reference_files,
    current_files,
    tolerance_files,
    factor,
    fof_types,
):

    zipped = zip(reference_files, current_files, tolerance_files)

    expanded_zip = expand_zip(zipped, fof_types)

    all_out = True

    for reference_file, current_file, tolerance_file in expanded_zip:

        out, err, tol = check_file_with_tolerances(
            tolerance_file, reference_file, current_file, factor
        )

        if out:
            logger.info("RESULT: check PASSED for %s", current_file)
        else:
            logger.info("RESULT: check FAILED for %s", current_file)
            logger.info("Differences")
            logger.info(err)
            logger.info("\nTolerance")
            logger.info(tol)
            logger.info("\nError relative to tolerance")
            logger.info(compute_division(err, tol))
            all_out = False

    sys.exit(0 if all_out else 1)
