"""
CLI for Checking Data Files with Tolerances

This module defines a CLI to compare two data files (reference and current)
against specified tolerances.
It utilizes utility functions for testing statistical files with tolerances and
computing divergence between DataFrames.
"""

import sys

import click

from util.click_util import cli_help
from util.dataframe_ops import check_stats_file_with_tolerances, compute_division
from util.log_handler import logger


@click.command()
@click.option("--input-file-ref", help=cli_help["input_file_ref"], default=None)
@click.option("--input-file-cur", help=cli_help["input_file_cur"], default=None)
@click.option(
    "--tolerance-file-name", help=cli_help["tolerance_file_name"], default=None
)
@click.option("--factor", type=float, help=cli_help["factor"])
@click.option(
    "--input-file-fof-ref",
    default=None,
    help=cli_help["input_file_fof_ref"],
)
@click.option(
    "--input-file-fof-cur",
    default=None,
    help=cli_help["input_file_fof_cur"],
)
@click.option(
    "--tolerance-file-fof-name",
    default=None,
    help=cli_help["tolerance_file_fof_name"],
)
@click.option(
    "--fof-type",
    default="",
    help=cli_help["fof_type"],
)
def check(
    input_file_ref,
    input_file_cur,
    tolerance_file_name,
    factor,
    input_file_fof_ref,
    input_file_fof_cur,
    tolerance_file_fof_name,
    fof_type,
):  # pylint: disable=too-many-positional-arguments

    output_list = []

    if input_file_ref:
        out, err, tol = check_stats_file_with_tolerances(
            tolerance_file_name, input_file_ref, input_file_cur, factor
        )
        output_list.append([out, err, tol, input_file_cur])

    if input_file_fof_cur:
        fof_list = fof_type.split(",") if fof_type else []
        for ftypes in fof_list:
            out, err, tol = check_stats_file_with_tolerances(
                tolerance_file_fof_name.format(fof_type=ftypes),
                input_file_fof_ref.format(fof_type=ftypes),
                input_file_fof_cur.format(fof_type=ftypes),
                factor,
                type_f="fof",
            )
            output_list.append(
                (out, err, tol, input_file_fof_cur.format(fof_type=ftypes))
            )

    for output in output_list:
        out = output[0]
        err = output[1]
        tol = output[2]
        file = output[3]

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
