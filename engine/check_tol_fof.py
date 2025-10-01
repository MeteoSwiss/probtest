"""
CLI for Checking Data Files with Tolerances

This module defines a CLI to compare two data files (reference and current)
against specified tolerances.
It utilizes utility functions for testing statistical files with tolerances and
computing divergence between DataFrames.
"""

import sys

import click
import pandas as pd
import xarray as xr

from engine.tolerance_fof import split_feedback_dataset
from util.click_util import cli_help
from util.dataframe_ops import (
    check_intersection,
    check_variable,
    compute_division,
)
from util.log_handler import logger


def check_stats_file_with_tolerances(
    tolerance_file_name, input_file_ref, input_file_cur, factor
):
    ds_ref = xr.open_dataset(input_file_ref)
    _, ds_veri_ref = split_feedback_dataset(ds_ref)

    ds_cur = xr.open_dataset(input_file_cur)
    _, ds_veri_cur = split_feedback_dataset(ds_cur)

    skip_test, df_ref, df_cur = check_intersection(ds_veri_ref, ds_veri_cur)

    if skip_test:
        logger.error("RESULT: check FAILED")
        sys.exit(1)

    diff_df = (df_ref["veri_data"] - df_cur["veri_data"]) / (
        1.0 + df_ref["veri_data"].abs()
    )

    ds_tol = pd.read_csv(tolerance_file_name, index_col=0)
    ds_tol = ds_tol * factor

    out, err, tol = check_variable(diff_df, ds_tol)

    return out, err, tol


@click.command()
@click.option(
    "--input-file-ref",
    help=cli_help["input_file_ref"],
)
@click.option(
    "--input-file-cur",
    help=cli_help["input_file_cur"],
)
@click.option(
    "--tolerance-file-name",
    help=cli_help["tolerance_file_name"],
)
@click.option(
    "--factor",
    type=float,
    help=cli_help["factor"],
)
def check_tol_fof(input_file_ref, input_file_cur, tolerance_file_name, factor):

    out, err, tol = check_stats_file_with_tolerances(
        tolerance_file_name, input_file_ref, input_file_cur, factor
    )

    if out:
        logger.info("RESULT: check PASSED!")
    else:
        logger.info("RESULT: check FAILED")
        logger.info("Differences")
        logger.info(err)
        logger.info("\nTolerance")
        logger.info(tol)
        logger.info("\nError relative to tolerance")
        logger.info(compute_division(err, tol))

    sys.exit(0 if out else 1)
