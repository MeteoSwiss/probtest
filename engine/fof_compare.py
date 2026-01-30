"""
CLI for checking two fof files

This module provides a command-line interface (CLI) to check that
two given fof files are consistent.
Veri data are not considered, only reports and observations are compared.
"""

import os

import click
import xarray as xr

from util.click_util import CommaSeparatedStrings, cli_help
from util.dataframe_ops import check_file_with_tolerances
from util.fof_utils import (
    clean_logger_file_if_only_details,
    create_tolerance_csv,
    get_log_file_name,
)
from util.log_handler import get_detailed_logger, logger
from util.utils import FileInfo


@click.command()
@click.argument("file1")
@click.argument("file2")
@click.option(
    "--fof-types",
    type=CommaSeparatedStrings(),
    default="",
    help=cli_help["fof_types"],
)
@click.option(
    "--tolerance",
    default=1e-12,
)
def fof_compare(file1, file2, fof_types, tolerance):

    for fof_type in fof_types:
        file1_path = file1.format(fof_type=fof_type)
        file2_path = file2.format(fof_type=fof_type)

        n_rows = xr.open_dataset(file1_path).sizes["d_body"]
        tolerance_file = create_tolerance_csv(n_rows, tolerance)

        out, err, tol = check_file_with_tolerances(
            tolerance_file,
            FileInfo(file1_path),
            FileInfo(file2_path),
            factor=1,
            rules="",
        )

        log_file_name = get_log_file_name(file1_path)
        if out:
            logger.info("Files are consistent!")

        else:
            logger.info("Files are NOT consistent!")

            logger.info("Complete output available in %s", log_file_name)
            if not err.empty:
                detailed_logger = get_detailed_logger(log_file_name)
                detailed_logger.info(
                    "Differences, veri_data outside of tolerance range"
                )
                detailed_logger.info(err)
                detailed_logger.info(tol)

        clean_logger_file_if_only_details(log_file_name)
        if os.path.exists(tolerance_file):
            os.remove(tolerance_file)


if __name__ == "__main__":
    fof_compare()  # pylint: disable=no-value-for-parameter
