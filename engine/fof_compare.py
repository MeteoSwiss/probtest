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
    create_tolerance_csv,
    write_tolerance_log,
)
from util.utils import FileInfo
from util.log_handler import logger, initialize_detailed_logger

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
    "--tol",
    default=1e-12,
)
def fof_compare(file1, file2, fof_types,tol):

    for ft in fof_types:
        file1_path = file1.format(fof_type=ft)
        file2_path = file2.format(fof_type=ft)

        n_rows = xr.open_dataset(file1_path).sizes["d_body"]
        tolerance_file = create_tolerance_csv(n_rows, tol)

        out, err, tol = check_file_with_tolerances(
            tolerance_file, FileInfo(file1_path), FileInfo(file2_path), factor=1, rules=""
        )

        if out:
            print("Files are consistent!")
        else:
            print("Files are NOT consistent!")
            if err:
                file_logger = initialize_detailed_logger(
                "DETAILS",
                log_level="DEBUG",
                log_file=f"error_{ft}.log",
                )
                file_logger.info("Differences, veri_data outside of tolerance range")
                file_logger.info(err)
                file_logger.info(tol)

        #remove_log_if_only_header()
        if os.path.exists(tolerance_file):
            os.remove(tolerance_file)


if __name__ == "__main__":
    fof_compare()  # pylint: disable=no-value-for-parameter
