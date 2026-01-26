"""
CLI for checking two fof files

This module provides a command-line interface (CLI) to check that
two given fof files are consistent.
Veri data are not considered, only reports and observations are compared.
"""

import os

import click
import xarray as xr

from util.dataframe_ops import check_file_with_tolerances
from util.fof_utils import create_tolerance_csv
from util.utils import FileInfo


@click.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option(
    "--tol",
    default=1e-12,
)
def fof_compare(file1, file2, tol):

    n_rows = xr.open_dataset(file1).sizes["d_body"]
    tolerance_file = "tolerance_file.csv"

    create_tolerance_csv(n_rows, tol, tolerance_file)

    out, _, _ = check_file_with_tolerances(
        tolerance_file, FileInfo(file1), FileInfo(file2), factor=1, rules=""
    )
    if out:
        print("Files are consistent!")
    else:
        print("Files are NOT consistent!")

    if os.path.exists(tolerance_file):
        os.remove(tolerance_file)


if __name__ == "__main__":
    fof_compare()  # pylint: disable=no-value-for-parameter
