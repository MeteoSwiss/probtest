"""
CLI for checking two fof files

This module provides a command-line interface (CLI) to check that
two given fof files are consistent.
Veri data are not considered, only reports and observations are compared.
"""

import click
import xarray as xr
import pandas as pd

from util.fof_utils import (
    compare_var_and_attr_ds,
    primary_check,
    split_feedback_dataset,
)
from util.dataframe_ops import check_file_with_tolerances, parse_check
from util.utils import FileInfo


@click.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option(
    "--print-lines",
    is_flag=True,
    help="Prints the lines where there are differences. "
    "If --lines is not specified, then the first 10 "
    "differences per variables are shown.",
)
@click.option(
    "--lines",
    "-n",
    default=10,
    help="Option to specify how many lines to print " "with the --print-lines option",
)
@click.option(
    "--output",
    "-o",
    is_flag=True,
    help="Option to save differences in a CSV file. "
    "If the location is not specified, the file "
    "is saved in the same location as this code. ",
)
@click.option(
    "--location",
    "-l",
    default=None,
    help="If specified, location where to save the CSV file with the differences.",
)
@click.option(
    "--tol",
    default=1e-12,
)
def fof_compare(
    file1, file2, print_lines, lines, output, location, tol
):  # pylint: disable=too-many-positional-arguments

    if not primary_check(file1, file2):
        print("Different types of files")
        return

    n_righe = xr.open_dataset(file1).sizes["d_body"]
    tolerance_file = "tolerance_file.csv"

    def create_tolerance_csv(n_righe, tol, tolerance_file_name):
        df = pd.DataFrame(
            {"tolerance": [tol] * n_righe}
        )
        df.to_csv(tolerance_file_name)

    create_tolerance_csv(n_righe, tol, tolerance_file)

    out, err, tol = check_file_with_tolerances(
            tolerance_file,
            FileInfo(file1),
            FileInfo(file2),
            factor=4,
            rules="",
        )
    # print(out)
    # print(err)




if __name__ == "__main__":
    fof_compare()  # pylint: disable=no-value-for-parameter
