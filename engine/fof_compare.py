"""
CLI for checking two fof files

This module provides a command-line interface (CLI) to check that
two given fof files are consistent.
Veri data are not considered, only reports and observations are compared.
"""

import click
import xarray as xr

from util.fof_utils import (
    compare_var_and_attr_ds,
    primary_check,
    split_feedback_dataset,
)


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

    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

    ds_reports1_sorted, ds_obs1_sorted = split_feedback_dataset(ds1)
    ds_reports2_sorted, ds_obs2_sorted = split_feedback_dataset(ds2)

    total_elements_all, equal_elements_all = 0, 0

    if print_lines:
        nl = lines
    else:
        nl = 0

    for ds1, ds2 in [
        (ds_reports1_sorted, ds_reports2_sorted),
        (ds_obs1_sorted, ds_obs2_sorted),
    ]:
        t, e = compare_var_and_attr_ds(ds1, ds2, nl, output, location, tol)
        total_elements_all += t
        equal_elements_all += e

    if total_elements_all > 0:
        percent_equal_all = (equal_elements_all / total_elements_all) * 100
        percent_diff_all = 100 - percent_equal_all
        print(f"Total percentage of equality: {percent_equal_all:.2f}%")
        print(f"Total percentage of difference: {percent_diff_all:.2f}%")
        if equal_elements_all == total_elements_all:
            print("Files are consistent!")
        else:
            print("Files are NOT consistent!")


if __name__ == "__main__":
    fof_compare()  # pylint: disable=no-value-for-parameter
