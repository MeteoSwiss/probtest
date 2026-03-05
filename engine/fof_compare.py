"""
CLI for checking two fof files

This module provides a command-line interface (CLI) to check that
two given fof files are consistent.
Veri data are not considered, only reports and observations are compared.
"""

import tempfile

import click
import pandas as pd
import xarray as xr

from util.click_util import CommaSeparatedStrings, cli_help
from util.dataframe_ops import check_file_with_tolerances
from util.fof_utils import (
    get_log_file_name,
)
from util.log_handler import initialize_detailed_logger, logger
from util.utils import FileInfo


@click.command()
@click.option(
    "--file1",
    required=True,
    help="Path to the file 1; it must contain the {fof_type} " "placeholder.",
)
@click.option(
    "--file2",
    required=True,
    help="Path to the file 2; it must contain the {fof_type} " "placeholder.",
)
@click.option(
    "--fof-types",
    type=CommaSeparatedStrings(),
    required=True,
    help=cli_help["fof_types"],
)
@click.option(
    "--tolerance",
    default=1e-12,
)
@click.option("--rules", default="")
def fof_compare(file1, file2, fof_types, tolerance, rules):

    for fof_type in fof_types:
        file1_path = file1.format(fof_type=fof_type)
        file2_path = file2.format(fof_type=fof_type)

        n_rows_file1 = xr.open_dataset(file1_path).sizes["d_body"]
        n_rows_file2 = xr.open_dataset(file2_path).sizes["d_body"]

        if n_rows_file1 != n_rows_file2:
            raise ValueError("Files have different numbers of lines!")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=True, dir="/dev/shm"
        ) as tmp:
            df = pd.DataFrame({"tolerance": [tolerance] * n_rows_file1})
            df.to_csv(tmp.name)

            out, err, tol = check_file_with_tolerances(
                tmp.name,
                FileInfo(file1_path),
                FileInfo(file2_path),
                factor=1,
                rules=rules,
            )

            if out:
                logger.info("Files are consistent!")

            else:
                logger.info("Files are NOT consistent!")

                log_file_name = get_log_file_name(file1_path)
                logger.info("Complete output available in %s", log_file_name)
                if not err.empty:
                    detailed_logger = initialize_detailed_logger(
                        "DETAILS", log_level="DEBUG", log_file=log_file_name
                    )

                    detailed_logger.info(
                        "Differences, veri_data outside of tolerance range"
                    )
                    detailed_logger.info(err)
                    detailed_logger.info(tol)


if __name__ == "__main__":
    fof_compare()  # pylint: disable=no-value-for-parameter
