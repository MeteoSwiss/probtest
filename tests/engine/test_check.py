"""
This module contains test cases to validate the functionality
of check CLI commands.
"""

import os

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from engine.check import check


@pytest.fixture(name="fof_datasets", scope="function")
def fixture_fof_datasets(tmp_dir, fof_datasets_base):
    """
    FoF datasets written to disk, returns file paths.
    """
    ds1, ds2, tol_large, tol_small = fof_datasets_base

    ds1_file = os.path.join(tmp_dir, "fof1.nc")
    ds2_file = os.path.join(tmp_dir, "fof2.nc")
    tol_large_file = os.path.join(tmp_dir, "fof_tol_large.csv")
    tol_small_file = os.path.join(tmp_dir, "fof_tol_small.csv")

    ds1.to_netcdf(ds1_file)
    ds2.to_netcdf(ds2_file)
    tol_large.to_csv(tol_large_file)
    tol_small.to_csv(tol_small_file)

    yield ds1_file, ds2_file, tol_large_file, tol_small_file


def test_check_cli_stats(stats_dataframes):
    """
    Test that is not validated in the case of large tolerances but
    is validated in the case of small tolerances.
    """

    df1_stats, df2_stats, tol_large, tol_small = stats_dataframes

    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--reference-files",
            df1_stats,
            "--current-files",
            df2_stats,
            "--tolerance-files",
            tol_small,
            "--factor",
            "1.0",
            "--fof-types",
            "",
        ],
    )

    assert result.exit_code == 1

    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--reference-files",
            df1_stats,
            "--current-files",
            df2_stats,
            "--tolerance-files",
            tol_large,
            "--factor",
            "1.0",
            "--fof-types",
            "",
        ],
    )

    assert result.exit_code == 0


def test_check_cli_fof(fof_datasets):
    """
    Same as before but for fof files.
    """

    df1, df2, tol_large, tol_small = fof_datasets
    rules = {"check": [13, 18, 32], "state": [1, 5, 7, 9]}

    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--reference-files",
            df1,
            "--current-files",
            df2,
            "--tolerance-files",
            tol_small,
            "--factor",
            "1.0",
            "--fof-types",
            "",
            "--rules",
            rules,
        ],
    )

    assert result.exit_code == 1

    runner = CliRunner()
    result = runner.invoke(
        check,
        [
            "--reference-files",
            df1,
            "--current-files",
            df2,
            "--tolerance-files",
            tol_large,
            "--factor",
            "1.0",
            "--fof-types",
            "",
            "--rules",
            rules,
        ],
    )

    assert result.exit_code == 0
