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


@pytest.fixture(name="stats_dataframes", scope="function")
def fixture_dataframes(tmp_dir):
    """
    Create stats dataframes and reference tolerances.
    """

    file_ids = ["NetCDF:*atm_3d*.nc"]
    variables = ["var_1", "var_2"]
    times = range(4)
    stats = ["max", "mean", "min"]

    index = pd.MultiIndex.from_arrays(
        [file_ids * 4, ["var_1"] * 3 + ["var_2"], list(range(3)) + [0]],
        names=["file_ID", "variable", "height"],
    )
    columns = pd.MultiIndex.from_product([times, stats], names=["time", "statistic"])

    array1 = np.linspace(0.9, 1.9, 4 * 12).reshape(4, 12)
    array1[:2] *= -1
    df1 = pd.DataFrame(array1, index=index, columns=columns)

    array2 = np.linspace(1.1, 2.1, 4 * 12).reshape(4, 12)
    array2[:2] *= -1
    df2 = pd.DataFrame(array2, index=index, columns=columns)

    base_index = pd.Index(variables, name="variable")
    tol_large_base = pd.DataFrame(
        np.ones((2, 12)) * 0.21, index=base_index, columns=columns
    )
    tol_small_base = pd.DataFrame(
        np.ones((2, 12)) * 0.06, index=base_index, columns=columns
    )

    tol_large = pd.concat(
        [tol_large_base] * len(file_ids), keys=file_ids, names=["file_ID", "variable"]
    )
    tol_small = pd.concat(
        [tol_small_base] * len(file_ids), keys=file_ids, names=["file_ID", "variable"]
    )

    df1_stats = os.path.join(tmp_dir, "stats1.csv")
    df2_stats = os.path.join(tmp_dir, "stats2.csv")
    tol_large_stats = os.path.join(tmp_dir, "tol_large.csv")
    tol_small_stats = os.path.join(tmp_dir, "tol_small.csv")

    df1.to_csv(df1_stats)
    df2.to_csv(df2_stats)
    tol_large.to_csv(tol_large_stats)
    tol_small.to_csv(tol_small_stats)

    yield df1_stats, df2_stats, tol_large_stats, tol_small_stats


@pytest.fixture(name="fof_datasets", scope="function")
def fixture_fof_datasets(tmp_dir, sample_dataset_fof):
    """
    Create fof dataset and reference tolerances.
    """

    ds1 = sample_dataset_fof
    ds2 = ds1.copy(deep=True)
    ds2["veri_data"] = (("d_body",), ds2["veri_data"].values * 1.55)

    n_body_size = ds1.sizes["d_body"]

    tol_large = pd.DataFrame({"veri_data": np.full(n_body_size, 5)})
    tol_small = pd.DataFrame({"veri_data": np.full(n_body_size, 0.06)})

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
        ],
    )

    assert result.exit_code == 0
