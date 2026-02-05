"""
This module contains test cases to validate the functionality
of check CLI commands.
"""

import os

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from visualize.check_plot import check_plot



# @pytest.fixture(name="stats_dataframes", scope="function")
# def fixture_dataframes(tmp_dir):
#     """
#     Create stats dataframes and reference tolerances.
#     """

#     file_ids = ["NetCDF:*atm_3d*.nc"]
#     variables = ["var_1", "var_2"]
#     times = range(4)
#     stats = ["max", "mean", "min"]

#     index = pd.MultiIndex.from_arrays(
#         [file_ids * 4, ["var_1"] * 3 + ["var_2"], list(range(3)) + [0]],
#         names=["file_ID", "variable", "height"],
#     )
#     columns = pd.MultiIndex.from_product([times, stats], names=["time", "statistic"])

#     array1 = np.linspace(0.9, 1.9, 4 * 12).reshape(4, 12)
#     array1[:2] *= -1
#     df1 = pd.DataFrame(array1, index=index, columns=columns)

#     array2 = np.linspace(1.1, 2.1, 4 * 12).reshape(4, 12)
#     array2[:2] *= -1
#     df2 = pd.DataFrame(array2, index=index, columns=columns)

#     base_index = pd.Index(variables, name="variable")
#     tol_large_base = pd.DataFrame(
#         np.ones((2, 12)) * 0.21, index=base_index, columns=columns
#     )
#     tol_small_base = pd.DataFrame(
#         np.ones((2, 12)) * 0.06, index=base_index, columns=columns
#     )

#     tol_small = pd.concat(
#         [tol_small_base] * len(file_ids), keys=file_ids, names=["file_ID", "variable"]
#     )

#     df1_stats = os.path.join(tmp_dir, "stats1.csv")
#     df2_stats = os.path.join(tmp_dir, "stats2.csv")
#     tol_small_stats = os.path.join(tmp_dir, "tol_small.csv")

#     df1.to_csv(df1_stats)
#     df2.to_csv(df2_stats)
#     tol_small.to_csv(tol_small_stats)

#     yield df1_stats, df2_stats, tol_small_stats


def test_check_cli_stats(stats_dataframes, tmp_path):
    """
    Test that is not validated in the case of large tolerances but
    is validated in the case of small tolerances.
    """

    df1_stats, df2_stats,_, tol_small = stats_dataframes

    runner = CliRunner()
    result = runner.invoke(
        check_plot,
        [
            "--tolerance-files",
            tol_small,
            "--reference-files",
            df1_stats,
            "--current-files",
            df2_stats,
            "--factor",
            "1.0",
            "--savedir",
            tmp_path,
        ],
    )
    output = tmp_path / "check_plot.png"

    assert output.exists()
    assert result.exit_code == 0, result.output
