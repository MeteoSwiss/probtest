"""
This module contains test cases to validate the functionality
of check-plot CLI commands.
"""

from click.testing import CliRunner

from visualize.check_plot import check_plot


def test_check_cli_stats(stats_dataframes, tmp_path):
    """
    Test that is not validated in the case of large tolerances but
    is validated in the case of small tolerances.
    """

    df1_stats, df2_stats, _, tol_small = stats_dataframes

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
