"""
This module contains test cases to validate the functionality
of the ensemble-plot CLI command.
"""

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from tests.helpers import make_mean_stats
from visualize.ensemble_plot import (
    ensemble_plot,
    mean_differences,
    plot_differences,
    subplot_titles,
)

N_TIME = 6
N_MEMBERS = 4
ROWS = [
    ("NetCDF:*_2d.nc", "ps", -1.0),
    ("NetCDF:*_2d.nc", "tas", 2.0),
    ("NetCDF:*_3d.nc", "ta", 50000.0),
    ("NetCDF:*_3d.nc", "ta", 85000.0),
]


@pytest.fixture(name="ensemble")
def fixture_ensemble(tmp_path):
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(len(ROWS), N_TIME)) * 100.0
    members = ref + rng.normal(size=(N_MEMBERS, len(ROWS), N_TIME)) * 1e-3

    make_mean_stats(ROWS, ref).to_csv(tmp_path / "stats_ref.csv")
    for i, member in enumerate(members, start=1):
        make_mean_stats(ROWS, member).to_csv(tmp_path / f"stats_{i}.csv")
    current = ref + 5e-3
    make_mean_stats(ROWS, current).to_csv(tmp_path / "stats_current.csv")

    return tmp_path, ref, members, current


def run(tmp_path, *extra, savedir=None):
    return CliRunner().invoke(
        ensemble_plot,
        [
            "--ensemble-files",
            str(tmp_path / "stats_{member_id}.csv"),
            "--member-ids",
            "1,2,3,4",
            "--savedir",
            str(savedir or tmp_path / "plots"),
            *extra,
        ],
    )


def test_mean_differences(ensemble):
    _, ref, members, _ = ensemble
    diff = mean_differences(
        make_mean_stats(ROWS, members[0]), make_mean_stats(ROWS, ref)
    )
    assert list(diff.index) == ROWS
    assert list(diff.columns) == list(range(N_TIME))
    np.testing.assert_array_equal(diff.to_numpy(), members[0] - ref)


def test_subplot_titles():
    index = pd.MultiIndex.from_tuples(ROWS)
    assert subplot_titles(index) == [
        "NetCDF:*_2d.nc: ps",
        "NetCDF:*_2d.nc: tas",
        "NetCDF:*_3d.nc: ta (level 50000)",
        "NetCDF:*_3d.nc: ta (level 85000)",
    ]


def test_plot_differences(ensemble):
    _, ref, members, current = ensemble
    df_ref = make_mean_stats(ROWS, ref)
    member_diffs = [mean_differences(make_mean_stats(ROWS, m), df_ref) for m in members]
    current_diff = mean_differences(make_mean_stats(ROWS, current), df_ref)

    fig = plot_differences(member_diffs, current_diff)

    axes = [axis for axis in fig.axes if axis.get_visible()]
    assert len(axes) == len(ROWS)
    for i, axis in enumerate(axes):
        # the members, the current run and the zero line
        assert len(axis.lines) == N_MEMBERS + 2
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), members[0][i] - ref[i])
        np.testing.assert_array_equal(
            axis.lines[N_MEMBERS].get_ydata(), current[i] - ref[i]
        )
        # symmetric around zero and containing everything
        low, high = axis.get_ylim()
        assert low == -high
        assert high > np.abs(current[i] - ref[i]).max()

    fig = plot_differences(member_diffs)
    assert len(fig.axes[0].lines) == N_MEMBERS + 1


def test_ensemble_plot_cli(ensemble):
    tmp_path = ensemble[0]
    result = run(tmp_path, "--current-files", str(tmp_path / "stats_current.csv"))
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert (tmp_path / "plots" / "ensemble_plot.png").stat().st_size > 0


def test_ensemble_plot_cli_without_current_and_with_variables(ensemble):
    tmp_path = ensemble[0]
    result = run(tmp_path, "--variables", "ta")
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert (tmp_path / "plots" / "ensemble_plot.png").exists()


def test_ensemble_plot_cli_unknown_variable(ensemble):
    tmp_path = ensemble[0]
    result = run(tmp_path, "--variables", "not_there")
    assert result.exit_code == 1
    assert not (tmp_path / "plots").exists()
