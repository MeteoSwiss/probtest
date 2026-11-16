"""
Tests for the tolerance methods, in particular the extreme-values (min/max)
method, on the level of the functions and of the tolerance and check CLIs.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from engine.check import check
from engine.tolerance import tolerance
from util.tolerance_methods import (
    ENSEMBLE_STATISTICS,
    build_extreme_values_statistics,
    check_extreme_values,
    parse_ensemble_statistics,
)

N_TIME = 4
N_MEMBERS = 5
ROWS = [
    ("NetCDF:*_2d.nc", "ps", -1.0),
    ("NetCDF:*_2d.nc", "tas", -1.0),
    ("NetCDF:*_3d.nc", "ta", 50000.0),
    ("NetCDF:*_3d.nc", "ta", 85000.0),
]


def make_stats(mean):
    """
    Stats DataFrame in the layout of the stats files from an array of
    horizontal means with the shape (rows, time).
    """
    index = pd.MultiIndex.from_tuples(ROWS, names=["file_ID", "variable", "height"])
    columns = pd.MultiIndex.from_product(
        [range(mean.shape[1]), ["mean", "max", "min"]], names=["time", "statistic"]
    )
    df = pd.DataFrame(index=index, columns=columns, dtype=np.float64)
    for t in range(mean.shape[1]):
        df[(t, "mean")] = mean[:, t]
        # max and min of the field must not influence the method
        df[(t, "max")] = mean[:, t] + 1000.0
        df[(t, "min")] = mean[:, t] - 1000.0
    return df


def reference_extreme_values(x, ref, ens_min, ens_max, factor):
    """
    Transcription of the extreme-values test of chilitest
    (utils/chilitest/chilitest/test.py in ICON).
    """
    eps_ref = np.finfo(np.float64).eps * np.abs(ref)
    diff = x - ref
    tmax = factor * np.maximum(ens_max - ref, eps_ref)
    tmin = factor * np.minimum(ens_min - ref, -eps_ref)
    return np.where(diff >= 0, diff <= tmax, np.abs(diff) <= np.abs(tmin))


@pytest.fixture(name="ensemble")
def fixture_ensemble():
    rng = np.random.default_rng(42)
    ref = rng.normal(size=(len(ROWS), N_TIME)) * 100.0
    members = ref + rng.normal(size=(N_MEMBERS, len(ROWS), N_TIME)) * 1e-3

    # no spread at all for one value
    members[:, 0, 0] = ref[0, 0]
    # no spread around an exact zero
    ref[1, 1] = 0.0
    members[:, 1, 1] = 0.0
    # all members on one side of the reference
    members[:, 2, 2] = ref[2, 2] + np.abs(members[:, 2, 2] - ref[2, 2]) + 1e-6

    return ref, members


def test_build_extreme_values_statistics(ensemble):
    ref, members = ensemble
    df_ens = build_extreme_values_statistics(
        make_stats(ref), [make_stats(m) for m in members]
    )

    assert list(df_ens.index.names) == ["file_ID", "variable", "height"]
    assert list(df_ens.columns.names) == ["time", "statistic"]
    assert list(df_ens.columns) == [
        (t, s) for t in range(N_TIME) for s in ENSEMBLE_STATISTICS
    ]

    np.testing.assert_array_equal(df_ens.xs("ref", axis=1, level=1), ref)
    np.testing.assert_array_equal(df_ens.xs("min", axis=1, level=1), members.min(0))
    np.testing.assert_array_equal(df_ens.xs("max", axis=1, level=1), members.max(0))


@pytest.mark.parametrize("factor", [1.0, 5.0])
@pytest.mark.parametrize("amplitude", [0.0, 1e-17, 5e-4, 2e-3, 1e-2])
def test_check_extreme_values_matches_chilitest(ensemble, factor, amplitude):
    ref, members = ensemble
    df_ens = build_extreme_values_statistics(
        make_stats(ref), [make_stats(m) for m in members]
    )

    rng = np.random.default_rng(7)
    cur = ref + rng.normal(size=ref.shape) * amplitude

    result = check_extreme_values(df_ens, make_stats(cur), factor)

    expected = reference_extreme_values(
        cur, ref, members.min(0), members.max(0), factor
    )
    assert result.passed == bool(expected.all())
    assert result.summary["failed"].sum() == (~expected).sum()
    assert len(result.failures) == (~expected).sum()
    assert result.summary["total"].sum() == expected.size

    # failures per variable; ta has two levels
    failed_per_variable = [
        (~expected[0]).sum(),
        (~expected[1]).sum(),
        (~expected[2:]).sum(),
    ]
    assert list(result.summary["failed"]) == failed_per_variable


def test_check_extreme_values_identical_run_passes(ensemble):
    ref, members = ensemble
    df_ens = build_extreme_values_statistics(
        make_stats(ref), [make_stats(m) for m in members]
    )
    result = check_extreme_values(df_ens, make_stats(ref), 1.0)
    assert result.passed
    assert result.failures.empty


def test_check_extreme_values_one_sided(ensemble):
    """
    All members above the reference: a deviation below it is only allowed up
    to the machine epsilon.
    """
    ref, members = ensemble
    df_ens = build_extreme_values_statistics(
        make_stats(ref), [make_stats(m) for m in members]
    )
    cur = ref.copy()
    cur[2, 2] = ref[2, 2] - 1e-6

    result = check_extreme_values(df_ens, make_stats(cur), 1.0)

    assert not result.passed
    assert len(result.failures) == 1
    assert result.failures.index[0] == ("NetCDF:*_3d.nc", "ta", 50000.0, 2)
    failure = result.failures.iloc[0]
    assert failure["observed"] == cur[2, 2]
    assert failure["reference"] == ref[2, 2]
    assert failure["lower"] == -np.finfo(np.float64).eps * abs(ref[2, 2])
    assert failure["upper"] > 0


def test_check_extreme_values_nan_fails(ensemble):
    ref, members = ensemble
    df_ens = build_extreme_values_statistics(
        make_stats(ref), [make_stats(m) for m in members]
    )
    cur = ref.copy()
    cur[3, 1] = np.nan

    result = check_extreme_values(df_ens, make_stats(cur), 1.0)

    assert not result.passed
    assert list(result.summary["failed"]) == [0, 0, 1]


def test_build_extreme_values_statistics_nan_member_fails(ensemble):
    ref, members = ensemble
    members = members.copy()
    members[0, 1, 0] = np.nan
    df_ens = build_extreme_values_statistics(
        make_stats(ref), [make_stats(m) for m in members]
    )
    assert np.isnan(df_ens.loc[ROWS[1], (0, "max")])

    result = check_extreme_values(df_ens, make_stats(ref), 1.0)
    assert not result.passed
    assert list(result.summary["failed"]) == [0, 1, 0]


@pytest.fixture(name="stats_files")
def fixture_stats_files(ensemble, tmp_path):
    ref, members = ensemble
    make_stats(ref).to_csv(tmp_path / "stats_ref.csv")
    for i, member in enumerate(members, start=1):
        make_stats(member).to_csv(tmp_path / f"stats_{i}.csv")
    return tmp_path


def run_tolerance(stats_dir, tolerance_file):
    return CliRunner().invoke(
        tolerance,
        [
            "--ensemble-files",
            str(stats_dir / "stats_{member_id}.csv"),
            "--tolerance-files",
            str(tolerance_file),
            "--member-ids",
            ",".join(str(i) for i in range(1, N_MEMBERS + 1)),
            "--method",
            "extreme_values",
        ],
    )


def test_tolerance_and_check_cli(ensemble, stats_files, caplog):
    ref, members = ensemble
    tolerance_file = stats_files / "exp_tolerance.csv"

    result = run_tolerance(stats_files, tolerance_file)
    assert result.exit_code == 0, result.output

    df_ens = parse_ensemble_statistics(tolerance_file)
    np.testing.assert_array_equal(df_ens.xs("max", axis=1, level=1), members.max(0))

    # the reference passes without --reference-files
    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            check,
            [
                "--current-files",
                str(stats_files / "stats_ref.csv"),
                "--tolerance-files",
                str(tolerance_file),
                "--factor",
                "1",
                "--method",
                "extreme_values",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "RESULT: check PASSED" in caplog.text
    assert "passed: NetCDF:*_2d.nc ps" in caplog.text
    caplog.clear()

    # a run far outside the ensemble fails, and every variable is reported
    current_file = stats_files / "stats_current.csv"
    make_stats(ref + 1.0).to_csv(current_file)
    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(
            check,
            [
                "--current-files",
                str(current_file),
                "--tolerance-files",
                str(tolerance_file),
                "--factor",
                "1",
                "--method",
                "extreme_values",
                "--verbose",
            ],
        )
    assert result.exit_code == 1
    assert "RESULT: check FAILED" in caplog.text
    for variable, total in [("ps", 4), ("tas", 4), ("ta", 8)]:
        assert f"{variable} ({total} of {total} values" in caplog.text
    assert "lower" in caplog.text and "upper" in caplog.text


def test_check_cli_rejects_relative_tolerance_file(stats_files):
    """
    A tolerance file of the relative method is not silently accepted.
    """
    tolerance_file = stats_files / "exp_tolerance.csv"
    result = CliRunner().invoke(
        tolerance,
        [
            "--ensemble-files",
            str(stats_files / "stats_{member_id}.csv"),
            "--tolerance-files",
            str(tolerance_file),
            "--member-ids",
            "1,2,3,4,5",
        ],
    )
    assert result.exit_code == 0, result.output

    result = CliRunner().invoke(
        check,
        [
            "--current-files",
            str(stats_files / "stats_ref.csv"),
            "--tolerance-files",
            str(tolerance_file),
            "--factor",
            "1",
            "--method",
            "extreme_values",
        ],
    )
    assert result.exit_code != 0
