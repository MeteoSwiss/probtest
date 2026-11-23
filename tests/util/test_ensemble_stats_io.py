"""
Tests for the storage formats of the ensemble statistics of the extreme-values
method, in particular the .npz layout of chilitest.
"""

import json
import logging

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from engine.check import check
from engine.tolerance import tolerance
from tests.helpers import make_mean_stats
from util.ensemble_stats_io import (
    ENSEMBLE_STATISTICS,
    meta_file_name,
    read_ensemble_statistics,
    read_npz_set,
    write_ensemble_statistics,
    write_npz_set,
)
from util.tolerance_methods import (
    build_extreme_values_statistics,
    check_extreme_values,
)

N_TIME = 4
N_MEMBERS = 5
PLEV = [50000.0, 85000.0]
FILE_2D = "NetCDF:run_*/*_2d.nc"
FILE_3D = "NetCDF:run_*/*_3d.nc"
# tas has a vertical dimension of size one (height_2m) and therefore a level
# label, ps has none
ROWS = [
    (FILE_2D, "ps", -1.0),
    (FILE_2D, "tas", 2.0),
    (FILE_3D, "hus", PLEV[0]),
    (FILE_3D, "hus", PLEV[1]),
    (FILE_3D, "ta", PLEV[0]),
    (FILE_3D, "ta", PLEV[1]),
]


def make_stats(mean):
    return make_mean_stats(ROWS, mean)


@pytest.fixture(name="ensemble")
def fixture_ensemble():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(len(ROWS), N_TIME)) * 100.0
    members = ref + rng.normal(size=(N_MEMBERS, len(ROWS), N_TIME)) * 1e-3
    return ref, members


@pytest.fixture(name="df_ens")
def fixture_df_ens(ensemble):
    ref, members = ensemble
    return build_extreme_values_statistics(
        make_stats(ref), [make_stats(m) for m in members]
    )


def write_like_chilitest(stats_dir, ref, members):
    """
    Write reference statistics the way chilitest does
    (EnsembleStatistics.save_ensemble_statistics in ICON's
    utils/chilitest/chilitest/statistics.py): arrays in the shape (time) for
    2d and (time, plev) for 3d, the decoded time axis, and a meta file.
    """
    times = np.datetime64("2020-01-01T00:00") + np.arange(N_TIME) * np.timedelta64(
        2, "m"
    )
    np.savez(
        stats_dir / "aes_amip_stats_2d.npz",
        times=times,
        ps_ref=ref[0],
        ps_min=members[:, 0].min(0),
        ps_max=members[:, 0].max(0),
        tas_ref=ref[1],
        tas_min=members[:, 1].min(0),
        tas_max=members[:, 1].max(0),
    )
    np.savez(
        stats_dir / "aes_amip_stats_3d.npz",
        times=times,
        plev=np.array(PLEV),
        hus_ref=ref[2:4].T,
        hus_min=members[:, 2:4].min(0).T,
        hus_max=members[:, 2:4].max(0).T,
        ta_ref=ref[4:6].T,
        ta_min=members[:, 4:6].min(0).T,
        ta_max=members[:, 4:6].max(0).T,
    )
    with open(stats_dir / "aes_amip_stats_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "variables": {"2d": ["ps", "tas"], "3d": ["hus", "ta"]},
                "spatial_average_type": {
                    "2d": "global_mean",
                    "3d": "surface_mean_per_p_level",
                },
            },
            f,
        )
    return stats_dir / "aes_amip_stats_{dim}.npz"


def test_csv_round_trip(df_ens, tmp_path):
    path = tmp_path / "exp_tolerance.csv"
    write_ensemble_statistics(df_ens, path)
    pd.testing.assert_frame_equal(read_ensemble_statistics(path), df_ens)


def test_npz_round_trip(df_ens, ensemble, tmp_path):
    ref, _ = ensemble
    path = tmp_path / "exp_stats_{dim}.npz"
    write_ensemble_statistics(df_ens, path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "exp_stats_2d.npz",
        "exp_stats_3d.npz",
        "exp_stats_meta.json",
    ]

    df_read = read_ensemble_statistics(path, df_cur=make_stats(ref))
    pd.testing.assert_frame_equal(df_read, df_ens)


def test_npz_layout_is_the_one_of_chilitest(df_ens, ensemble, tmp_path):
    """
    What chilitest's EnsembleStatistics.load expects and what its test uses.
    """
    ref, members = ensemble
    path = tmp_path / "exp_stats_{dim}.npz"
    write_npz_set(df_ens, path)

    with open(meta_file_name(path), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta == {
        "variables": {"2d": ["ps", "tas"], "3d": ["hus", "ta"]},
        "spatial_average_type": {
            "2d": "global_mean",
            "3d": "surface_mean_per_p_level",
        },
    }

    with np.load(tmp_path / "exp_stats_2d.npz") as data:
        assert set(data.files) == {"times"} | {
            f"{v}_{s}" for v in ["ps", "tas"] for s in ENSEMBLE_STATISTICS
        }
        for key in data.files:
            assert data[key].dtype == np.float64
            assert data[key].shape == (N_TIME,)
        np.testing.assert_array_equal(data["tas_ref"], ref[1])
        np.testing.assert_array_equal(data["tas_max"], members[:, 1].max(0))

    with np.load(tmp_path / "exp_stats_3d.npz") as data:
        assert set(data.files) == {"times", "plev"} | {
            f"{v}_{s}" for v in ["hus", "ta"] for s in ENSEMBLE_STATISTICS
        }
        for key in data.files:
            assert data[key].dtype == np.float64
        np.testing.assert_array_equal(data["plev"], PLEV)
        assert np.shape(data["ta_ref"]) == (N_TIME, len(PLEV))
        np.testing.assert_array_equal(data["ta_ref"], ref[4:6].T)
        np.testing.assert_array_equal(data["hus_min"], members[:, 2:4].min(0).T)


def test_npz_times(df_ens, tmp_path):
    path = tmp_path / "exp_stats_{dim}.npz"
    times = np.datetime64("2020-01-01") + np.arange(N_TIME) * np.timedelta64(1, "h")
    write_npz_set(df_ens, path, times={"2d": times})

    with np.load(tmp_path / "exp_stats_2d.npz") as data:
        np.testing.assert_array_equal(data["times"], times)
    with np.load(tmp_path / "exp_stats_3d.npz") as data:
        np.testing.assert_array_equal(data["times"], np.arange(N_TIME))


def test_read_files_written_by_chilitest(df_ens, ensemble, tmp_path):
    ref, members = ensemble
    path = write_like_chilitest(tmp_path, ref, members)

    # the file IDs and the level label of tas are taken from the current stats
    df_read = read_npz_set(path, df_cur=make_stats(ref))
    pd.testing.assert_frame_equal(df_read, df_ens)

    # without them, the dimensions serve as file IDs
    df_plain = read_npz_set(path)
    assert list(df_plain.index) == [
        ("2d", "ps", -1.0),
        ("2d", "tas", -1.0),
        ("3d", "hus", PLEV[0]),
        ("3d", "hus", PLEV[1]),
        ("3d", "ta", PLEV[0]),
        ("3d", "ta", PLEV[1]),
    ]
    np.testing.assert_array_equal(df_plain.to_numpy(), df_ens.to_numpy())

    # explicit assignment
    df_explicit = read_npz_set(path, dimensions={"2d": "a", "3d": "b"})
    assert set(df_explicit.index.get_level_values("file_ID")) == {"a", "b"}


def test_check_against_files_written_by_chilitest(ensemble, tmp_path):
    ref, members = ensemble
    path = write_like_chilitest(tmp_path, ref, members)

    cur = make_stats(ref)
    assert check_extreme_values(read_npz_set(path, df_cur=cur), cur, 1.0).passed

    cur = make_stats(ref + 1.0)
    result = check_extreme_values(read_npz_set(path, df_cur=cur), cur, 1.0)
    assert not result.passed
    assert result.summary["failed"].sum() == ref.size


def test_tolerance_and_check_cli_npz(ensemble, tmp_path, caplog):
    ref, members = ensemble
    make_stats(ref).to_csv(tmp_path / "stats_ref.csv")
    for i, member in enumerate(members, start=1):
        make_stats(member).to_csv(tmp_path / f"stats_{i}.csv")
    stats_dir = tmp_path / "stats_dir"
    tolerance_files = str(stats_dir / "exp_stats_{dim}.npz")

    result = CliRunner().invoke(
        tolerance,
        [
            "--ensemble-files",
            str(tmp_path / "stats_{member_id}.csv"),
            "--tolerance-files",
            tolerance_files,
            "--member-ids",
            "1,2,3,4,5",
            "--method",
            "extreme_values",
        ],
    )
    assert result.exit_code == 0, result.output
    with np.load(stats_dir / "exp_stats_3d.npz") as data:
        np.testing.assert_array_equal(data["ta_max"], members[:, 4:6].max(0).T)

    def run_check(current):
        current_file = tmp_path / "stats_current.csv"
        make_stats(current).to_csv(current_file)
        return CliRunner().invoke(
            check,
            [
                "--current-files",
                str(current_file),
                "--tolerance-files",
                tolerance_files,
                "--factor",
                "1",
                "--method",
                "extreme_values",
            ],
        )

    with caplog.at_level(logging.INFO):
        assert run_check(ref).exit_code == 0
    assert "RESULT: check PASSED" in caplog.text
    caplog.clear()

    current = ref.copy()
    current[5, 3] += 1.0
    with caplog.at_level(logging.INFO):
        assert run_check(current).exit_code == 1
    assert "FAILED: NetCDF:run_*/*_3d.nc ta (1 of 8 values" in caplog.text
    assert "passed: NetCDF:run_*/*_3d.nc hus" in caplog.text


def test_npz_needs_placeholder(df_ens, tmp_path):
    with pytest.raises(SystemExit):
        write_npz_set(df_ens, tmp_path / "exp_stats.npz")


def test_npz_ambiguous_dimensions(df_ens, tmp_path):
    """
    Two file IDs without levels cannot both be stored as 2d.
    """
    other = df_ens.loc[[ROWS[0]]].rename(index={FILE_2D: "NetCDF:other"})
    with pytest.raises(SystemExit):
        write_npz_set(pd.concat([df_ens, other]), tmp_path / "exp_stats_{dim}.npz")


def test_read_npz_missing_statistic(tmp_path):
    np.savez(tmp_path / "exp_stats_2d.npz", ps_ref=np.zeros(3), ps_min=np.zeros(3))
    with pytest.raises(SystemExit):
        read_npz_set(tmp_path / "exp_stats_{dim}.npz")
