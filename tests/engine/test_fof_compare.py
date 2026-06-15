"""
This module contains test cases to validate the functionality
of fof-compare CLI commands.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from engine.fof_compare import fof_compare


@pytest.fixture(name="fof_datasets", scope="function")
def fixture_fof_datasets(fof_datasets_base, tmp_dir):
    """
    FOF datasets written to disk, returns file paths.
    """
    ds1, ds2, _, _ = fof_datasets_base
    ds3 = ds2.copy(deep=True)
    ds3["flags"] = (("d_body",), ds3["flags"].values * 2)

    ds1_file = os.path.join(tmp_dir, "fof1_SYNOP.nc")
    ds2_file = os.path.join(tmp_dir, "fof2_SYNOP.nc")
    ds3_file = os.path.join(tmp_dir, "fof3_SYNOP.nc")

    ds1.to_netcdf(ds1_file)
    ds2.to_netcdf(ds2_file)
    ds3.to_netcdf(ds3_file)

    yield ds1_file, ds2_file, ds3_file


def test_fof_compare_works(fof_datasets, tmp_dir, monkeypatch):
    """
    Test that fof-compare works and produces a log file.
    """

    df1, df2, _ = fof_datasets

    df1 = df1.replace("SYNOP", "{fof_type}")
    df2 = df2.replace("SYNOP", "{fof_type}")
    monkeypatch.chdir(tmp_dir)
    rules = "{}"
    runner = CliRunner()

    result = runner.invoke(
        fof_compare,
        [
            "--file1",
            df1,
            "--file2",
            df2,
            "--fof-types",
            "SYNOP",
            "--tolerance",
            "1e-12",
            "--rules",
            rules,
        ],
    )

    assert result.exit_code == 0

    log_file = Path(tmp_dir + "/error_fof1_SYNOP.log")

    assert (log_file).exists()


def test_fof_compare_not_consistent(fof_datasets, tmp_dir, monkeypatch, caplog):
    """
    Test that if there are differences in the files, then fof-compare writes
    in the log file that the files are not consistent.
    """

    df1, _, df3 = fof_datasets
    df1 = df1.replace("SYNOP", "{fof_type}")
    df3 = df3.replace("SYNOP", "{fof_type}")
    monkeypatch.chdir(tmp_dir)

    rules = "{}"
    runner = CliRunner()
    with caplog.at_level(logging.INFO):
        runner.invoke(
            fof_compare,
            [
                "--file1",
                df1,
                "--file2",
                df3,
                "--fof-types",
                "SYNOP",
                "--tolerance",
                "5",
                "--rules",
                rules,
            ],
        )

    assert "Files are NOT consistent!" in caplog.text


def test_fof_compare_rules(fof_datasets, tmp_dir, monkeypatch, caplog):
    """
    Test that if there are differences in the files, then fof-compare writes
    in the log file that the files are not consistent.
    """

    df1, _, df3 = fof_datasets
    df1 = df1.replace("SYNOP", "{fof_type}")
    df3 = df3.replace("SYNOP", "{fof_type}")
    monkeypatch.chdir(tmp_dir)

    rules = '{"flags": [9, 18]}'
    runner = CliRunner()
    with caplog.at_level(logging.INFO):
        runner.invoke(
            fof_compare,
            [
                "--file1",
                df1,
                "--file2",
                df3,
                "--fof-types",
                "SYNOP",
                "--tolerance",
                "5",
                "--rules",
                rules,
            ],
        )

    assert "Files are consistent!" in caplog.text


def test_fof_compare_consistent(fof_datasets, tmp_dir, monkeypatch, caplog):
    """
    Test that if there are no differences in the files and the tolerance is big
    enough, then fof-compare writes in the log file that the files are consistent.
    """

    df1, df2, _ = fof_datasets
    df1 = df1.replace("SYNOP", "{fof_type}")
    df2 = df2.replace("SYNOP", "{fof_type}")
    monkeypatch.chdir(tmp_dir)

    rules = "{}"
    runner = CliRunner()
    with caplog.at_level(logging.INFO):
        runner.invoke(
            fof_compare,
            [
                "--file1",
                df1,
                "--file2",
                df2,
                "--fof-types",
                "SYNOP",
                "--tolerance",
                "5",
                "--rules",
                rules,
            ],
        )

    assert "Files are consistent!" in caplog.text


@pytest.fixture(name="radar_fof_paths", scope="function")
def fixture_radar_fof_paths(sample_dataset_radar_fof, tmp_dir):
    """
    Write padded radar fof files to disk and return paths containing the
    {fof_type} placeholder (resolved to 'MLL'). Variants:
    - ref/same: identical padded radar files.
    - pert: one real-region veri_data value pushed far beyond any tolerance.
    - bigpad: same n_body as ref but a larger d_body (more NaN padding rows) ->
      exercises the n_body-vs-d_body fix (size gate + tolerance length).
    """
    ds = sample_dataset_radar_fof

    same = ds.copy(deep=True)

    pert = ds.copy(deep=True)
    vd = pert["veri_data"].values.copy()
    vd[0, 0] = vd[0, 0] + 100.0
    pert["veri_data"] = (("d_veri", "d_body"), vd)

    bigpad = ds.pad(d_body=(0, 4), constant_values=np.nan)
    bigpad.attrs = dict(ds.attrs)  # keep the real n_hdr/n_body counts

    # nanfill: a real-region veri_data NaN (an observation missing in ref) becomes
    # a real value -> NaN-vs-real mismatch that must be flagged, not silently passed.
    nanfill = ds.copy(deep=True)
    vdn = nanfill["veri_data"].values.copy()
    vdn[0, 2] = 3.0  # index 2 is NaN in the fixture's real region
    nanfill["veri_data"] = (("d_veri", "d_body"), vdn)

    pattern = os.path.join(tmp_dir, "fofRADAR{fof_type}_%s.nc")
    paths = {}
    for name, dataset in (
        ("ref", ds),
        ("same", same),
        ("pert", pert),
        ("bigpad", bigpad),
        ("nanfill", nanfill),
    ):
        placeholder_path = pattern % name
        dataset.to_netcdf(placeholder_path.format(fof_type="MLL"))
        paths[name] = placeholder_path
    return paths


def _run_fof_compare(file1, file2, tolerance, tmp_dir, monkeypatch, caplog):
    monkeypatch.chdir(tmp_dir)
    runner = CliRunner()
    with caplog.at_level(logging.INFO):
        runner.invoke(
            fof_compare,
            [
                "--file1", file1,
                "--file2", file2,
                "--fof-types", "MLL",
                "--tolerance", str(tolerance),
                "--rules", "",
            ],
        )


def test_fof_compare_radar_consistent(radar_fof_paths, tmp_dir, monkeypatch, caplog):
    """Two identical padded radar fof files compare as consistent."""
    _run_fof_compare(
        radar_fof_paths["ref"], radar_fof_paths["same"], 1.0, tmp_dir, monkeypatch, caplog
    )
    assert "Files are consistent!" in caplog.text


def test_fof_compare_radar_not_consistent(radar_fof_paths, tmp_dir, monkeypatch, caplog):
    """A single real-region veri_data value out of tolerance is detected."""
    _run_fof_compare(
        radar_fof_paths["ref"], radar_fof_paths["pert"], 1.0, tmp_dir, monkeypatch, caplog
    )
    assert "Files are NOT consistent!" in caplog.text


def test_fof_compare_radar_different_padding(radar_fof_paths, tmp_dir, monkeypatch, caplog):
    """
    Two radar files with the same n_body but different d_body (padding) must still
    compare -- the size gate and tolerance length use n_body, not d_body.
    """
    _run_fof_compare(
        radar_fof_paths["ref"], radar_fof_paths["bigpad"], 1.0, tmp_dir, monkeypatch, caplog
    )
    assert "Files are consistent!" in caplog.text


def test_fof_compare_radar_nan_vs_real(radar_fof_paths, tmp_dir, monkeypatch, caplog):
    """
    A veri_data cell that is NaN in one file but a real value in the other (an
    observation appearing/disappearing) must be flagged, not silently passed --
    even with a generous tolerance.
    """
    _run_fof_compare(
        radar_fof_paths["ref"], radar_fof_paths["nanfill"], 1.0, tmp_dir, monkeypatch, caplog
    )
    assert "Files are NOT consistent!" in caplog.text
