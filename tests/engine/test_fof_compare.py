"""
This module contains test cases to validate the functionality
of fof-compare CLI commands.
"""

import logging
import os
from pathlib import Path

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
    ds3["flags"] = (("d_body",), ds3["flags"].values * 1.55)

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
    rules = ""
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

    rules = ""
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


def test_fof_compare_consistent(fof_datasets, tmp_dir, monkeypatch, caplog):
    """
    Test that if there are no differences in the files and the tolerance is big
    enough, then fof-compare writes in the log file that the files are consistent.
    """

    df1, df2, _ = fof_datasets
    df1 = df1.replace("SYNOP", "{fof_type}")
    df2 = df2.replace("SYNOP", "{fof_type}")
    monkeypatch.chdir(tmp_dir)

    rules = ""
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
