"""
This module contains test cases to validate the functionality of member
selection and tolerance testing CLI commands.
"""

import logging
import os
import re

import pytest

from tests.helpers import run_select_members_cli, run_tolerance_cli


def test_multiple_ensemble_files_error(stats_file_set, caplog):
    caplog.set_level(logging.ERROR)

    with pytest.raises(Exception):
        run_select_members_cli(
            stats_file_set["stats"] + ",extra_stats",
            stats_file_set["members"],
            tolerance_files=stats_file_set["tol"],
        )

    # Check that the log contains the error
    assert any("ERROR" in record.message for record in caplog.records)


def test_fof_ensemble_file_error(stats_file_set, caplog):
    caplog.set_level(logging.ERROR)

    with pytest.raises(Exception):
        run_select_members_cli(
            "fofDumy.nc",
            stats_file_set["members"],
            tolerance_files=stats_file_set["tol"],
        )

    # Check that the log contains the error
    assert any("ERROR" in record.message for record in caplog.records)


def test_multiple_tolerance_files_error(stats_file_set, caplog):
    caplog.set_level(logging.ERROR)

    with pytest.raises(Exception):
        run_select_members_cli(
            stats_file_set["stats"],
            stats_file_set["members"],
            tolerance_files=stats_file_set["tol"] + ",extra_stats",
        )

    # Check that the log contains the error
    assert any("ERROR" in record.message for record in caplog.records)


def test_fof_tolerance_file_error(stats_file_set, caplog):
    caplog.set_level(logging.ERROR)

    with pytest.raises(Exception):
        run_select_members_cli(
            stats_file_set["stats"],
            stats_file_set["members"],
            tolerance_files="fofDymmy.nc",
        )

    # Check that the log contains the error
    assert any("ERROR" in record.message for record in caplog.records)

def test_invalid_max_member_cout_error(stats_file_set, caplog):
    caplog.set_level(logging.ERROR)

    with pytest.raises(Exception):
        run_select_members_cli(
            stats_file_set["stats"],
            stats_file_set["members"],
            tolerance_files=stats_file_set["tol"],
            max_member_count=20,
            total_member_count=15,
        )

    # Check that the log contains the error
    assert any("ERROR" in record.message for record in caplog.records)

def test_select_members(stats_file_set):

    run_select_members_cli(
        stats_file_set["stats"],
        stats_file_set["members"],
        tolerance_files=stats_file_set["tol"],
    )

    assert os.path.isfile(
        stats_file_set["members"]
    ), "File 'selected_members.csv' was not created"

    with open(stats_file_set["members"], "r", encoding="utf-8") as file:
        content = file.read().strip()
    expected_content = "4,6,15\nexport FACTOR=5"
    assert content == expected_content, "The member selection failed"


def test_select_members_increase_factor(stats_file_set):
    run_select_members_cli(
        stats_file_set["stats"],
        stats_file_set["members"],
        tolerance_files=stats_file_set["tol"],
        max_member_count=2,
        max_factor=1.0e5,
    )

    assert os.path.isfile(
        stats_file_set["members"]
    ), "File 'selected_members.csv' was not created"

    with open(stats_file_set["members"], "r", encoding="utf-8") as file:
        content = file.read().strip()
    expected_content = "6,15\nexport FACTOR=26175"
    assert (
        content == expected_content
    ), "Increasing the factor within the member selection failed"


def test_select_members_failure(stats_file_set, caplog):
    log = run_select_members_cli(
        stats_file_set["stats"],
        stats_file_set["members"],
        tolerance_files=stats_file_set["tol"],
        max_member_count=1,
        min_factor=0.1,
        max_factor=1,
        log=caplog,
    )
    assert "ERROR" in log, "The member selection did not fail, although it should have."


def test_tolerance(stats_file_set, caplog):
    run_tolerance_cli(
        stats_file_set["stats"], stats_file_set["tol"], member_ids="1,2,3,4,5"
    )

    log = run_select_members_cli(
        stats_file_set["stats"],
        stats_file_set["members"],
        tolerance_files=stats_file_set["tol"],
        enable_check_only=True,
        log=caplog,
    )

    match = re.search(r".(\d+) member\(s\) out of 20 pass.", log)
    passed_count = match.group(1) if match else "0"
    error_message = (
        f"The test_tolerance output is incorrect. It should pass for "
        f"18 out of 20 but it passed for {passed_count} out of 20."
    )

    assert "18 member(s) out of 20" in log, error_message
