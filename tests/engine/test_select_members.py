import os
import re

from tests.helpers import run_select_members_cli, run_tolerance_cli


def test_select_members(stats_file_set):
    run_select_members_cli(
        "stats_{member_id}.csv", "selected_members.csv", "tolerance.csv"
    )

    assert os.path.isfile(
        "selected_members.csv"
    ), "File 'selected_members.csv' was not created"

    with open("selected_members.csv", "r") as file:
        content = file.read().strip()
    expected_content = "50,21,40,39,16\nexport FACTOR=5"
    assert content == expected_content, "The member selection failed"


def test_select_members_increase_factor(stats_file_set):
    run_select_members_cli(
        "stats_{member_id}.csv",
        "selected_members.csv",
        "tolerance.csv",
        max_member_num=5,
        iterations=1,
    )

    assert os.path.isfile(
        "selected_members.csv"
    ), "File 'selected_members.csv' was not created"

    with open("selected_members.csv", "r") as file:
        content = file.read().strip()
    expected_content = "50,21,40,39,16\nexport FACTOR=10"
    assert (
        content == expected_content
    ), "Increasing the factor within the member selection failed"


def test_select_members_failure(stats_file_set, caplog):
    log = run_select_members_cli(
        "stats_{member_id}.csv",
        "selected_members.csv",
        "tolerance.csv",
        max_member_num=5,
        iterations=1,
        max_factor=5,
        log=caplog,
    )
    assert "ERROR" in log, "The member selection did not fail, although it should have."


def test_test_tolerance(stats_file_set, caplog):
    run_tolerance_cli("stats_{member_id}.csv", "tolerance.csv", member_num=5)

    log = run_select_members_cli(
        "stats_{member_id}.csv",
        "selected_members.csv",
        "tolerance.csv",
        test_tolerance=True,
        log=caplog,
    )

    match = re.search(r"passed for (\d+) out of 50", log)
    passed_count = match.group(1) if match else "0"
    error_message = (
        f"The test-tolerance output is incorrect. It should pass for "
        f"49 out of 50 but it passed for {passed_count} out of 50."
    )

    assert "49 out of 50" in log, error_message
