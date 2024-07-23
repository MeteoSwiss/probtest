import logging
import os
import random
import re

import click
import pytest

from engine.select_members import select_members
from tests.helpers import run_select_members_cli, run_tolerance_cli


def create_dummy_stats_file(filename, configurations, seed, perturbation):
    random.seed(seed)
    max_time_dim = max(config["time_dim"] for config in configurations)
    time_header = ",".join(f"{t},{t},{t}" for t in range(max_time_dim))
    header = [
        f"time,,,{time_header}",
        "statistic,," + ",max,mean,min" * max_time_dim,
        "file_ID,variable,height,,,,,,,,,",
    ]

    data = []
    for config in configurations:
        time_dim = config["time_dim"]
        height_dim = config["height_dim"]
        variable = config["variable"]
        file_format = config["file_format"]

        for h in range(height_dim):
            row = f"{file_format},{variable},{h}.0"
            for t in range(time_dim):
                base_mean = round(random.uniform(0, 5), 5)
                mean = base_mean + round(random.uniform(-perturbation, perturbation), 5)
                max_val = mean + round(random.uniform(0, perturbation), 5)
                min_val = mean - round(random.uniform(0, perturbation), 5)
                row += f",{max_val},{mean},{min_val}"
            for _ in range(time_dim, max_time_dim):
                row += ",,,"
            data.append(row)

    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")
        for row in data:
            f.write(row + "\n")


@pytest.fixture(scope="module")
def setup_files():
    configurations = [
        {
            "time_dim": 3,
            "height_dim": 5,
            "variable": "v1",
            "file_format": "Format1:*test_3d*.nc",
        },
        {
            "time_dim": 3,
            "height_dim": 2,
            "variable": "v2",
            "file_format": "Format2:*test_2d*.nc",
        },
        {
            "time_dim": 2,
            "height_dim": 4,
            "variable": "v3",
            "file_format": "Format3:*test_2d*.nc",
        },
    ]
    seed = 42
    create_dummy_stats_file("stats_ref.csv", configurations, seed - 1, 0.0)
    for i in range(1, 50):
        filename = f"stats_{i}.csv"
        create_dummy_stats_file(filename, configurations, seed + i, 1e-3)
    create_dummy_stats_file("stats_50.csv", configurations, seed + 50, 1e2)
    yield
    # Teardown code to remove created files
    os.remove("stats_ref.csv")
    for i in range(1, 51):
        os.remove(f"stats_{i}.csv")
    if os.path.exists("selected_members.csv"):
        os.remove("selected_members.csv")
    if os.path.exists("tolerance.csv"):
        os.remove("tolerance.csv")


def test_select_members(setup_files):
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


def test_select_members_increase_factor(setup_files):
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


def test_select_members_failure(setup_files, caplog):
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit):
        context = click.Context(select_members)
        context.invoke(
            select_members,
            stats_file_name="stats_{member_id}.csv",
            selected_members_file_name="selected_members.csv",
            tolerance_file_name="tolerance.csv",
            max_member_num=5,
            iterations=1,
            max_factor=5,
        )
    assert (
        "ERROR" in caplog.text
    ), "The member selection did not fail, although it should have."


def test_test_tolerance(setup_files, caplog):
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
