import logging
import os
import random
import re
import unittest
from io import StringIO

import click
from click.testing import CliRunner

from engine.select_members import select_members
from engine.tolerance import tolerance


def create_dummy_stats_file(filename, configurations, seed, perturbation):
    random.seed(seed)
    # Create header
    max_time_dim = max([config["time_dim"] for config in configurations])
    time_header = ",".join([f"{t},{t},{t}" for t in range(max_time_dim)])
    header = [
        f"time,,,{time_header}",
        "statistic,," + ",max,mean,min" * max_time_dim,
        "file_ID,variable,height,,,,,,,,,",
    ]

    # Generate data rows
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
            # Fill in remaining time slots with empty values if needed
            for _ in range(time_dim, max_time_dim):
                row += ",,,"
            data.append(row)

    # Write to file
    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")
        for row in data:
            f.write(row + "\n")


class TestMemberSelection(unittest.TestCase):

    def setUp(self):
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
        # Create reference file
        create_dummy_stats_file("stats_ref.csv", configurations, seed - 1, 0.0)
        # Create 49 stats files
        for i in range(1, 50):
            filename = f"stats_{i}.csv"
            create_dummy_stats_file(filename, configurations, seed + i, 1e-3)
        # Create a stat file with higher perturbation to fail the tolerance test
        create_dummy_stats_file("stats_50.csv", configurations, seed + 50, 1e2)

    def test_select_members(self):
        # Test successful member selection
        context = click.Context(select_members)
        context.invoke(
            select_members,
            stats_file_name="stats_{member_id}.csv",
            selected_members_file_name="selected_members.csv",
            tolerance_file_name="tolerance.csv",
        )

        self.assertTrue(
            os.path.isfile("selected_members.csv"),
            "File 'selected_members.csv' was not created",
        )

        with open("selected_members.csv", "r") as file:
            content = file.read().strip()
        expected_content = "50,21,40,39,16\nexport FACTOR=5"
        self.assertTrue(content == expected_content, "The member selection failed")

    def test_select_members_increase_factor(self):
        # Set max_member_num equal to min_member_num (default) and iterations to 1
        # to ensure the factor has to be increased after the first selection
        context = click.Context(select_members)
        context.invoke(
            select_members,
            stats_file_name="stats_{member_id}.csv",
            selected_members_file_name="selected_members.csv",
            tolerance_file_name="tolerance.csv",
            max_member_num=5,
            iterations=1,
        )

        self.assertTrue(
            os.path.isfile("selected_members.csv"),
            "File 'selected_members.csv' was not created",
        )

        with open("selected_members.csv", "r") as file:
            content = file.read().strip()
        expected_content = "50,21,40,39,16\nexport FACTOR=10"
        self.assertTrue(
            content == expected_content,
            "Increasing the factor within the member selection failed",
        )

    def test_select_members_failure(self):
        # Set up logging to capture error output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)
        logger.addHandler(handler)

        # Set max_member_num equal to min_member_num (default) and iterations to 1
        # and max_factor to 5. Like this there will only be one random selection
        # and the member selection should fail
        try:
            # Handle SystemExit which will be triggered on failure of the
            # member selection
            with self.assertRaises(SystemExit):  # Expect SystemExit
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

            # Ensure logs are flushed and captured
            handler.flush()
            log_output = log_stream.getvalue()

            # Check log output for error messages
            self.assertTrue(
                "ERROR" in log_output,
                "The member selection did not fail, although it should have.",
            )

        except AssertionError:
            # Ensure logs are flushed and captured before re-raising the exception
            handler.flush()
            log_output = log_stream.getvalue()

            # Check log output for error messages
            self.assertTrue(
                "ERROR" in log_output,
                "The member selection did not fail, although it should have.",
            )

        # Clean up
        logger.removeHandler(handler)
        log_stream.close()

    def test_test_tolerance(self):
        # Create tolerances out of the first five members
        context = click.Context(tolerance)
        context.invoke(
            tolerance,
            stats_file_name="stats_{member_id}.csv",
            tolerance_file_name="tolerance.csv",
            member_num=[5],
        )

        # Set up logging to capture output
        log_stream = StringIO()
        logging.basicConfig(stream=log_stream, level=logging.INFO, format="%(message)s")

        runner = CliRunner()
        runner.invoke(
            select_members,
            [
                "--stats-file-name",
                "stats_{member_id}.csv",
                "--selected-members-file-name",
                "selected_members.csv",
                "--tolerance-file-name",
                "tolerance.csv",
                "--test-tolerance",
            ],
        )

        # Capture and filter log output
        log_output = log_stream.getvalue()

        # Extract and validate the number of passed tests from the log output
        match = re.search(r"passed for (\d+) out of 50", log_output)
        passed_count = match.group(1)
        error_message = (
            f"The test-tolerance output is incorrect. It should pass for "
            f"49 out of 50 but it passed for {passed_count} out of 50."
        )

        self.assertTrue("49 out of 50" in log_output, error_message)


if __name__ == "__main__":
    unittest.main()
