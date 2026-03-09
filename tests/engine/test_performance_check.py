"""
This module contains unit tests to verify the functionality of performance-check
CLI commands.
"""

from click.testing import CliRunner

from engine.performance_check import performance_check

JSON_REFERENCE = "tests/data/reference"
JSON_ADD = "tests/data/add"
JSON_REF = "tests/data/ref"


def test_performance_check_pass(caplog):
    """Current runtime within allowed limits → PASS"""
    runner = CliRunner()

    with caplog.at_level("INFO"):
        result = runner.invoke(
            performance_check,
            [
                "--timing-current",
                JSON_ADD,
                "--timing-reference",
                JSON_REF,
                "--timer-sections",
                "total,physics,nh_solve",
            ],
        )

    assert result.exit_code == 0
    assert "PASSED" in caplog.text
