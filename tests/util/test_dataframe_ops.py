"""
This module contains unit tests for the `dataframe_ops.py` module.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from util.dataframe_ops import parse_check


@pytest.fixture(name="setup_csv_files")
def fixture_setup_csv_files(tmp_path):
    # Create sample CSV files for testing
    tolerance_data = pd.DataFrame(
        {"A": [0.1, 0.2], "B": [0.3, 0.4]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b"), ("c", "d")], names=["col1", "col2"]
        ),
    )
    ref_data = pd.DataFrame(
        {"A": [1, 2], "B": [3, 4]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b", "c"), ("d", "e", "f")], names=["col1", "col2", "col3"]
        ),
    )
    cur_data = pd.DataFrame(
        {"A": [2, 3], "B": [4, 5]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b", "c"), ("d", "e", "f")], names=["col1", "col2", "col3"]
        ),
    )

    tolerance_file = tmp_path / "tolerance_test.csv"
    ref_file = tmp_path / "input_ref_test.csv"
    cur_file = tmp_path / "input_cur_test.csv"

    tolerance_data.to_csv(tolerance_file)
    ref_data.to_csv(ref_file)
    cur_data.to_csv(cur_file)

    return {
        "tolerance_file": tolerance_file,
        "ref_file": ref_file,
        "cur_file": cur_file,
    }


@patch("util.dataframe_ops.parse_probtest_csv")
@patch("util.dataframe_ops.logger")
def test_parse_check(mock_logger, mock_parse_probtest_csv, setup_csv_files):
    # Mock the return value of parse_probtest_csv
    mock_parse_probtest_csv.side_effect = lambda file, index_col: pd.read_csv(
        file, index_col=index_col
    )

    factor = 2.0

    df_tol, df_ref, df_cur = parse_check(
        setup_csv_files["tolerance_file"],
        setup_csv_files["ref_file"],
        setup_csv_files["cur_file"],
        factor,
    )

    # Check that the tolerance DataFrame has been scaled
    expected_tol = pd.DataFrame(
        {"A": [0.2, 0.4], "B": [0.6, 0.8]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b"), ("c", "d")], names=["col1", "col2"]
        ),
    )

    pd.testing.assert_frame_equal(df_tol, expected_tol)

    # Check that the reference and current DataFrames are read correctly
    expected_ref = pd.DataFrame(
        {"A": [1, 2], "B": [3, 4]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b", "c"), ("d", "e", "f")], names=["col1", "col2", "col3"]
        ),
    )

    expected_cur = pd.DataFrame(
        {"A": [2, 3], "B": [4, 5]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b", "c"), ("d", "e", "f")], names=["col1", "col2", "col3"]
        ),
    )

    pd.testing.assert_frame_equal(df_ref, expected_ref)
    pd.testing.assert_frame_equal(df_cur, expected_cur)

    # Check logging
    mock_logger.info.assert_any_call("applying a factor of %s to the spread", factor)
    mock_logger.info.assert_any_call(
        "checking %s against %s using tolerances from %s",
        setup_csv_files["cur_file"],
        setup_csv_files["ref_file"],
        setup_csv_files["tolerance_file"],
    )
