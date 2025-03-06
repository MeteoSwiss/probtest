"""
This module contains unit tests for the `dataframe_ops.py` module.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from util.dataframe_ops import adjust_time_index, parse_check


def test_adjust_time_index():
    # Create a sample DataFrame with MultiIndex for 'time' and 'statistic'
    index = pd.MultiIndex.from_product(
        [["0 days 00:00:00", "0 days 00:01:00"], ["mean", "max", "min"]],
        names=["time", "statistic"],
    )
    data = np.random.random((5, 6))  # Sample data
    df = pd.DataFrame(data, columns=index)

    # Apply the function to a list of DataFrames (in this case, just one DataFrame)
    result = adjust_time_index([df])[0]

    # Create the expected new time index based on unique statistics
    expected_time_values = np.repeat(
        range(2), 3
    )  # Two unique time points, three statistics per time
    expected_index = pd.MultiIndex.from_arrays(
        [expected_time_values, ["mean", "max", "min"] * 2],
        names=["time", "statistic"],
    )

    # Verify that the new MultiIndex matches the expected index
    assert result.columns.equals(
        expected_index
    ), "The time index was not adjusted correctly."

    # Verify that the data remains unchanged
    pd.testing.assert_frame_equal(df, result, check_like=True)


@patch("util.dataframe_ops.parse_probtest_csv")
def test_parse_check(mock_parse_probtest_csv, setup_csv_files):
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
