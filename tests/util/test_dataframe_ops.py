"""
This module contains unit tests for the `dataframe_ops.py` module.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from util.dataframe_ops import (
    compute_division,
    compute_rel_diff_dataframe,
    force_monotonic,
    parse_check,
    parse_probtest_csv,
    unify_time_index,
)


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


def test_force_monotonic():
    """
    Test that the function modify the dataframe forcing the values of every line
    to become non-decreasing monotonic along the columns
    """
    # Creation of a DataFrame with MultiIndex on the columns
    arrays = [
        ["var1", "var1", "var2", "var2"],
        ["mean", "max", "mean", "max"],
    ]
    columns = pd.MultiIndex.from_arrays(arrays)
    data = [
        [1, 5, 2, 7],
        [3, 2, 1, 9],
        [2, 8, 5, 4],
    ]
    df = pd.DataFrame(data, columns=columns)

    force_monotonic(df)

    # Property verification
    for stat in df.columns.levels[1]:
        sub_df = df.loc[:, (slice(None), stat)]
        assert (sub_df.diff(axis=1).fillna(0) >= 0).all().all()

    # Comparison with expected dataframe
    expected = pd.DataFrame([[1, 5, 2, 7], [3, 2, 3, 9], [2, 8, 5, 8]], columns=columns)
    pd.testing.assert_frame_equal(df, expected, check_exact=True)


def test_compute_rel_diff_basic():
    """
    Test that the function is giving the expected values with normal numbers
    """
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [1, 1], "B": [3, 5]})

    result = compute_rel_diff_dataframe(df1, df2)
    expected = pd.DataFrame([[0, 0], [1./3., 0.2]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_compute_rel_diff_with_negatives():
    """
    Test that the function is giving the expected values also with negative numbers
    """
    df1 = pd.DataFrame({"A": [-1, -2], "B": [3, -4]})
    df2 = pd.DataFrame({"A": [-2, -1], "B": [3, -5]})

    result = compute_rel_diff_dataframe(df1, df2)
    expected = pd.DataFrame([[1./2., 0], [1./3., 0.2]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_compute_rel_diff_with_zeros():
    """
    Test that the function is giving the expected values also with zeros in numerator
    """
    df1 = pd.DataFrame({"A": [0, 0], "B": [0, 0]})
    df2 = pd.DataFrame({"A": [1, -1], "B": [2, -2]})

    result = compute_rel_diff_dataframe(df1, df2)
    expected = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=True)


def test_compute_rel_diff_identical():
    """
    Test that the function is giving the expected values aift dataframe are identical
    """
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = df1.copy()

    result = compute_rel_diff_dataframe(df1, df2)

    assert (result == 0).all().all()


def test_compute_division_normal_case():
    """
    Test that the function is giving the expected values with normal numbers
    """
    df1 = pd.DataFrame({"A": [10, 20], "B": [30, 40]})
    df2 = pd.DataFrame({"A": [2, 4], "B": [5, 10]})

    result = compute_division(df1, df2)
    expected = pd.DataFrame({"A": [5.0, 5.0], "B": [6.0, 4.0]})

    pd.testing.assert_frame_equal(result, expected)


def test_compute_division_with_zero_in_denominator():
    """
    Test that the function is giving the expected values also with zeros in denominator
    """
    df1 = pd.DataFrame({"A": [10, 20], "B": [30, 40]})
    df2 = pd.DataFrame({"A": [0, 4], "B": [5, 0]})

    result = compute_division(df1, df2)
    expected = pd.DataFrame({"A": [np.nan, 5.0], "B": [6.0, np.nan]})

    pd.testing.assert_frame_equal(result, expected)


def test_division_with_zero_in_numerator():
    """
    Test that the function is giving the expected values also with zeros in numerator
    """
    df1 = pd.DataFrame({"A": [0, 20], "B": [0, 40]})
    df2 = pd.DataFrame({"A": [2, 4], "B": [5, 10]})

    result = compute_division(df1, df2)
    expected = pd.DataFrame({"A": [0.0, 5.0], "B": [0.0, 4.0]})

    pd.testing.assert_frame_equal(result, expected)


def test_division_both_zero():
    """
    Check the function in case all values are equal to zero
    """
    df1 = pd.DataFrame({"A": [0, 0], "B": [0, 0]})
    df2 = pd.DataFrame({"A": [0, 1], "B": [2, 0]})

    result = compute_division(df1, df2)
    expected = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})

    pd.testing.assert_frame_equal(result, expected)


# Creation of a temporary file for function test
@pytest.fixture(name="sample_csv", scope="function")
def fixture_sample_csv(tmp_path):
    csv_content = """col1,col2,3,3,2,2
sub1,sub2,A,B,A,B
a,b,1,3,5,7
d,e,2,4,6,8
"""
    file_path = tmp_path / "multi.csv"
    file_path.write_text(csv_content)
    return file_path


def test_parse_probtest_csv(sample_csv):
    """
    Check that the first multiindex of the rows is reversed because
    it is not in ascending order
    """
    df = parse_probtest_csv(sample_csv, index_col=[0, 1])

    expected = pd.DataFrame(
        {
            (2, "A"): [5, 6],
            (2, "B"): [7, 8],
            (3, "A"): [1, 2],
            (3, "B"): [3, 4],
        },
        index=pd.MultiIndex.from_tuples(
            [("a", "b"), ("d", "e")],
            names=["col1", "col2"],
        ),
    )
    expected.index.names = df.index.names
    expected.columns.names = df.columns.names

    pd.testing.assert_frame_equal(df, expected)


# Create a dataframe to test unify_time_index
@pytest.fixture(name="sample_unify_time", scope="module")
def fixture_sample_unify_time():
    features = ["A", "B"]
    times = [6, 4, 2]

    multi_index = pd.MultiIndex.from_product(
        [features, times], names=["feature", "time"]
    )

    data1 = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30],
    ]
    df1 = pd.DataFrame(data1, columns=multi_index)

    data2 = [
        [101, 102, 103, 104, 105, 106],
        [107, 108, 109, 110, 111, 112],
        [113, 114, 115, 116, 117, 118],
    ]
    df2 = pd.DataFrame(data2, columns=multi_index)

    fid_dfs = [df1, df2]
    return fid_dfs


def test_unify_time_index(sample_unify_time):
    """
    Test that the function unify column index and put it in
    ascending order and starting from 0
    """
    result_dfs = unify_time_index(sample_unify_time)

    features = ["A", "B"]
    times = [0, 1, 2]  # same as before but in ascending order and starting from 0

    multi_index = pd.MultiIndex.from_product(
        [features, times], names=["feature", "time"]
    )

    data1 = [
        [3, 2, 1, 6, 5, 4],
        [9, 8, 7, 12, 11, 10],
        [15, 14, 13, 18, 17, 16],
        [21, 20, 19, 24, 23, 22],
        [27, 26, 25, 30, 29, 28],
    ]
    df1 = pd.DataFrame(data1, columns=multi_index)

    data2 = [
        [103, 102, 101, 106, 105, 104],
        [109, 108, 107, 112, 111, 110],
        [115, 114, 113, 118, 117, 116],
    ]
    df2 = pd.DataFrame(data2, columns=multi_index)

    expected = [df1, df2]

    for res, exp in zip(result_dfs, expected):
        pd.testing.assert_frame_equal(res, exp)
