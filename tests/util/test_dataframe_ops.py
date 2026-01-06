"""
This module contains unit tests for the `dataframe_ops.py` module.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from util.constants import CHECK_THRESHOLD
from util.dataframe_ops import (
    check_intersection,
    check_variable,
    compute_division,
    compute_rel_diff_dataframe,
    df_from_file_ids,
    force_monotonic,
    has_enough_data,
    multiple_solutions_from_dict,
    parse_check,
    parse_probtest_fof,
    parse_probtest_stats,
    read_input_file,
    split_feedback_dataset,
    unify_time_index,
)


@pytest.fixture(name="_tmp_netcdf_files", scope="function")
def fixture_tmp_netcdf_files(tmp_dir):
    file_id = [("netcdf", "*atm_3d_ml*.nc")]

    _, pattern = file_id[0]
    base_name = pattern.replace("*", "")

    created_files = []

    for i in range(1, 2):
        filename = f"test{i}_{base_name}"
        tmp_dir = Path(tmp_dir)
        file_path = tmp_dir / filename

        ncells = 5
        ncells_2 = 4
        nedges = 3
        nvertices = 2
        time = np.array([0, 1]) + i * 10

        data = xr.Dataset(
            {
                "var_ncells": (
                    ("time", "ncells"),
                    np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]]),
                ),
                "var_ncells_2": (
                    ("time", "ncells_2"),
                    np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
                ),
                "var_nedges": (
                    ("time", "nedges"),
                    np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
                ),
                "var_nvertices": (
                    ("time", "nvertices"),
                    np.array([[100.0, 200.0], [300.0, 400.0]]),
                ),
            },
            coords={
                "time": time,
                "ncells": np.arange(ncells),
                "ncells_2": np.arange(ncells_2),
                "nedges": np.arange(nedges),
                "nvertices": np.arange(nvertices),
            },
        )

        data.to_netcdf(file_path)
        created_files.append(file_path)

    return created_files


def test_df_from_file_ids(_tmp_netcdf_files, tmp_dir):
    """
    Test that the function collects the dataframes for each combination of file
    ID and specification.
    """

    file_id1 = [("netcdf", "*atm_3d_ml*.nc")]
    file_id2 = [("netcdf", "*atm_3d_ml*wrong.nc")]

    file_specification1 = {
        "netcdf": {
            "format": "netcdf",
            "time_dim": "time",
            "horizontal_dims": ["ncells", "ncells_2", "nedges", "nvertices"],
        },
    }

    file_specification2 = {
        "netcdfwrong": {
            "format": "netcdf",
            "time_dim": "time",
            "horizontal_dims": ["ncells", "ncells_2", "nedges", "nvertices"],
        },
    }

    with pytest.raises(SystemExit) as e:
        df_from_file_ids(file_id2, tmp_dir, file_specification1)

    assert e.value.code == 2

    with pytest.raises(SystemExit) as e:
        df_from_file_ids(file_id1, tmp_dir, file_specification2)

    assert e.value.code == 1


def test_read_input_file(tmp_dir):
    """
    Test that the file's name is correctly read using the specification.
    """
    file_id = [("netcdf", "*atm_3d_ml*.nc")]

    input_dir = tmp_dir

    file_specification = {
        "netcdf": {
            "format": "wrong_format",
            "time_dim": "time",
            "horizontal_dims": ["ncells", "ncells_2", "nedges", "nvertices"],
        }
    }

    for file_type, file_pattern in file_id:
        f = "test1_atm_3d_ml.nc"

        with pytest.raises(SystemExit) as e:
            read_input_file(
                label=f"{file_type}:{file_pattern}",
                file_name=f"{input_dir}/{f}",
                specification=file_specification[file_type],
            )

        assert e.value.code == 1


@patch("util.dataframe_ops.parse_probtest_stats")
def test_parse_check(mock_parse_probtest_stats, setup_csv_files):
    # Mock the return value of parse_probtest_stats
    mock_parse_probtest_stats.side_effect = lambda file, index_col: pd.read_csv(
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
    Test that the function is giving the expected values with basic numbers
    """
    df1 = pd.DataFrame([[1.0, 3.0], [2.0, 4.0]], columns=["A", "B"])
    df2 = pd.DataFrame([[1.0, 3.0], [1.0, 5.0]], columns=["A", "B"])

    result = compute_rel_diff_dataframe(df1, df2)
    expected = pd.DataFrame([[0.0, 0.0], [1.0 / 3.0, 0.2]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_compute_rel_diff_with_negatives():
    """
    Test that the function is giving the expected values also with negative numbers
    """
    df1 = pd.DataFrame([[-1.0, 3.0], [-2.0, -4.0]], columns=["A", "B"])
    df2 = pd.DataFrame([[-2.0, 3.0], [-1.0, -5.0]], columns=["A", "B"])

    result = compute_rel_diff_dataframe(df1, df2)
    expected = pd.DataFrame([[0.5, 0.0], [1.0 / 3.0, 0.2]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_compute_rel_diff_with_zeros():
    """
    Test that the function is giving the expected values also with zeros in numerator
    """
    df1 = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], columns=["A", "B"])
    df2 = pd.DataFrame([[1.0, 2.0], [-1.0, -2.0]], columns=["A", "B"])

    result = compute_rel_diff_dataframe(df1, df2)
    expected = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_compute_rel_diff_identical():
    """
    Test that the function is giving the expected values aift dataframe are identical
    """
    df1 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["A", "B"])
    df2 = df1.copy()

    result = compute_rel_diff_dataframe(df1, df2)

    assert (result == 0).all().all()


def test_compute_division_basic():
    """
    Test that the function is giving the expected values with basic numbers
    """
    df1 = pd.DataFrame([[10.0, 20.0], [30.0, 40.0]], columns=["A", "B"])
    df2 = pd.DataFrame([[2.0, 4.0], [5.0, 10.0]], columns=["A", "B"])

    result = compute_division(df1, df2)
    expected = pd.DataFrame([[5.0, 5.0], [6.0, 4.0]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_compute_division_with_zero_in_denominator():
    """
    Test that the function is giving the expected values also with zeros in denominator
    """
    df1 = pd.DataFrame([[10.0, 20.0], [30.0, 40.0]], columns=["A", "B"])
    df2 = pd.DataFrame([[0.0, 4.0], [5.0, 0.0]], columns=["A", "B"])

    result = compute_division(df1, df2)
    expected = pd.DataFrame([[np.nan, 5.0], [6.0, np.nan]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_division_with_zero_in_numerator():
    """
    Test that the function is giving the expected values also with zeros in numerator
    """
    df1 = pd.DataFrame([[0.0, 20.0], [0.0, 40.0]], columns=["A", "B"])
    df2 = pd.DataFrame([[2.0, 4.0], [5.0, 10.0]], columns=["A", "B"])

    result = compute_division(df1, df2)
    expected = pd.DataFrame([[0.0, 5.0], [0.0, 4.0]], columns=["A", "B"])

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_division_both_zero():
    """
    Check the function in case all values are equal to zero
    """
    df1 = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})
    df2 = pd.DataFrame({"A": [0.0, 1.0], "B": [2.0, 0.0]})

    result = compute_division(df1, df2)
    expected = pd.DataFrame({"A": [0.0, 0.0], "B": [0.0, 0.0]})

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


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


def test_parse_probtest_stats(sample_csv):
    """
    Check that the first multiindex of the rows is reversed because
    it is not in ascending order
    """
    df = parse_probtest_stats(sample_csv, index_col=[0, 1])

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


@pytest.fixture(name="sample_csv_3_cols", scope="function")
def fixture_sample_csv_3cols(tmp_path):
    csv_content = """file_ID,variable,height,1,1,2,2
sub1,var_1,0,1,2,3,4
sub1,var_1,1,5,6,7,8
sub1,var_2,0,9,10,11,12
sub2,var_1,0,13,14,15,16
"""
    file_path_3_cols = tmp_path / "multi.csv"
    file_path_3_cols.write_text(csv_content)
    return file_path_3_cols


def test_parse_probtest_stats_3cols(sample_csv_3_cols, index_col=None):
    """
    Test that if no index_col is given, the default values is [0,1,2].
    """

    parse_probtest_stats(sample_csv_3_cols, index_col)

    if index_col is None:
        index_col = [0, 1, 2]
    assert index_col == [0, 1, 2]


@pytest.fixture(name="sample_df_with_obs", scope="function")
def fixture_sample_df_with_obs():
    df = pd.DataFrame(
        {
            "d_body": [0, 2, 1, 3, 5, 4],
            "lat": [1, 2, 3, 4, 5, 5],
            "lon": [5, 7, 9, 8, 3, 3],
            "varno": [3, 4, 3, 4, 4, 4],
            "statid": ["a", "c", "b", "d", "e", "e"],
            "time_nomi": [0, 0, 30, 30, 60, 60],
            "level": [1000, 900, 950, 850, 750, 800],
            "veri_data": [45, 45, 34, 56, 78, 67],
            "obs": [3.74e-01, 7.31e-01, 9.50e-01, 5.98e-01, 1.55e-01, 1.56e-01],
            "bcor": [5.80e-02, 6.01e-01, 8.66e-01, 7.08e-01, 9.69e-01, 2.00e-02],
            "level_typ": [8.32e-01, 1.81e-01, 2.12e-01, 1.83e-01, 5.24e-01, 3.04e-01],
            "level_sig": [4.31e-01, 6.11e-01, 2.91e-01, 1.39e-01, 3.66e-01, 2.92e-01],
            "state": [1, 1, 1, 1, 1, 1],
            "flags": [9, 9, 9, 9, 9, 9],
            "check": [13, 13, 13, 13, 13, 13],
            "e_o": [3.40e-02, 2.58e-01, 9.09e-01, 6.62e-01, 5.20e-01, 3.11e-01],
            "qual": [7.96e-01, 8.10e-01, 5.09e-01, 1.63e-01, 1.38e-01, 4.25e-01],
            "plevel": [8.01e-01, 7.70e-02, 4.06e-01, 8.47e-01, 7.55e-01, 3.20e-01],
        }
    )

    return df


@pytest.fixture(name="sample_df_with_rep", scope="function")
def fixture_sample_df_with_rep():
    df = pd.DataFrame(
        {
            "d_hdr": [0, 2, 1, 3, 4],
            "lat": [1, 2, 3, 4, 5],
            "lon": [5, 7, 9, 8, 3],
            "statid": ["a", "c", "b", "d", "e"],
            "time_nomi": [0, 0, 30, 30, 60],
            "codetype": [5, 5, 5, 5, 5],
            "l_body": [1, 1, 1, 1, 2],
            "i_body": [1, 3, 2, 4, 5],
        }
    )

    return df


def test_parse_probtest_fof(
    sample_dataset_fof, tmp_path, sample_df_with_obs, sample_df_with_rep
):

    fake_path = tmp_path / "sample_dataset_fof.nc"
    sample_dataset_fof.to_netcdf(fake_path)
    df_rep, df_obs = parse_probtest_fof(fake_path)

    assert df_rep.equals(sample_df_with_rep)
    assert df_obs.equals(sample_df_with_obs)


@pytest.fixture(name="ds_no_intersection", scope="function")
def sample_dataframes_no_index_overlap():
    """
    Returns two DataFrames that do not have indices in common,
    so that check_intersection() fails.
    """
    df1 = pd.DataFrame(
        {
            "var_a": np.arange(5),
            "var_b": np.arange(10, 15),
        },
        index=[0, 1, 2, 3, 4],
    )

    df2 = pd.DataFrame(
        {
            "var_c": np.arange(20, 25),
            "var_d": np.arange(30, 35),
        },
        index=[5, 6, 7, 8, 9],
    )

    return df1, df2


@pytest.fixture(name="ds_intersection", scope="function")
def sample_dataframes_prova():

    df1 = pd.DataFrame(
        {
            "var_a": np.arange(5),
            "var_b": np.arange(10, 15),
            "var_c": ["x", "y", "z", "w", "v"],
        },
        index=[0, 1, 2, 3, 4],
    )

    df2 = pd.DataFrame(
        {
            "var_a": np.arange(5, 10),
            "var_c": ["a", "b", "c", "d", "e"],
            "var_d": np.arange(20, 25),
        },
        index=[0, 1, 2, 3, 4],
    )

    return df1, df2


def test_check_intersection_fail(ds_no_intersection):
    """
    Test that with no intersection, check_intersection fails.
    """
    df1, df2 = ds_no_intersection

    skip_test, _, _ = check_intersection(df1, df2)
    assert skip_test == 1


def test_check_intersection_pass(ds_intersection):
    """
    Test that with intersection, check_intersection passes.
    """
    df1, df2 = ds_intersection

    skip_test, df_ref, df_cur = check_intersection(df1, df2)

    assert skip_test == 0
    assert df_ref.equals(df1)
    assert df_cur.equals(df2)


def test_check_variable():
    diff_df = pd.DataFrame([[1.0, 3.0], [2.0, 4.0]], columns=["A", "B"])
    df_tol = pd.DataFrame([[1.0, 3.0], [1.0, 5.0]], columns=["A", "B"])

    diff_df_expected = pd.DataFrame([[2.0, 4.0]], columns=["A", "B"], index=[1])
    df_tol_expected = pd.DataFrame([[1.0, 5.0]], columns=["A", "B"], index=[1])

    a, b, c = check_variable(diff_df, df_tol)

    assert a is False
    assert b.equals(diff_df_expected)
    assert c.equals(df_tol_expected)


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


def test_has_enough_data():
    """
    Test that without sufficient data, the test will fail.
    """
    df_empty = pd.DataFrame()

    with pytest.raises(SystemExit) as e:
        has_enough_data(df_empty)

    assert e.value.code == 1


@pytest.fixture(name="stats_dataframes", scope="function")
def fixture_dataframes():
    """
    Create stats dataframes and reference tolerances.
    """
    index = pd.MultiIndex.from_arrays(
        [
            ["NetCDF:*atm_3d*.nc"] * 4,
            ["var_1"] * 3 + ["var_2"],
            list(range(3)) + [0],
        ],
        names=["file_ID", "variable", "height"],
    )
    columns = pd.MultiIndex.from_product(
        [range(4), ["max", "mean", "min"]], names=["time", "statistic"]
    )

    array1 = np.linspace(0.9, 1.9, 4 * 12).reshape(4, 12)
    array1[:2] *= -1
    df1 = pd.DataFrame(array1, index=index, columns=columns)

    array2 = np.linspace(1.1, 2.1, 4 * 12).transpose().reshape(4, 12)
    array2[:2] *= -1
    df2 = pd.DataFrame(array2, index=index, columns=columns)

    tol_large = pd.DataFrame(
        np.ones((2, 12)) * 0.21, index=["var_1", "var_2"], columns=columns
    )
    tol_small = pd.DataFrame(
        np.ones((2, 12)) * 0.06, index=["var_1", "var_2"], columns=columns
    )

    return df1, df2, tol_large, tol_small


@pytest.fixture(name="fof_datasets", scope="function")
def fixture_fof_datasets(sample_dataset_fof):
    """
    Create fof dataset and reference tolerances.
    """
    ds1 = sample_dataset_fof
    ds2 = ds1.copy(deep=True)

    ds2["veri_data"] = (("d_body",), ds2["veri_data"].values * 1.55)

    n_body_size = ds1.sizes["d_body"]

    tol_large = pd.DataFrame({"veri_data": np.full(n_body_size, 5)})
    tol_small = pd.DataFrame({"veri_data": np.full(n_body_size, 0.06)})

    return ds1, ds2, tol_large, tol_small


def _check(df1, df2, tol_large, tol_small, file_type):

    diff_df = compute_rel_diff_dataframe(df1, df2)

    if file_type == "stats":
        diff_df = diff_df.groupby(["variable"]).max()
    if file_type == "fof":
        diff_df = diff_df.to_frame()

    out1, err1, _ = check_variable(diff_df, tol_large)
    out2, err2, _ = check_variable(diff_df, tol_small)

    assert out1, (
        "Check with large tolerances did not validate! "
        + f"Here is the DataFrame:\n{err1}"
    )
    assert not out2, (
        f"Check with small tolerances did validate! " f"Here is the DataFrame:\n{err2}"
    )


def test_check_stats(stats_dataframes):
    df1, df2, tol_large, tol_small = stats_dataframes
    _check(df1, df2, tol_large, tol_small, file_type="stats")


def test_check_fof_with_tol(fof_datasets):
    ds1, ds2, tol_large, tol_small = fof_datasets
    _, ds_veri1 = split_feedback_dataset(ds1)
    _, ds_veri2 = split_feedback_dataset(ds2)
    df_veri1 = ds_veri1.to_dataframe().reset_index()
    df_veri2 = ds_veri2.to_dataframe().reset_index()
    _check(
        df_veri1["veri_data"],
        df_veri2["veri_data"],
        tol_large,
        tol_small,
        file_type="fof",
    )


def test_check_one_zero_stats(stats_dataframes):
    """
    Test that a null value in ds1 causes failure,
    and that a variation within tolerance is accepted.
    """
    df1, df2, tol_large, tol_small = stats_dataframes
    df1 = df1.copy()
    df1.loc[("NetCDF:*atm_3d*.nc", "var_1", 2), (0, "max")] = 0

    diff_df = compute_rel_diff_dataframe(df1, df2)
    diff_df = diff_df.groupby(["variable"]).max()
    out, err, _ = check_variable(diff_df, tol_large)

    assert not out, f"Check with 0-value reference validated incorrectly:\n{err}"

    df2 = df2.copy()
    df2.loc[("NetCDF:*atm_3d*.nc", "var_1", 2), (0, "max")] = CHECK_THRESHOLD / 2
    _check(df1, df2, tol_large, tol_small, file_type="stats")


def test_check_one_zero_fof(fof_datasets):
    """
    Similar to test_check_one_zero, but applied to fof files.
    """
    ds1, ds2, tol_large, tol_small = fof_datasets

    ds2_copy = ds2.copy(deep=True)
    ds1["veri_data"][2] = 0
    ds2_copy["veri_data"][2] = CHECK_THRESHOLD / 2

    _, ds_veri1 = split_feedback_dataset(ds1)
    _, ds_veri2 = split_feedback_dataset(ds2)
    df_veri1 = ds_veri1.to_dataframe().reset_index()
    df_veri2 = ds_veri2.to_dataframe().reset_index()

    diff_df = compute_rel_diff_dataframe(df_veri1["veri_data"], df_veri2["veri_data"])
    diff_df = diff_df.to_frame()

    out, err, _ = check_variable(diff_df, tol_large)

    assert not out, f"Check with 0-value reference validated incorrectly:\n{err}"

    _, ds_veri2_copy = split_feedback_dataset(ds2_copy)
    ds_veri2_copy = ds_veri2_copy.copy(deep=True)
    ds_veri2_copy["veri_data"][2] = CHECK_THRESHOLD / 2
    _check(
        df_veri1["veri_data"],
        ds_veri2_copy["veri_data"],
        tol_large,
        tol_small,
        file_type="fof",
    )


def test_check_smalls_stats(stats_dataframes):
    """
    Both values are close to 0 and should be accepted even though
    their relative difference is large.
    Close to 0 means < util.constants.CHECK_THRESHOLD.
    """

    df1, df2, tol_large, tol_small = stats_dataframes
    df1 = df1.copy()
    df2 = df2.copy()

    df1.loc["var_1", (2, "min")] = CHECK_THRESHOLD * 1e-5
    df2.loc["var_1", (2, "min")] = -CHECK_THRESHOLD / 2

    _check(df1, df2, tol_large, tol_small, file_type="stats")


def test_check_smalls_fof(fof_datasets):
    """
    Test similar to test_check_smalls but on FOF datasets.
    """
    ds1, ds2, tol_large, tol_small = fof_datasets

    ds1 = ds1.copy(deep=True)
    ds2 = ds2.copy(deep=True)

    ds1["veri_data"][2] = CHECK_THRESHOLD * 1e-5
    ds2["veri_data"][2] = -CHECK_THRESHOLD / 2

    _, ds_veri1 = split_feedback_dataset(ds1)
    _, ds_veri2 = split_feedback_dataset(ds2)
    df_veri1 = ds_veri1.to_dataframe().reset_index()
    df_veri2 = ds_veri2.to_dataframe().reset_index()

    _check(
        df_veri1["veri_data"],
        df_veri2["veri_data"],
        tol_large,
        tol_small,
        file_type="fof",
    )


@pytest.fixture(name="dataframes_dict", scope="function")
def sample_dataframes_dict():
    """
    Returns a dictionary containing two simple DataFrames:
    - 'reports': example report data
    - 'observation': example observation data
    """

    df_reports = pd.DataFrame(
        {"lat": [40, 41, 42], "value": [10, 20, 30], "r_check": [1, 1, 1]}
    )

    df_observation = pd.DataFrame(
        {"id": [1, 2, 3], "check": [9, 9, 9], "state": [13, 13, 13]}
    )

    df_dict = {"reports": df_reports, "observation": df_observation}

    return df_dict


def test_multiple_solutions_from_dict_no_rules(dataframes_dict):

    dict_ref = dataframes_dict
    dict_cur = {key: df.copy() for key, df in dict_ref.items()}
    rules = ""

    errors = multiple_solutions_from_dict(dict_ref, dict_cur, rules)
    assert errors == []


def test_multiple_solutions_from_dict_with_rules(dataframes_dict):
    dict_ref = dataframes_dict
    dict_cur = {key: df.copy() for key, df in dict_ref.items()}
    dict_cur["observation"].loc[1, "check"] = 1
    dict_cur["observation"].loc[1, "state"] = 14

    rules = {"check": [9, 1], "state": [13, 14]}

    errors = multiple_solutions_from_dict(dict_ref, dict_cur, rules)
    assert errors == []


def test_multiple_solutions_from_dict_with_rules_wrong(dataframes_dict):
    dict_ref = dataframes_dict
    dict_cur = {key: df.copy() for key, df in dict_ref.items()}
    dict_cur["observation"].loc[1, "check"] = 6
    dict_cur["observation"].loc[1, "state"] = 14

    rules = {"check": [9, 1], "state": [13, 14]}

    errors = multiple_solutions_from_dict(dict_ref, dict_cur, rules)

    expected = [
        {
            "row": 1,
            "column": "check",
            "file1": np.int64(9),
            "file2": np.int64(6),
            "error": "values different and not admitted",
        }
    ]
    assert errors == expected
