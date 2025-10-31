"""
This module contains test cases to validate the functionality
of check CLI commands.
"""

import numpy as np
import pandas as pd
import pytest

from util.constants import CHECK_THRESHOLD
from util.dataframe_ops import (
    check_variable,
    compute_rel_diff_dataframe,
    split_feedback_dataset,
)


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


def test_check_fof(fof_datasets):
    ds1, ds2, tol_large, tol_small = fof_datasets
    _, _, ds_veri1 = split_feedback_dataset(ds1)
    _, _, ds_veri2 = split_feedback_dataset(ds2)
    df_veri1 = ds_veri1.to_dataframe().reset_index()
    df_veri2 = ds_veri2.to_dataframe().reset_index()
    _check(
        df_veri1["veri_data"],
        df_veri2["veri_data"],
        tol_large,
        tol_small,
        file_type="fof",
    )


def test_check_one_zero_stats(dataframes):
    """
    Test that a null value in ds1 causes failure,
    and that a variation within tolerance is accepted.
    """
    df1, df2, tol_large, tol_small = dataframes
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

    _, _, ds_veri1 = split_feedback_dataset(ds1)
    _, _, ds_veri2 = split_feedback_dataset(ds2)
    df_veri1 = ds_veri1.to_dataframe().reset_index()
    df_veri2 = ds_veri2.to_dataframe().reset_index()

    diff_df = compute_rel_diff_dataframe(df_veri1["veri_data"], df_veri2["veri_data"])
    diff_df = diff_df.to_frame()

    out, err, _ = check_variable(diff_df, tol_large)

    assert not out, f"Check with 0-value reference validated incorrectly:\n{err}"

    _, _, ds_veri2_copy = split_feedback_dataset(ds2_copy)
    ds_veri2_copy = ds_veri2_copy.copy(deep=True)
    ds_veri2_copy["veri_data"][2] = CHECK_THRESHOLD / 2
    _check(
        df_veri1["veri_data"],
        ds_veri2_copy["veri_data"],
        tol_large,
        tol_small,
        file_type="fof",
    )


def test_check_smalls_stats(dataframes):
    """
    Both values are close to 0 and should be accepted even though
    their relative difference is large.
    Close to 0 means < util.constants.CHECK_THRESHOLD.
    """

    df1, df2, tol_large, tol_small = dataframes
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

    _, _, ds_veri1 = split_feedback_dataset(ds1)
    _, _, ds_veri2 = split_feedback_dataset(ds2)
    df_veri1 = ds_veri1.to_dataframe().reset_index()
    df_veri2 = ds_veri2.to_dataframe().reset_index()

    _check(
        df_veri1["veri_data"],
        df_veri2["veri_data"],
        tol_large,
        tol_small,
        file_type="fof",
    )
