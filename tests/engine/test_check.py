"""
This module contains unittests for verifying the behavior of data frame operations
related to the PROBTEST suite. It ensures the correctness of relative difference
calculations and the checking of variable values against specified tolerances.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from util.constants import CHECK_THRESHOLD
from util.dataframe_ops import (
    check_intersection,
    check_variable,
    compute_rel_diff_dataframe, split_feedback_dataset
)


@pytest.fixture
def dataframes():
    """Crea i dataframe e le tolleranze di riferimento."""
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

    tol_large = pd.DataFrame(np.ones((2, 12)) * 0.21, index=["var_1", "var_2"], columns=columns)
    tol_small = pd.DataFrame(np.ones((2, 12)) * 0.06, index=["var_1", "var_2"], columns=columns)

    return df1, df2, tol_large, tol_small


@pytest.fixture
def fof_datasets():


    n_hdr_size = 5
    n_body_size = 6

    # === Coordinate e variabili header ===
    lat = np.array([1, 3, 2, 4, 5])
    lon = np.array([5, 9, 7, 8, 3])
    varno = np.array([3, 3, 4, 4, 4, 4])
    statid = ["a", "b", "c", "d", "e"]
    time_nomi = [0, 30, 0, 30, 60]
    codetype = [5, 5, 5, 5, 5]
    lbody = np.array([1, 1, 1, 1, 2])
    ibody = np.arange(1, n_hdr_size + 1)
    level = np.array([1000, 950, 900, 850, 800, 750])

    # === Variabili body ===
    veri_data = np.array([45, 34, 45, 56, 67, 78])
    obs = np.array([0.374, 0.950, 0.731, 0.598, 0.156, 0.155])
    bcor = np.array([0.058, 0.866, 0.601, 0.708, 0.020, 0.969])
    level_typ = np.array([0.832, 0.212, 0.181, 0.183, 0.304, 0.524])
    level_sig = np.array([0.431, 0.291, 0.611, 0.139, 0.292, 0.366])
    state = np.array([0.456, 0.785, 0.199, 0.514, 0.592, 0.046])
    flags = np.array([0.607, 0.170, 0.065, 0.948, 0.965, 0.808])
    check = np.array([0.304, 0.097, 0.684, 0.440, 0.122, 0.495])
    e_o = np.array([0.034, 0.909, 0.258, 0.662, 0.311, 0.520])
    qual = np.array([0.796, 0.509, 0.810, 0.163, 0.425, 0.138])
    plevel = np.array([0.801, 0.406, 0.077, 0.847, 0.320, 0.755])

    # === Dataset 1 (riferimento) ===
    ds1 = xr.Dataset(
        {
            "lat": (("d_hdr",), lat),
            "lon": (("d_hdr",), lon),
            "varno": (("d_body",), varno),
            "statid": (("d_hdr",), statid),
            "time_nomi": (("d_hdr",), time_nomi),
            "codetype": (("d_hdr",), codetype),
            "level": (("d_body",), level),
            "l_body": (("d_hdr",), lbody),
            "i_body": (("d_hdr",), ibody),
            "veri_data": (("d_body",), veri_data),
            "obs": (("d_body",), obs),
            "bcor": (("d_body",), bcor),
            "level_typ": (("d_body",), level_typ),
            "level_sig": (("d_body",), level_sig),
            "state": (("d_body",), state),
            "flags": (("d_body",), flags),
            "check": (("d_body",), check),
            "e_o": (("d_body",), e_o),
            "qual": (("d_body",), qual),
            "plevel": (("d_body",), plevel),
        },
        coords={
            "d_hdr": np.arange(n_hdr_size),
            "d_body": np.arange(n_body_size),
        },
        attrs={"n_hdr": n_hdr_size, "n_body": n_body_size},
    )

    # === Dataset 2 (variazione leggera per test comparativo) ===
    ds2 = ds1.copy(deep=True)
    ds2["veri_data"] = (("d_body",), veri_data *1.55)  # +5% differenza
   # ds2["obs"] = (("d_body",), obs * 0.95)  # -5% differenza, ad esempio

    # === Dataset di tolleranza ===
    tol_large = pd.DataFrame({"veri_data": np.full(n_body_size, 35)})

    # Tolleranza "piccola"
    tol_small = pd.DataFrame({"veri_data": np.full(n_body_size, 0.06)})

    return ds1, ds2, tol_large, tol_small


def _check(df1, df2, tol_large, tol_small,file_type):
    """Helper: esegue il controllo delle tolleranze."""

    diff_df = compute_rel_diff_dataframe(df1, df2)

    if file_type == "stats":
        diff_df = diff_df.groupby(["variable"]).max()
    if file_type == "fof":
        diff_df = diff_df.to_frame()

    out1, err1, _ = check_variable(diff_df, tol_large)
    out2, err2, _ = check_variable(diff_df, tol_small)


    assert out1, f"Check con tolleranza larga non ha validato:\n{err1}"
    assert not out2, f"Check con tolleranza stretta ha validato erroneamente:\n{err2}"


def test_check(dataframes):
    df1, df2, tol_large, tol_small = dataframes
    _check(df1, df2, tol_large, tol_small, file_type = "stats")



def test_check_fof(fof_datasets):
    ds1, ds2, tol_large, tol_small = fof_datasets
    _, _, ds_veri1 = split_feedback_dataset(ds1)
    _, _, ds_veri2 = split_feedback_dataset(ds2)
    df_veri1 = ds_veri1.to_dataframe().reset_index()
    df_veri2 = ds_veri2.to_dataframe().reset_index()
    _check(df_veri1["veri_data"], df_veri2["veri_data"], tol_large, tol_small, file_type="fof")


def test_check_one_zero(dataframes):
    df1, df2, tol_large, tol_small = dataframes
    df1 = df1.copy()
    df1.loc[("NetCDF:*atm_3d*.nc", "var_1", 2), (0, "max")] = 0

    diff_df = compute_rel_diff_dataframe(df1, df2)
    diff_df = diff_df.groupby(["variable"]).max()
    out, err, _ = check_variable(diff_df, tol_large)

    assert not out, f"Check con 0-value reference ha validato erroneamente:\n{err}"

    df2 = df2.copy()
    df2.loc[("NetCDF:*atm_3d*.nc", "var_1", 2), (0, "max")] = CHECK_THRESHOLD / 2
    _check(df1, df2, tol_large, tol_small, file_type = "stats")


def test_check_smalls(dataframes):

    df1, df2, tol_large, tol_small = dataframes
    df1 = df1.copy()
    df2 = df2.copy()

    df1.loc["var_1", (2, "min")] = CHECK_THRESHOLD * 1e-5
    df2.loc["var_1", (2, "min")] = -CHECK_THRESHOLD / 2

    _check(df1, df2, tol_large, tol_small, file_type = "stats")


def test_check_smalls_fof(fof_datasets):
    """Test analogo a test_check_smalls ma sui dataset FOF."""
    ds1, ds2, tol_large, tol_small = fof_datasets

    # Copie modificabili
    ds1 = ds1.copy(deep=True)
    ds2 = ds2.copy(deep=True)

    # Imposta valori piccoli su veri_data
    ds1["veri_data"][2] = CHECK_THRESHOLD * 1e-5
    ds2["veri_data"][2] = -CHECK_THRESHOLD / 2

    _, _, ds_veri1 = split_feedback_dataset(ds1)
    _, _, ds_veri2 = split_feedback_dataset(ds2)
    df_veri1 = ds_veri1.to_dataframe().reset_index()
    df_veri2 = ds_veri2.to_dataframe().reset_index()


    # Esegui il check come con i dataframe
    _check(df_veri1["veri_data"], df_veri2["veri_data"], tol_large, tol_small, file_type="fof")



def test_no_intersection(dataframes):
    df1, df2, _, _ = dataframes
    df1 = df1.rename(index={"var_1": "var_3", "var_2": "var_4"})
    skip_test, _, _ = check_intersection(df1, df2)

    assert skip_test != 0, (
        "Nessuna intersezione di variabili tra reference e test, "
        "ma il test non è fallito."
    )

# def test_no_intersection_fof(fof_datasets):
#     df1, df2, _, _ = fof_datasets

#     _, _, ds_veri1 = split_feedback_dataset(df1)
#     _, _, ds_veri2 = split_feedback_dataset(df2)
#     df_veri1 = ds_veri1.to_dataframe().reset_index()
#     df_veri2 = ds_veri2.to_dataframe().reset_index()

#     df_veri1 = df_veri1.rename(columns={"veri_data": "var_3"})
#     # df_veri1 = df_veri1["var_3"]

#     # df_veri2 = df_veri2["veri_data"]

#     skip_test, _, _ = check_intersection(df_veri1, df_veri2)

#     assert skip_test != 0, (
#         "Nessuna intersezione di variabili tra reference e test, "
#         "ma il test non è fallito."
#     )

# def test_missing_variables(dataframes):
#     df1, df2, _, _ = dataframes
#     df1 = df1.drop("var_1", level="variable")

#     expected_warning_1 = (
#         "WARNING: The following variables are in the "
#         "test case but not in the reference case and therefore not tested: var_1"
#     )
#     expected_warning_2 = (
#         "WARNING: The following variables are in the "
#         "reference case but not in the test case and therefore not tested: var_1"
#     )

#     with pytest.warns(UserWarning, match=expected_warning_1):
#         check_intersection(df1, df2)

#     with pytest.warns(UserWarning, match=expected_warning_2):
#         check_intersection(df2, df1)


# def test_check_swapped(dataframes):
#     """Verifica che i controlli siano simmetrici (df1 vs df2 o viceversa)."""
#     df1, df2, tol_large, tol_small = dataframes
#     _check(df2, df1, tol_large, tol_small)
