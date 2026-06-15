"""
This module contains unit tests for the `util/fof_utils.py` module.
"""

from unittest.mock import mock_open, patch

import numpy as np
import pytest
import xarray as xr

from util.fof_utils import (  # write_lines,
    clean_value,
    compare_arrays,
    compare_var_and_attr_ds,
    get_observation_variables,
    get_report_variables,
    replace_nan_with_sentinel_float64,
    split_feedback_dataset,
)
from util.log_handler import initialize_detailed_logger


@pytest.fixture(name="ds1", scope="function")
def fixture_ds1(sample_dataset_fof):
    return sample_dataset_fof


def test_get_report_variables(ds1):
    """
    Test that one gets the variable names of reports.
    """
    variables = get_report_variables(ds1)

    assert variables == [
        "lat",
        "lon",
        "statid",
        "time_nomi",
        "codetype",
        "l_body",
        "i_body",
    ]


def test_get_observation_variable(ds1):
    """
    Test that one gets the variable names of variables.
    """
    variables = get_observation_variables(ds1)

    assert variables == [
        "varno",
        "level",
        "veri_data",
        "obs",
        "bcor",
        "level_typ",
        "level_sig",
        "state",
        "flags",
        "check",
        "e_o",
        "qual",
        "plevel",
    ]


@pytest.fixture(name="ds_report", scope="function")
def fixture_sample_dataset_report(sample_dataset_fof):

    ds = sample_dataset_fof
    ds_report = ds[
        ["lat", "lon", "statid", "time_nomi", "codetype", "l_body", "i_body"]
    ].copy()
    ds_report = ds_report.assign_coords(y=np.arange(ds_report.sizes["d_hdr"]))

    ds_report.attrs = {"n_hdr": ds.attrs["n_hdr"], "n_body": ds.attrs["n_body"]}

    return ds_report


@pytest.fixture(name="ds_obs", scope="function")
def fixture_sample_dataset_obs(sample_dataset_fof):

    ds = sample_dataset_fof

    ds_observation = ds[
        [
            "lat",
            "lon",
            "statid",
            "time_nomi",
            "varno",
            "level",
            "state",
            "flags",
            "check",
            "e_o",
            "qual",
            "plevel",
        ]
    ].copy()

    ds_observation = ds_observation.assign_coords(
        y=np.arange(ds_observation.sizes["d_body"])
    )
    ds_observation.attrs = {"n_hdr": ds.attrs["n_hdr"], "n_body": ds.attrs["n_body"]}

    return ds_observation


@pytest.fixture(name="ds_veri", scope="function")
def fixture_sample_dataset_veri(sample_dataset_fof):

    ds = sample_dataset_fof

    ds_veri = ds[
        ["lat", "lon", "statid", "time_nomi", "varno", "level", "veri_data"]
    ].copy()

    ds_veri = ds_veri.assign_coords(y=np.arange(ds_veri.sizes["d_body"]))
    ds_veri.attrs = {"n_hdr": ds.attrs["n_hdr"], "n_body": ds.attrs["n_body"]}

    return ds_veri


def test_split_report(ds1, ds_report, ds_obs):
    """
    Test that the dataset is correctly split into reports, observations
    and veri data according to their dimensions.
    """
    reports, observations = split_feedback_dataset(ds1)

    assert reports == ds_report and observations == ds_obs


def test_split_radar_fof(sample_dataset_radar_fof):
    """
    Radar FOF files have d_hdr > n_hdr and d_body > n_body (NaN-padded tail), carry
    dlat/dlon per-observation positions, and a 2-D veri_data (d_veri, d_body).
    split_feedback_dataset must strip the padding by the n_* counts (NOT by NaN),
    keep real-region NaNs, and sort observations by dlat/dlon.
    """
    ds = sample_dataset_radar_fof
    n_hdr = ds.attrs["n_hdr"]
    n_body = ds.attrs["n_body"]

    reports, observations = split_feedback_dataset(ds)

    # padding stripped to the real counts on both dimensions
    assert reports.sizes["d_hdr"] == n_hdr
    assert observations.sizes["d_body"] == n_body

    # report padding rows are gone -> only the real station ids remain
    assert list(reports["statid"].values.astype(str)) == ["r1", "r2", "r3"]

    # 2-D veri_data (d_veri, d_body) survives the slice with the right body length
    assert observations["veri_data"].sizes["d_body"] == n_body

    # a real-region NaN in veri_data is preserved (only padding is removed, by count)
    assert np.isnan(observations["veri_data"].values).any()

    # observations are deterministically ordered by dlat (radar branch)
    dlat_vals = observations["dlat"].values
    assert list(dlat_vals) == sorted(dlat_vals), "observations not sorted by dlat"


def test_split_radar_fof_scattered_padding_raises(sample_dataset_radar_fof):
    """
    The strip-by-count assumes real observations are front-packed (padding is a
    NaN tail). A file with a real observation scattered into the padding region
    must fail loudly rather than be silently mis-stripped. dlat is the discriminator.
    """
    ds = sample_dataset_radar_fof.copy(deep=True)
    # swap a real dlat (index 5) with a padding NaN (index 7): now a non-NaN dlat
    # sits beyond n_body and a NaN sits inside the real region.
    dlat = ds["dlat"].values.copy()
    dlat[5], dlat[7] = dlat[7], dlat[5]
    ds["dlat"] = (("d_body",), dlat)
    with pytest.raises(ValueError, match="not front-packed"):
        split_feedback_dataset(ds)


def test_split_feedback_sort_is_deterministic(sample_dataset_radar_fof, sample_dataset_fof):
    """
    The pre-comparison sort must yield the SAME canonical order regardless of the
    order observations are stored in -- this is what establishes row correspondence
    between the two files being compared. Reorder observations within a header block
    (a valid permutation that preserves the i_body/l_body structure) and confirm the
    split output is identical. Covers the radar (dlat/dlon) and non-radar (lat/lon)
    sort branches.
    """
    cases = [
        # reverse the 3 observations of the last header block (body indices 3,4,5)
        (sample_dataset_radar_fof, [0, 1, 2, 5, 4, 3, 6, 7]),
        # reverse the 2 observations of the last header block (body indices 4,5)
        (sample_dataset_fof, [0, 1, 2, 3, 5, 4]),
    ]
    for ds, body_perm in cases:
        rep1, obs1 = split_feedback_dataset(ds.copy(deep=True))
        rep2, obs2 = split_feedback_dataset(ds.isel(d_body=body_perm))
        xr.testing.assert_identical(obs1, obs2)
        xr.testing.assert_identical(rep1, rep2)


@pytest.fixture(name="arr1", scope="function")
def fixture_arr1():
    return np.array([1.0, 5.0, 3.0, 4.0, 7.0], dtype=np.float32)


@pytest.fixture(name="arr2", scope="function")
def fixture_arr2():
    return np.array([1.0, 5.0, 3.0, 4.0, 7.0], dtype=np.float32)


@pytest.fixture(name="arr3", scope="function")
def fixture_arr3():
    return np.array([23.0, 5.0, 3.0, 22.0, 7.0], dtype=np.float32)


@pytest.fixture(name="arr1_nan", scope="function")
def fixture_arr1_nan():
    return np.array([1.0, 5.0, 3.0, np.nan, 7.0], dtype=np.float32)


@pytest.fixture(name="arr2_nan", scope="function")
def fixture_arr2_nan():
    return np.array([1.0, 5.0, 3.0, np.nan, 7.0], dtype=np.float32)


def test_compare_array_equal(arr1, arr2, arr1_nan, arr2_nan):
    """
    Test that two arrays are considered equal if:
    - they have the same content
    - they have nan values in the same positions
    """
    total, equal, diff = compare_arrays(arr1, arr2, "var_name")
    total_nan, equal_nan, diff_nan = compare_arrays(arr1_nan, arr2_nan, "var_name")

    assert (total, equal, total_nan, equal_nan, diff.size, diff_nan.size) == (
        5,
        5,
        5,
        5,
        0,
        0,
    )


def test_compare_array_diff(arr1, arr3):
    """
    Test that if I compare two different arrays I get the number of total and equal
    vales and the number of the position where values are different."""
    total, equal, diff = compare_arrays(arr1, arr3, "var_name")

    assert (total, equal, diff.tolist()) == (5, 3, [0, 3])


@pytest.fixture(name="arr_nan", scope="function")
def fixture_arr():
    return np.array([1.0, np.nan, 3.0, 4.0, np.nan], dtype=np.float32)


def test_fill_nans_for_float32_nan(arr_nan):
    """
    Test that if an array containing nan is given, these values are replaced
    by -999999.0.
    """
    array = replace_nan_with_sentinel_float64(arr_nan)
    expected = np.array([1.0, -999999.0, 3.0, 4.0, -999999.0], dtype=np.float64)
    assert np.array_equal(array, expected)


def test_fill_nans_for_float32(arr1):
    """
    Test that if an array without nan is given, the output of the function
    is the same as the input.
    """
    array = replace_nan_with_sentinel_float64(arr1)
    assert np.array_equal(array, arr1)


@pytest.fixture(name="arr_with_spaces", scope="function")
def fixture_arr_with_spaces():
    return np.array([1.0, "ABF   '", 3.0, 4.0, b"ABF   '"], dtype=object)


def test_clean_value(arr_with_spaces):
    """
    Test that if values with unnecessary spaces are provided,
    they are removed and cleaned up.
    """
    cleaned = np.array([clean_value(x) for x in arr_with_spaces])
    expected = np.array(["1.0", "ABF", "3.0", "4.0", "ABF"])
    np.testing.assert_array_equal(cleaned, expected)


@pytest.fixture(name="ds2", scope="function")
def fixture_sample_dataset_2(sample_dataset_fof):
    """
    Sample fof dataset slightly modified.
    """
    data = sample_dataset_fof.copy(deep=True)

    # only last value of codetype slightly changes
    data["codetype"] = (("d_hdr",), [5, 5, 5, 5, 3])
    return data


def test_compare_var_and_attr_ds(ds1, ds2):
    """
    Test that, given two datasets, returns the number of elements in which
    the variables are the same and in which they differ.
    """
    with patch("builtins.open", mock_open()):

        detailed_logger = initialize_detailed_logger(
            "DETAILS", log_level="DEBUG", log_file="test_log.log"
        )

        total1, equal1 = compare_var_and_attr_ds(ds1, ds2, detailed_logger)

        assert (total1, equal1) == (103, 102)


@pytest.fixture(name="ds3")
def fixture_sample_dataset_3(sample_dataset_fof):

    ds = sample_dataset_fof.isel(d_body=slice(0, 5))
    ds["codetype"] = (("d_hdr",), [5, 5, 5, 5, 3])
    ds.attrs["n_body"] = 5
    ds.attrs["plevel"] = np.array([0.374, 0.950, 0.731, 0.598, 0.156])

    return ds
