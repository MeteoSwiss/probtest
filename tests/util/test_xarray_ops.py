"""
This module contains unit tests for the "xarray_ops.py" module
"""

import sys

import numpy as np
import pytest
import xarray as xr

from util.xarray_ops import statistics_over_horizontal_dim


@pytest.fixture(name="sample_data", scope="module")
def fixture_sample_data():
    # Create easy-to-use file for testing
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    da = xr.DataArray(
        data,
        dims=("x", "y"),
        coords={"x": [10, 20], "y": ["a", "b", "c"]},
        name="test_var",
    )
    return da


ALL_STATS = ["mean", "max", "min", "sum", "std", "var", "median", "prod"]
ALL_DIMS = ["x", "y", "x:y"]


@pytest.mark.parametrize("stat", ALL_STATS)
@pytest.mark.parametrize("dims", ALL_DIMS)
def test_statistics_against_manual(sample_data, stat, dims):
    result = statistics_over_horizontal_dim(sample_data, [dims], [stat])
    values = [r.values for r in result]
    numbers = values[0].tolist()

    data = sample_data

    expected_values = {
        "mean": {"x": [2.5, 3.5, 4.5], "y": [2, 5], "x:y": [3.5]},
        "max": {"x": [4, 5, 6], "y": [3, 6], "x:y": [6]},
        "min": {"x": [1, 2, 3], "y": [1, 4], "x:y": [1]},
        "sum": {"x": [5, 7, 9], "y": [6, 15], "x:y": [21]},
        "std": {
            "x": [1.5, 1.5, 1.5],
            "y": [0.816496580927726, 0.816496580927726],
            "x:y": [np.std(data)],
        },
        "var": {
            "x": [2.25, 2.25, 2.25],
            "y": [0.6666666666666666, 0.6666666666666666],
            "x:y": [np.var(data)],
        },
        "median": {"x": [2.5, 3.5, 4.5], "y": [2, 5], "x:y": [3.5]},
        "prod": {"x": [4, 10, 18], "y": [6, 120], "x:y": [720]},
    }

    expected = expected_values[stat][dims]

    np.testing.assert_allclose(numbers, expected)


def test_statistics_with_fill(sample_data):
    """
    Test that the fill value mask is applied correctly
    """
    # Add a fake _FillValue attribute and inject fill values
    data = sample_data.copy()
    data.attrs["_FillValue"] = -999
    data = data.where(~np.isnan(data), other=-999)

    result = statistics_over_horizontal_dim(
        data, [data.dims[0]], ["mean", "sum"], fill_value_key="_FillValue"
    )

    expected_mean = data.where(data != -999).mean(dim=data.dims[0], skipna=True)
    expected_sum = data.where(data != -999).sum(dim=data.dims[0], skipna=True)

    assert np.allclose(result[0], expected_mean)
    assert np.allclose(result[1], expected_sum)


def test_no_matching_dimension(sample_data, monkeypatch):
    """
    Test that sys.exit(1) is called in case the dimension name doesn't exist
    """

    monkeypatch.setattr(
        sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code))
    )
    with pytest.raises(SystemExit):
        statistics_over_horizontal_dim(sample_data, ["nonexistent_dim"], ["mean"])


def test_invalid_statistic(sample_data):
    """
    Test that statistics that does not exist can not be computed
    """

    dim = sample_data.dims[0]
    with pytest.raises(AttributeError):
        statistics_over_horizontal_dim(sample_data, [dim], ["not_a_stat"])
