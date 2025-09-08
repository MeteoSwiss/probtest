"""
This module contains unit tests for the "xarray_ops.py" module
"""
import pytest
import xarray as xr
import numpy as np
import sys
import inspect

from util.xarray_ops import statistics_over_horizontal_dim

@pytest.fixture(scope="module")
def sample_data():
    # Create easy-to-use file for testing
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    da = xr.DataArray(
        data,
        dims=("x", "y"),
        coords={"x": [10, 20], "y": ["a", "b", "c"]},
        name="test_var"
    )
    return da


# Manual list of supported and stable statistics in xarray
ALL_STATS = [
    "mean",
    "max",
    "min",
    "sum",
    "std",
    "var",
    "median",
    "prod",
]

@pytest.mark.parametrize("stat", ALL_STATS)
def test_statistics_no_fill(sample_data, stat):
    """
    Test that for every dimension of the xarray the statistics are computed correctly"
    """
    for dim in sample_data.dims:
        result = statistics_over_horizontal_dim(sample_data, [dim], [stat])
        expected = getattr(sample_data, stat)(dim=dim, skipna=False)
        assert np.allclose(result[0], expected)



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




def test_statistics_multiple_dims(sample_data):

    """
    Test that statistics can be computed across multiple dimensions at once
    """

    if len(sample_data.dims) < 2:
        pytest.skip("Dataset does not have at least 2 dims to combine")
    hor_dims = ":".join(sample_data.dims[:2])
    result = statistics_over_horizontal_dim(sample_data, [hor_dims], ["max", "var"])


    expected_max = sample_data.max(dim=sample_data.dims[:2], skipna=False)
    expected_var = sample_data.var(dim=sample_data.dims[:2], skipna=False)


    assert np.allclose(result[0], expected_max)
    assert np.allclose(result[1], expected_var)



def test_no_matching_dimension(sample_data, monkeypatch):
    """
    Test that sys.exit(1) is called in case the dimension name doesn't exist
    """

    monkeypatch.setattr(sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit):
        statistics_over_horizontal_dim(sample_data, ["nonexistent_dim"], ["mean"])




def test_invalid_statistic(sample_data):
    """
    Test that statistics that does not exist can not be computed
    """

    dim = sample_data.dims[0]
    with pytest.raises(AttributeError):
        statistics_over_horizontal_dim(sample_data, [dim], ["not_a_stat"])
