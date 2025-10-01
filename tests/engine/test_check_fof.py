"""
This module contains unit tests for the `engine/check_fof.py` module.
"""

import numpy as np
import pytest
import xarray as xr

from engine.check_fof import compare_arrays


@pytest.fixture(name="nc1", scope="function")
def fixture_nc1():
    """
    Creation of a sample dataset
    """
    data = xr.Dataset(
        {
            "lat": (("y", "x"), np.array([[10, 20], [30, 40]])),
            "lon": (("y", "x"), np.array([[100, 110], [120, 130]])),
            "statid": (("y", "x"), np.array([["a", "b"], ["c", "d"]])),
        }
    )
    return data


@pytest.fixture(name="nc2", scope="function")
def fixture_nc2():
    """
    Creation of a sample dataset identical to nc1
    """
    data = xr.Dataset(
        {
            "lat": (("y", "x"), np.array([[10, 20], [30, 40]])),
            "lon": (("y", "x"), np.array([[100, 110], [120, 130]])),
            "statid": (("y", "x"), np.array([["a", "b"], ["c", "d"]])),
        }
    )
    return data


@pytest.fixture(name="nc3", scope="function")
def fixture_nc3():
    """
    Creation of a sample dataset with reversed order
    """
    data = xr.Dataset(
        {
            "lat": (("y", "x"), np.array([[30, 40], [10, 20]])),
            "lon": (("y", "x"), np.array([[120, 130], [100, 110]])),
            "statid": (("y", "x"), np.array([["c", "d"], ["a", "b"]])),
        }
    )
    return data


# def test_compute_same_hash_identical_ds(nc1, nc2):
#     """
#     Test that two identical datasets provide the same hash
#     """
#     hash1 = compute_hash_for_vars_and_attrs(nc1)
#     hash2 = compute_hash_for_vars_and_attrs(nc2)

#     assert hash1 == hash2


# def test_compute_same_hash_reversed_order(nc1, nc3):
#     """
#     Test that datasets with same values, but reversed order provide
#     same hash
#     """
#     hash1 = compute_hash_for_vars_and_attrs(nc1)
#     hash3 = compute_hash_for_vars_and_attrs(nc3)

#     assert hash1 == hash3


@pytest.fixture(name="arr1", scope="function")
def fixture_arr1():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture(name="arr2", scope="function")
def fixture_arr2():
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture(name="arr3", scope="function")
def fixture_arr3():
    return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])


@pytest.fixture(name="arr4", scope="function")
def fixture_arr4():
    return np.array([1.0, 2.0, 3.0, 6.0, 7.0])


def test_compare_arrays_equal(arr1, arr2):
    """
    Test that, for identical arrays, the size of the array
    and the number of identical values coincide
    """
    t, e = compare_arrays(arr1, arr2, var_name=None)
    assert t == e == 5


def test_compare_arrays_nan(arr3):
    """
    Test that, for nan arrays, the size of the array
    and the number of identical values coincide
    """
    arr4 = arr3.copy()
    t, e = compare_arrays(arr3, arr4, var_name=None)
    assert t == e == 5


def test_compare_arrays_different(arr1, arr4):
    """
    Test that, for different arrays, the size of the array
    and the number of identical values do not coincide
    """
    t, e = compare_arrays(arr1, arr4, var_name=None)
    assert t == 5
    assert e == 3
