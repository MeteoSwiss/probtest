"""
This module contains unit tests for the `model_output_parser` module.
"""

from unittest.mock import MagicMock

import pytest

from util.model_output_parser import get_ds


@pytest.fixture(name="mock_ds_grib")
def fixture_mock_ds_grib():
    """
    Fixture that creates a mock GRIB object to simulate different cases of data
    selection and exceptions that `get_ds` needs to handle.
    """
    ds_grib = MagicMock()

    # Simulating the successful selection and conversion to xarray
    ds_grib.sel.return_value.to_xarray.return_value = "valid_xarray"

    # Simulating the metadata retrieval (unique values)
    ds_grib.sel.return_value.metadata.side_effect = [
        ["forecast"],  # stepType
        [100],  # numberOfPoints
        ["hours"],  # stepUnits
        ["analysis"],  # dataType
        ["regular_ll"],  # gridType
    ]

    return ds_grib


def test_get_ds_success(mock_ds_grib):
    """
    Test case where get_ds successfully retrieves the dataset on the first attempt.
    """
    pid = 1
    lev = "surface"

    result = get_ds(mock_ds_grib, pid, lev)

    # Ensure the dataset is selected once
    mock_ds_grib.sel.assert_called_once_with(paramId=pid, typeOfLevel=lev)

    # The result should contain the mocked xarray dataset
    assert result == ["valid_xarray"]


@pytest.mark.parametrize(
    "to_xarray_return_value, expected_result",
    [
        ((KeyError(), "valid_stepType_xarray"), ["valid_stepType_xarray"]),
        (
            (KeyError(), KeyError(), "valid_numberOfPoints_xarray"),
            ["valid_numberOfPoints_xarray"],
        ),
        (
            (KeyError(), KeyError(), KeyError(), "valid_stepUnits_xarray"),
            ["valid_stepUnits_xarray"],
        ),
        (
            (KeyError(), KeyError(), KeyError(), KeyError(), "valid_dataType_xarray"),
            ["valid_dataType_xarray"],
        ),
        (
            (
                KeyError(),
                KeyError(),
                KeyError(),
                KeyError(),
                KeyError(),
                "valid_gridType_xarray",
            ),
            ["valid_gridType_xarray"],
        ),
    ],
)
def test_get_ds_recursive_selection(
    mock_ds_grib, to_xarray_return_value, expected_result
):
    """
    Test case where get_ds recursively selects the dataset by metadata fields.
    """
    pid = 1
    lev = "surface"

    mock_ds_grib.sel.return_value.to_xarray.side_effect = to_xarray_return_value

    result = get_ds(mock_ds_grib, pid, lev)

    # Ensure the recursive logic is triggered by calling sel multiple times
    assert mock_ds_grib.sel.call_count >= len(to_xarray_return_value)

    # The result should contain the mocked xarray dataset
    assert result == expected_result


def test_get_ds_keyerror_handling(caplog, mock_ds_grib):
    """
    Test case where get_ds fails to retrieve data and handles multiple KeyErrors.
    """
    pid = 1
    lev = "surface"

    # Simulate KeyErrors for all attempts to select datasets
    mock_ds_grib.sel.return_value.to_xarray.side_effect = KeyError()

    result = get_ds(mock_ds_grib, pid, lev)

    # Assert that the warning was logged
    assert "GRIB file of level surface and paramId 1 cannot be read." in caplog.text
    assert not result
