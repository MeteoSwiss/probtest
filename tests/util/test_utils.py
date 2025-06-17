"""
This module contains unit tests for the `utils.py` module.
"""

import pytest

from util.utils import get_seed_from_member_id, process_member_ids


def test_process_member_num_single_element():
    """
    Test case for a single element in the input list.
    It should generate a list from 1 to that number and convert each to a string.
    """
    input_data = [5]
    expected_output = [(5, "5")]
    assert process_member_ids(input_data) == expected_output


def test_process_member_num_multiple_elements():
    """
    Test case for multiple elements in the input list.
    It should convert each number to a string.
    """
    input_data = [2, 3, 4]
    expected_output = [(2, "2"), (3, "3"), (4, "4")]
    assert process_member_ids(input_data) == expected_output


def test_process_member_num_empty_list():
    """
    Test case for an empty input list.
    It should return an empty list.
    """
    input_data = []
    expected_output = []
    assert process_member_ids(input_data) == expected_output


def test_process_member_num_single_element_zero():
    """
    Test case for a single element in the input list being zero.
    It should return an empty list.
    """
    input_data = [0]
    expected_output = [(0, "0")]
    assert process_member_ids(input_data) == expected_output


def test_get_seed_from_member_id_invalid():
    """
    Test that the function raises a ValueError for invalid member numbers.
    """
    with pytest.raises(
        ValueError,
        match="Invalid member number",
    ):
        get_seed_from_member_id(0)

    with pytest.raises(
        ValueError,
        match="Invalid member number",
    ):
        get_seed_from_member_id(121)

    with pytest.raises(
        ValueError,
        match="Invalid member number",
    ):
        get_seed_from_member_id(-5)


def test_get_seed_from_member_id_unique_seeds():
    """
    Test that all returned seeds are unique.
    """
    seeds = [get_seed_from_member_id(i) for i in range(1, 121)]
    assert len(seeds) == len(set(seeds)), "Seeds are not unique!"
