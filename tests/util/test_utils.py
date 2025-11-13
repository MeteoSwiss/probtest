"""
This module contains unit tests for the `utils.py` module.
"""

import pytest

from util.utils import (
    get_seed_from_member_id,
    prepend_type_to_member_id,
    to_list,
    value_list,
)


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


def test_prepend_type_to_member_id():
    assert prepend_type_to_member_id("double", 3) == "double_3"


@pytest.fixture(name="none_value", scope="function")
def fixture_none_value():
    return


@pytest.fixture(name="string", scope="function")
def fixture_string():
    return "a,b,c,  ,e"


@pytest.fixture(name="simple_list", scope="function")
def fixture_list():
    return [1, 2, 3, 4]


def test_to_list_empty(none_value):
    """
    Test that the function returns an empty list if no value is given.
    """
    empty_return = to_list(none_value)
    assert empty_return == []


def test_to_list_full(string):
    """
    Test that if the function receives a string, it is split at each comma and
    unnecessary spaces are removed.
    """
    string_list_to_list = to_list(string)
    assert string_list_to_list == ["a", "b", "c", "e"]


def test_to_list_no_changes(simple_list):
    """
    Tests that if the function receives a value that is neither a string
    nor None, nothing changes.
    """
    to_list_no_changes = to_list(simple_list)
    assert to_list_no_changes == [1, 2, 3, 4]


def test_value_list():
    """
    Test that the correct placeholder is given according to the different cases.
    """
    placeholders = {"fof": False}
    assert value_list("fof", [1, 2, 3], placeholders) == [""]
    placeholders = {"fof": True}
    assert value_list("fof", [1, 2, 3], placeholders) == [1, 2, 3]


# def test_expand_zip():
#     """
#     Test that the zip is expanded correctly.
#     """
#     zipped = [
#         "test_{fof_type}.nc",
#         "test_{fof_type}_{member_id}.nc",
#         "test_{member_id}.nc",
#     ]
#     fof_type = ["AIREP", "PILOT"]
#     member_ids = [1, 2]
#     expanded_zip1 = expand_zip(zipped, fof_type, member_ids, member_type=None)
#     expanded_zip2 = expand_zip(zip(zipped), fof_type, member_ids, member_type=None)
#     assert expanded_zip1, expanded_zip2 == [
#         "test_AIREP.nc",
#         "test_PILOT.nc",
#         "test_AIREP_1.nc",
#         "test_AIREP_2.nc",
#         "test_PILOT_1.nc",
#         "test_PILOT_2.nc",
#         "test_1.nc",
#         "test_2.nc",
#     ]

#     expanded_zip3 = expand_zip(
#         "test_{fof_type}.nc", fof_type, member_ids, member_type=None
#     )
#     assert expanded_zip3 == ["test_AIREP.nc", "test_PILOT.nc"]
