"""
This module contains unit tests for the `utils.py` module.
"""

import pytest

from util.utils import (
    expand_fof,
    expand_members,
    get_seed_from_member_id,
    prepend_type_to_member_id,
    to_list,
    validate_single_stats_file,
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


def test_expand_fof():
    """
    Test that the fof zip is expanded correctly using the fof_type names.
    {member_id} is left untouched.
    """
    zipped = [
        "test_{fof_type}.nc",
        "test_{fof_type}_{member_id}.nc",
        "test_{member_id}.nc",
    ]

    fof_type = ["AIREP", "PILOT"]
    expanded_zip1 = expand_fof(zipped, fof_type)
    expanded_zip2 = expand_fof(zip(zipped), fof_type)
    assert expanded_zip1, expanded_zip2 == [
        "test_AIREP.nc",
        "test_PILOT.nc",
        "test_AIREP_{member_id}.nc",
        "test_PILOT_{member_id}.nc",
        "test_{member_id}.nc",
    ]


def test_expand_members():
    """
    Test that the members zip is expanded correctly using the members_ids values.
    {member_type} is left untouched.
    """
    zipped = [
        "test_{member_type}.nc",
        "stats_{member_type}_{member_id}.nc",
        "test_{member_id}.nc",
    ]
    member_ids = [1, 2]
    expanded_zip1 = expand_members(zipped, member_ids, member_type=None)
    expanded_zip2 = expand_members(zip(zipped), member_ids, member_type=None)
    assert expanded_zip1, expanded_zip2 == [
        "test_{member_type}.nc",
        "stats_{member_type}_1.nc",
        "stats_{member_type}_2.nc",
        "test_1.nc",
        "test_2.nc",
    ]


def test_expand_members_member_type():
    """
    Test that the members zip is expanded correctly using the members_type name.
    """
    zipped = ["fof{member_type}.nc"]

    expanded_zip1 = expand_members(zipped, member_type="dp")
    assert expanded_zip1 == ["fof_member_id_dp_.nc"]


def test_expand_members_full_fof():
    """
    Test that the members zip is expanded correctly using the member_id values
    in case member_type is not defined but the placeholders is given.
    """
    zipped = ["fof{member_type}{member_id}.nc"]

    expanded_zip1 = expand_members(zipped, member_ids=[1, 2], member_type=None)
    assert expanded_zip1 == ["fof_member_id_1.nc", "fof_member_id_2.nc"]

    zipped = ["fof_{member_id}.nc"]
    expanded_zip2 = expand_members(zipped, member_ids=["ref"])
    assert expanded_zip2 == ["fof_ref.nc"]


def test_single_valid_stats_file():

    stats_file = "stats.csv"
    errors = []
    result = validate_single_stats_file([stats_file], "ensemble", errors)

    assert result == stats_file
    assert not errors


def test_single_invalid_file_type():
    # create a FOF file
    fof_file = "fofDummy.nc"

    errors = []
    result = validate_single_stats_file([fof_file], "tolerance", errors)

    assert result is None
    assert len(errors) == 1


def test_zero_files():
    errors = []
    result = validate_single_stats_file([], "ensemble", errors)

    assert result is None
    assert len(errors) == 1
    assert "Expected exactly one ensemble file" in errors[0]


def test_multiple_files():
    errors = []
    result = validate_single_stats_file(
        ["stats1.csv", "stats2.csv"], "tolerance", errors
    )

    assert result is None
    assert len(errors) == 1
