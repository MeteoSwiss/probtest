"""
This module contains unit tests for the `utils.py` module.
"""

import pytest

from util.utils import get_seed_from_member_id


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
