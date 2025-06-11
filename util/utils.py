""""
This module provides utility functions for list and string operations, as well as
a function to generate seeds based on a member number for probtest.
"""

import re
from typing import List, Tuple


def unique_elements(inlist):
    unique = []
    for element in inlist:
        if element not in unique:
            unique.append(element)
    return unique


def first_idx_of(li, el):
    return li.index(el)


def last_idx_of(li, el):
    return len(li) - li[::-1].index(el) - 1


def numbers(s):
    "join all numbers from a string and return as an int (base 10)"
    return int("".join(re.findall(r"\d+", s)))


def get_seed_from_member_number(member_number: int) -> int:
    """
    Returns the seed corresponding to the given member number.
    """
    if member_number <= 0 or member_number > 120:
        raise ValueError(
            "Invalid member number: only values between 1 and 120 are valid."
        )

    seeds = [
        1031,
        1093,
        1171,
        1237,
        1303,
        1409,
        1471,
        1543,
        1607,
        1669,
        1753,
        1847,
        1913,
        1999,
        2081,
        2141,
        2239,
        2309,
        2381,
        2447,
        2549,
        2647,
        2699,
        2767,
        2843,
        2927,
        3019,
        3109,
        3203,
        3299,
        3359,
        3457,
        3529,
        3593,
        3673,
        3761,
        3847,
        3919,
        4007,
        4091,
        4159,
        4253,
        4339,
        4441,
        4517,
        4603,
        4679,
        4787,
        4877,
        4957,
        5021,
        5107,
        5209,
        5303,
        5407,
        5477,
        5557,
        5651,
        5717,
        5813,
        5867,
        5981,
        6073,
        6151,
        6247,
        6317,
        6379,
        6491,
        6581,
        6689,
        6779,
        6857,
        6949,
        7013,
        7121,
        7213,
        7309,
        7433,
        7517,
        7577,
        7669,
        7741,
        7853,
        7933,
        8053,
        8123,
        8231,
        8297,
        8419,
        8521,
        8609,
        8689,
        8753,
        8839,
        8941,
        9029,
        9133,
        9209,
        9311,
        9397,
        9463,
        9547,
        9649,
        9743,
        9829,
        9907,
        10037,
        10111,
        10193,
        10289,
        10369,
        10477,
        10597,
        10667,
        10771,
        10867,
        10973,
        11071,
        11161,
        11261,
    ]

    seed = seeds[member_number - 1]

    return seed


def process_member_ids(member_ids: List[int]) -> List[Tuple[int, str]]:
    """
    Converts a list of integers to a list of tuples with each integer and its
    string representation.
    """
    return [(m_num, str(m_num)) for m_num in member_ids]
