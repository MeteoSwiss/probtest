import pathlib
import re

import numpy as np


def unique_elements(inlist):
    unique = []
    for element in inlist:
        if element not in unique:
            unique.append(element)
    return unique


def first_idx_of(list, element):
    return list.index(element)


def last_idx_of(list, element):
    return len(list) - list[::-1].index(element) - 1


def get_seed_from_member_num(member_num, use_64_bits=True):
    # Ensure member_num is a valid positive integer
    if member_num <= 0 or member_num > 100:
        raise ValueError("Invalid member number 'member_num'.")

    path_seeds = pathlib.Path(__file__).parent.parent.absolute() / "seeds.txt"
    with open(path_seeds, "r") as file:
        # Read the specified line and extract the number
        for i, line in enumerate(file, 1):
            if i == member_num:
                seed = int(line.strip())
    if not use_64_bits:
        seed = np.uint32(seed & 0xFFFFFFFF)

    return seed


def numbers(s):
    "join all numbers from a string and return as an int (base 10)"
    return int("".join(re.findall(r"\d+", s)))
