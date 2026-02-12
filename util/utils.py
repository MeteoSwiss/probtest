"""
This module provides utility functions for list and string operations, as well as
a function to get fixed seeds based on the ensemble member number.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


def unique_elements(inlist):
    unique = []
    for element in inlist:
        if element not in unique:
            unique.append(element)
    return unique


def prepend_type_to_member_id(member_type, member_id):
    return (member_type + "_" + str(member_id)) if member_type else str(member_id)


def first_idx_of(li, el):
    return li.index(el)


def last_idx_of(li, el):
    return len(li) - li[::-1].index(el) - 1


def numbers(s):
    "join all numbers from a string and return as an int (base 10)"
    return int("".join(re.findall(r"\d+", s)))


def get_seed_from_member_id(member_id: int) -> int:
    """
    Returns the seed corresponding to the given member number.
    """
    if member_id <= 0 or member_id > 120:
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

    seed = seeds[member_id - 1]

    return seed


def to_list(value):
    """
    This function ensures that whatever value is passed to expand_zip,
    one always gets something iterable and consistent.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def value_list(placeholder, values, placeholders):
    """
    Decide which values to use for a placeholder.
    - If the placeholder is not present, returns [""]
    - If placeholder is present but no values provided, returns [None]
    """
    if not placeholders.get(placeholder, False):
        return [""]
    return values or [None]


def normalize_zipped(zipped):
    """Ensure zipped is a list of tuples."""
    if isinstance(zipped, zip):
        return list(zipped)
    if isinstance(zipped, (list, tuple)):
        return [(z,) if not isinstance(z, tuple) else z for z in zipped]
    return [(zipped,)]


def expand_fof(zipped, fof_type=None):
    """
    Expands a list of tuples (zip) by replacing the placeholder {fof_type}
    with the values provided. Other placeholders (e.g. {member_id}, {member_type})
    are left unchanged.
    """

    zipped = normalize_zipped(zipped)
    expanded = []
    fof_values = value_list("fof_type", to_list(fof_type), {"fof_type": True})

    for items in zipped:
        if not any("{fof_type}" in item for item in items):
            expanded.append(items[0] if len(items) == 1 else tuple(items))
            continue

        for fof_val in fof_values:
            safe_dict = defaultdict(
                lambda k=None: f"{{{k}}}",
                {
                    "fof_type": fof_val,
                    "member_id": "{member_id}",
                    "member_type": "{member_type}",
                },
            )
            formatted = [item.format_map(safe_dict) for item in items]
            expanded.append(formatted[0] if len(formatted) == 1 else tuple(formatted))

    return expanded


def expand_members(zipped, member_ids=None, member_type=None):
    """
    Expands a list of tuples (zip) by replacing the placeholders
    {member_id} and {member_type} with the values provided.
    If a placeholder is present but no corresponding value is given,
    the placeholder is left unchanged.
    """

    member_list = to_list(member_ids)
    member_type_list = to_list(member_type)
    zipped = normalize_zipped(zipped)

    expanded = []

    for items in zipped:
        try:
            file_type = (FileInfo(items[0])).file_type
        except (TypeError, ValueError):
            file_type = None

        placeholders = {
            key: any(f"{{{key}}}" in str(item) for item in items)
            for key in ("member_id", "member_type")
        }

        member_values = value_list("member_id", to_list(member_ids), placeholders)

        if file_type is FileType.FOF:
            if member_type_list:
                member_type_list = [
                    f"_member_id_{mtype}_" for mtype in member_type_list
                ]
            elif not (member_list and all(m == "ref" for m in member_list)):
                member_type_list = ["_member_id_"]
            else:
                member_type_list = []

        member_values_expanded = []

        if file_type is FileType.STATS and member_type_list:
            member_values_expanded = (
                [
                    f"{m_type}_{m_id}"
                    for m_type in member_type_list
                    for m_id in member_values
                ]
                if member_values
                else member_type_list.copy()
            )
        else:
            member_values_expanded = member_values.copy()

        for member_val in member_values_expanded:
            formatted = [
                item.format(
                    member_id=member_val or "{member_id}",
                    member_type=(
                        member_type_list[0] if member_type_list else "{member_type}"
                    ),
                )
                for item in items
            ]
            expanded.append(formatted[0] if len(formatted) == 1 else tuple(formatted))

    return expanded


class FileType(Enum):
    """
    Class that memorizes the distinction between file types
    """

    FOF = "fof"
    STATS = "csv"


@dataclass
class FileInfo:
    """
    Class that memorize the path and the type of a file.
    """

    path: str
    file_type: Optional[FileType] = None

    def __post_init__(self):

        name = self.path.lower()

        if "fof" in name:
            self.file_type = FileType.FOF
            return
        if "csv" in name or "stats" in name:
            self.file_type = FileType.STATS
            return FileType.STATS

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if "," in first_line or ";" in first_line:
                    self.file_type = FileType.STATS
                    return
        except (OSError, FileNotFoundError):
            pass

        raise ValueError(f"Unknown file type for '{self.path}'")


def validate_single_stats_file(
    files: List[str],
    role: str,
    errors: List[str],
) -> Optional[str]:
    """
    Validate that `files` contains exactly one stats file.

    Appends error messages into `errors`.
    Returns the file name or None.
    """
    if len(files) == 1:
        file_name = files[0]
        file_info = FileInfo(file_name)

        if file_info.file_type != FileType.STATS:
            errors.append(
                f"Expected a stats file as {role} file, "
                f"but received a {file_info.file_type} file. "
                "Please provide a stats file."
            )
            return None

        return file_name

    errors.append(
        f"Expected exactly one {role} file, "
        f"but received {len(files)} files. "
        "Please provide a single file."
    )
    return None
