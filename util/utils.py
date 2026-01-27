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
        2547566401,
        3390797009,
        2583677123,
        2473118423,
        2143626467,
        4173353531,
        1628145667,
        3371547601,
        3301010651,
        2632851427,
        2655986491,
        439228313,
        2037313633,
        2681973191,
        92727749,
        424192897,
        3575206447,
        1431770341,
        2158581499,
        1389519269,
        3860054131,
        1423646869,
        100708747,
        271345577,
        2226813817,
        541149953,
        4244349953,
        3656774221,
        319908709,
        2563334467,
        2130688607,
        240773417,
        2530154009,
        3415037839,
        1546988207,
        2804242241,
        2454029327,
        3076953343,
        2214797017,
        148621783,
        919008731,
        3073167649,
        2544996131,
        1560372031,
        747252683,
        676033667,
        1621633327,
        1497119137,
        2115885181,
        3052669511,
        363171217,
        2241551287,
        1013429699,
        2261903797,
        2731086991,
        2705856533,
        2958671119,
        1063346831,
        2497761787,
        3199470749,
        2719907293,
        1367554031,
        431355971,
        1508132207,
        2657310209,
        515649373,
        422364193,
        1538703743,
        4215084967,
        3862569617,
        3112572893,
        2766389459,
        1220880631,
        1925252027,
        1314078553,
        2728133203,
        3409408591,
        1924140919,
        3036144263,
        4211976127,
        2710652809,
        4294706461,
        2258428867,
        3141914599,
        2581251989,
        395560769,
        3630067789,
        2785597153,
        1599809011,
        3586998517,
        3974001257,
        1972680337,
        385683763,
        1766890739,
        3719579741,
        74170891,
        165779983,
        2615058419,
        412023839,
        3993305273,
        1678330057,
        1202490409,
        3263858743,
        2986703311,
        3476257331,
        2694509239,
        3275559121,
        3515429279,
        2871479801,
        490303711,
        1979395757,
        1959584257,
        819309853,
        2907252791,
        4029093293,
        2289808813,
        2119985837,
        121836503,
        3955242553,
        736607237,
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

        if "fof" in name or "ekf" in name:
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
