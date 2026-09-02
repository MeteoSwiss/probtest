"""
This module provides utilities for parsing log files and converting them into
timing data.
"""

import re
from typing import Optional

import numpy as np
from dateutil.parser import ParserError
from dateutil.parser import parse as parse_date

from util.constants import DATETIME_FORMAT
from util.log_handler import logger

TIMING_START_REGEX = r"\s+L?\s*[a-zA-Z_.]+"
TIMING_ELEMENT_REGEX = r"(?:\[?\d+[.msh]?\d*s?\]? +)"
TIMING_REGEX = TIMING_START_REGEX + r"\s+(?:" + TIMING_ELEMENT_REGEX + r"){6,20} *(?!.)"
HEADER_REGEX = r"name +.*calls.*"
INDENT_REGEX = r"^ *L? "
HOUR_REGEX = r"(\d+)h(\d+)m(\d+)s"
MINUTE_REGEX = r"(\d+[.]?\d*)m(\d+[.]?\d*)s"
SEC_REGEX = r"(\d+[.]?\d*)s"
NUMBER_REGEX = r"(\d+[.]?\d*)"

DICT_REGEX = r"^\s*{} *: *(.*)"


def _find_target_table(
    elements: list[str],
    current_table: Optional[int],
    header_elements_list: list[list[str]],
) -> Optional[int]:
    """
    Return the index of the table whose header column count matches
    `elements`: prefer `current_table`, falling back to any earlier header
    with a matching column count (needed when two tables' rows interleave).
    Returns None if no header matches.
    """
    if current_table is not None and len(elements) == len(
        header_elements_list[current_table]
    ):
        return current_table

    for k in range(len(header_elements_list) - 1, -1, -1):
        if k != current_table and len(elements) == len(header_elements_list[k]):
            return k

    return None


def read_logfile(filename: str) -> tuple[list[dict[str, list]], dict[str, object]]:
    with open(filename, "r", encoding="latin-1") as f:
        # read file into list of lines, remove empty lines
        full_file = f.read()
        data = [e for e in full_file.split("\n") if e != ""]

        # filter by timing headers and elements
        data = [
            e for e in data if re.search(HEADER_REGEX, e) or re.search(TIMING_REGEX, e)
        ]

        # store line numbers of timing table headers
        header_lines = [i for i, e in enumerate(data) if re.search(HEADER_REGEX, e)]
        header_positions: set[int] = set(header_lines)

        # parse each table's header up front, so rows can be routed to the
        # right table even if two tables' rows end up interleaved in the log
        # (this can happen with certain MPI rank output orderings)
        header_elements_list: list[list[str]] = [
            [
                e.lstrip().rstrip()
                for e in data[i_header].split("  ")
                if e not in ["", " "]
            ]
            for i_header in header_lines
        ]
        timing_data: list[dict[str, list]] = [
            {**{e: [] for e in header_elements}, "indent": [], "name": []}
            for header_elements in header_elements_list
        ]

        # walk the file once, routing each row to the table whose header
        # column count it matches: prefer the most recently seen header, but
        # fall back to any earlier header with a matching column count so
        # interleaved rows still land in their real table
        current_table: Optional[int] = None
        for i, line in enumerate(data):
            if i in header_positions:
                current_table = header_lines.index(i)
                continue

            elements: list[str] = [
                e.replace("[", "").replace("]", "")
                for e in line.split(" ")
                if e not in ["", "L"]
            ]

            target: Optional[int] = _find_target_table(
                elements, current_table, header_elements_list
            )

            if target is None:
                logger.warning(
                    "Skipping table row that matches no known header: %s", line
                )
                continue

            header_elements: list[str] = header_elements_list[target]
            timing_data_k: dict[str, list] = timing_data[target]

            # find indentation level for each table line
            indent_match = re.search(INDENT_REGEX, line)
            assert indent_match is not None
            first = indent_match.group(0)
            # assume 1 indent is 3 white spaces
            timing_data_k["indent"].append(len(first) // 3)

            timing_data_k["name"].append(elements[0])
            for i_element in np.arange(1, len(elements)):
                timing_data_k[header_elements[i_element]].append(
                    parse_time(elements[i_element])
                )

        # skip the small wrt_output table (by name) and any table left empty
        timing_data = [
            t for t in timing_data if t["name"] and t["name"][0] != "wrt_output"
        ]
        # start parsing meta data from log
        meta_data: dict[str, object] = {}

        # get start and finish time from job
        # --- robust start/finish datetime extraction ---

        datelines = []

        for line in full_file.splitlines():
            if line.count(":") >= 2:
                try:
                    dt = parse_date(line, fuzzy=False)
                    datelines.append(dt)
                except (ParserError, ValueError):
                    continue

        if len(datelines) < 2:
            raise Exception("Could not robustly determine start and finish time.")

        start_dt = datelines[0]
        finish_dt = datelines[-1]

        meta_data["start_time"] = start_dt.strftime(DATETIME_FORMAT)
        meta_data["finish_time"] = finish_dt.strftime(DATETIME_FORMAT)

        # get meta data from ICON log (in the form "Key : Value")
        revision = re.search(
            DICT_REGEX.format("revision"), full_file, re.IGNORECASE | re.MULTILINE
        )
        branch = re.search(
            DICT_REGEX.format("branch"), full_file, re.IGNORECASE | re.MULTILINE
        )

        meta_data["revision"] = revision.group(1).strip() if revision else None
        meta_data["branch"] = branch.group(1).strip() if branch else None
    meta_data["n_tables"] = len(timing_data)
    meta_data["entries"] = [len(e["indent"]) for e in timing_data]

    return timing_data, meta_data


def parse_time(time_string):
    m1 = re.match(HOUR_REGEX, time_string)
    m2 = re.match(MINUTE_REGEX, time_string)
    m3 = re.match(SEC_REGEX, time_string)
    m4 = re.match(NUMBER_REGEX, time_string)
    if m1:
        h, m, s = [m1.group(i) for i in [1, 2, 3]]
    elif m2:
        m, s = [m2.group(i) for i in [1, 2]]
        h = 0
    elif m3:
        s = m3.group(1)
        h = 0
        m = 0
    elif m4:
        s = m4.group(0)
        h = 0
        m = 0
    else:
        s = 0
        m = 0
        h = 0
        logger.warning("did not match regex")
    out = float(h) * 60 * 60 + float(m) * 60 + float(s)
    return out
