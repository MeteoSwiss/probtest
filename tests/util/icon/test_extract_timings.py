"""
This module contains a focused regression test for read_logfile's handling of
interleaved timing tables, stray log messages, and truncated rows.
"""

import logging

import pytest

from util.icon.extract_timings import read_logfile

TIMING_FILE_INTERLEAVED: str = "tests/data/timing_example_interleaved.txt"


def test_read_logfile_handles_interleaved_tables(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    The 13-column ("total" tree) table's header is immediately followed by
    the 8-column ("wrt_output") table's header, and rows from the two tables
    end up mixed together rather than in two contiguous blocks. read_logfile
    must route each row to the table matching its column count instead of
    crashing or silently mis-assigning rows.
    """
    with caplog.at_level(logging.WARNING):
        timing_data, meta_data = read_logfile(TIMING_FILE_INTERLEAVED)

    # the small (wrt_output) table only has 3 rows and is dropped by the
    # "not interested in small tables" filter
    assert meta_data["n_tables"] == 1

    # a stray rank message and a truncated row sit between nh_hdiff and
    # physics but don't match the table-row format, so both are dropped
    big_table: dict[str, list] = timing_data[0]
    assert big_table["name"] == [
        "total",
        "integrate_nh",
        "nh_solve",
        "nh_hdiff",
        "physics",
        "transport",
        "write_restart",
    ]

    # only the orphan row (right shape, wrong column count) should warn
    warnings = [record.message for record in caplog.records]
    assert len(warnings) == 1
    assert "orphan_row" in warnings[0]
    assert "matches no known header" in warnings[0]
