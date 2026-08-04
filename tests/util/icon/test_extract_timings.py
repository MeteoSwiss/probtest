"""
This module contains a focused regression test for read_logfile's handling of
timing tables whose rows are interleaved in the log file, instead of appearing
as clean, separate, contiguous blocks (as can happen with certain MPI rank
output orderings from ICON).
"""

import logging

from util.icon.extract_timings import read_logfile

# each data row needs at least one trailing space after its last value:
# TIMING_ELEMENT_REGEX requires a trailing "+" after every numeric token,
# including the last one, so a padding suffix is added to every row below.
_PAD = "    "

_LINES = [
    "Sun Jun 26 20:00:00 CEST 2022",
    "",
    "Timer report, ranks 0-3",
    " -----------------------------------   -------   ------------   --------   ------------   ------------   --------   -------------   --------------   -------------   --------------   -------------",
    " name                                  # calls   t_min          min rank   t_avg          t_max          max rank   total min (s)   total min rank   total max (s)   total max rank   total avg (s)",
    " -----------------------------------   -------   ------------   --------   ------------   ------------   --------   -------------   --------------   -------------   --------------   -------------",
    " ------------------------   -------   ------------   ------------   ------------   -------------   -------------   -------------",
    " name                       # calls   t_min          t_avg          t_max          total min (s)   total max (s)   total avg (s)",
    " ------------------------   -------   ------------   ------------   ------------   -------------   -------------   -------------",
    "",
    " wrt_output                 2             0.00000s        7.1658s       14.3315s         14.332          14.332           1.791" + _PAD,
    " -------------------------------------------------------------------------------------------------------------------------------",
    "",
    " total                                 8             15.8031s   [1]            15.8762s       16.0712s   [0]              15.803    [1]                    16.071    [0]                    15.876" + _PAD,
    " model_init                 1             35.9140s       35.9140s       35.9140s         35.914          35.914           4.489" + _PAD,
    "  L compute_domain_decomp   1             35.9139s       35.9139s       35.9139s         35.914          35.914           4.489" + _PAD,
    "  L integrate_nh                       8              2.5080s   [0]             3.0333s        3.1584s   [3]               2.508    [0]                     3.158    [3]                     3.033" + _PAD,
    "     L nh_solve                        40            0.03204s   [0]            0.11028s       0.40723s   [1]               0.536    [0]                     0.557    [3]                     0.551" + _PAD,
    "     L nh_hdiff                        16            0.00706s   [0]            0.26922s       0.64131s   [3]               0.016    [0]                     0.654    [3]                     0.538" + _PAD,
    "     L physics                         8              1.7170s   [0]             1.7191s        1.7200s   [6]               1.717    [0]                     1.720    [6]                     1.719" + _PAD,
    "     L transport                       8             0.19864s   [4]            0.19974s       0.20102s   [2]               0.199    [4]                     0.201    [2]                     0.200" + _PAD,
    " orphan_row   1   2.0s   3.0s   4.0s   5.0s   6.0s   7.0s   8.0s" + _PAD,
    "  L write_restart                      16            0.00000s   [0]            0.00002s       0.00003s   [0]               0.000    [5]                     0.000    [6]                     0.000" + _PAD,
    " exch_data                             328           0.00015s   [0]            0.02736s       0.63455s   [3]               0.107    [0]                     1.678    [3]                     1.122" + _PAD,
    " --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
    "",
    "Sun Jun 26 20:11:23 CEST 2022",
    "",
]

INTERLEAVED_LOG = "\n".join(_LINES)


def test_read_logfile_handles_interleaved_tables(tmp_path, caplog):
    """
    The 13-column ("total" tree) table's header is immediately followed by
    the 8-column ("wrt_output") table's header, and rows from the two tables
    end up mixed together rather than in two contiguous blocks. read_logfile
    must route each row to the table matching its column count instead of
    crashing or silently mis-assigning rows.
    """
    logfile = tmp_path / "interleaved.txt"
    logfile.write_text(INTERLEAVED_LOG)

    with caplog.at_level(logging.WARNING):
        timing_data, meta_data = read_logfile(str(logfile))

    # the small (wrt_output) table only has 3 rows and is dropped by the
    # "not interested in small tables" filter, same as today; what matters is
    # that its rows don't leak into the big table below
    assert meta_data["n_tables"] == 1

    big_table = timing_data[0]
    assert big_table["name"] == [
        "total",
        "integrate_nh",
        "nh_solve",
        "nh_hdiff",
        "physics",
        "transport",
        "write_restart",
    ]

    # the orphan row matches no known header's column count and must be
    # skipped with a warning, not silently mis-assigned or fatal
    assert any(
        "matches no known header" in record.message for record in caplog.records
    )
