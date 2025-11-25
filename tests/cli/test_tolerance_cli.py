"""
This module contains tests for verifying the correctness of the tolerance CLI
command.
The tests compare the output of the tolerance CLI with reference tolerance data
and check for discrepancies.
"""

import pytest

from tests.cli.test_tolerance_cli_split import (
    test_tolerance_cli_fof,
    test_tolerance_cli_stats,
)


@pytest.mark.parametrize(
    "mode,use_minimum_tolerance",
    [
        ("stats", True),
        ("stats", False),
        ("fof", False),
    ],
)
def test_tolerance_cli_unified(
    mode,
    use_minimum_tolerance,
    ref_data,
    tmp_dir,
    new_ref,
    df_ref_tolerance,
    df_ref_tolerance_fof,
    fof_file_set,
):  # pylint: disable=too-many-positional-arguments

    if mode == "stats":
        test_tolerance_cli_stats(
            ref_data, df_ref_tolerance, tmp_dir, new_ref, use_minimum_tolerance
        )

    elif mode == "fof":
        test_tolerance_cli_fof(
            fof_file_set, df_ref_tolerance_fof, new_ref, use_minimum_tolerance
        )
