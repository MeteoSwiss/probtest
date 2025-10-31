"""
This module contains tests for verifying the correctness of the tolerance CLI
command.
The tests compare the output of the tolerance CLI with reference tolerance data
and check for discrepancies.
"""

import os

import pandas as pd
import pytest

from tests.helpers import (
    assert_empty_df,
    load_pandas,
    pandas_error,
    run_tolerance_cli,
    store_as_potential_new_ref,
)


@pytest.mark.parametrize("use_minimum_tolerance", [True, False])
def test_tolerance_cli_stats(
    ref_data, df_ref_tolerance, tmp_dir, new_ref, use_minimum_tolerance
):

    stats_file_name = os.path.join(ref_data, "stats_{member_type}{member_id}.csv")
    tolerance_file_name = os.path.join(tmp_dir, "tolerance.csv")

    if use_minimum_tolerance:
        run_tolerance_cli(
            stats_file_name,
            tolerance_file_name,
            member_type="dp_",
            minimum_tolerance=1e-14,
        )

        df_test = load_pandas(tolerance_file_name, index_col=[0, 1])
        err = pandas_error(df_ref_tolerance, df_test)

    else:
        run_tolerance_cli(
            stats_file_name,
            tolerance_file_name,
            member_type="dp_",
        )

        df_test = load_pandas(tolerance_file_name, index_col=[0, 1])
        modified_df_ref_tolerance = df_ref_tolerance
        modified_df_ref_tolerance.loc[("NetCDF:*.nc", "T")] = 0.0

        err = pandas_error(modified_df_ref_tolerance, df_test)

    store_as_potential_new_ref(tolerance_file_name, new_ref)

    assert_empty_df(err, "Tolerance datasets are not equal!")


@pytest.mark.parametrize("use_minimum_tolerance", [True, False])
def test_tolerance_cli_fof(
    fof_file_set, df_ref_tolerance_fof, new_ref, use_minimum_tolerance
):

    fof_pattern = fof_file_set["path"]
    tolerance_files = fof_file_set["tol"]
    member_ids = "1,2,3,4"

    for fof_file in fof_file_set["fof"]:
        assert os.path.exists(fof_file), f"File FOF missing: {fof_file}"

    if use_minimum_tolerance:

        run_tolerance_cli(
            fof_pattern, tolerance_files, member_ids=member_ids, minimum_tolerance=1e-14
        )

        df_test = pd.read_csv(tolerance_files[0], index_col=[0])
        df_test.columns = [None]
        err = pandas_error(df_ref_tolerance_fof, df_test)

    else:
        run_tolerance_cli(fof_pattern, tolerance_files, member_ids=member_ids)

        df_test = pd.read_csv(tolerance_files[0], index_col=[0])
        df_test.columns = [None]
        err = pandas_error(df_ref_tolerance_fof, df_test)

    store_as_potential_new_ref(tolerance_files[0], new_ref)

    assert_empty_df(err, "Tolerance datasets are not equal!")
