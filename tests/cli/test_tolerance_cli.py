"""
This module contains tests for verifying the correctness of the tolerance CLI
command.
The tests compare the output of the tolerance CLI with reference tolerance data
and check for discrepancies.
"""

import os
import pytest

from tests.helpers import (
    assert_empty_df,
    load_pandas,
    pandas_error,
    run_tolerance_cli,
    store_as_potential_new_ref,
)


@pytest.mark.parametrize("use_minimum_tolerance", [True, False])
def test_tolerance_cli(ref_data, df_ref_tolerance, tmp_dir, new_ref, use_minimum_tolerance):
    stats_file_name = os.path.join(ref_data, "stats_{member_id}.csv")
    tolerance_file_name = os.path.join(tmp_dir, "tolerance.csv")

    if use_minimum_tolerance:
        run_tolerance_cli(stats_file_name, tolerance_file_name, member_type="dp", minimum_tolerance=1e-14)
        df_test = load_pandas(tolerance_file_name, index_col=[0, 1])
        err = pandas_error(df_ref_tolerance, df_test)
    else:
        run_tolerance_cli(stats_file_name, tolerance_file_name, member_type="dp")

        # Create modified tolerance df with T tolerances equal to 0.0 to match `minimum_tolerance` default
        df_test = load_pandas(tolerance_file_name, index_col=[0, 1])
        modified_df_ref_tolerance = df_ref_tolerance
        modified_df_ref_tolerance.loc[("NetCDF:*.nc", "T")] = 0.0
        
        err = pandas_error(modified_df_ref_tolerance, df_test)
    
    store_as_potential_new_ref(tolerance_file_name, new_ref)

    assert_empty_df(err, "Tolerance datasets are not equal!")
