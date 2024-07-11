import os

from tests.helpers.fixtures import df_ref_tolerance, ref_data, tmp_dir  # noqa: F401
from tests.helpers.helpers import (
    assert_empty_df,
    load_pandas,
    pandas_error,
    run_tolerance_cli,
)


def test_tolerance_cli(ref_data, df_ref_tolerance, tmp_dir):
    stats_file_name = os.path.join(ref_data, "stats_{member_id}.csv")
    tolerance_file_name = os.path.join(tmp_dir, "tolerance.csv")
    run_tolerance_cli(stats_file_name, tolerance_file_name)
    df_test = load_pandas(tolerance_file_name, index_col=[0, 1])
    err = pandas_error(df_ref_tolerance, df_test)
    assert_empty_df(err, "Tolerance datasets are not equal!")
