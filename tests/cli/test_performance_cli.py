import os

from tests.helpers.fixtures import (  # noqa: F401
    df_ref_performance,
    ref_data,
    timing_logfile,
    tmp_dir,
)
from tests.helpers.helpers import assert_empty_df, pandas_error, run_performance_cli
from util.tree import TimingTree


def test_performance_cli(timing_logfile, tmp_dir, df_ref_performance):
    timing_database = os.path.join(tmp_dir, "test")
    run_performance_cli(timing_logfile, timing_database)
    df_test = TimingTree.from_json(timing_database)
    for i in range(len(df_ref_performance.data)):
        err = pandas_error(df_ref_performance.data[i], df_test.data[i])

        assert_empty_df(err, "Performance datasets are not equal!")
