import os

from tests.helpers.fixtures import (  # noqa: F401
    df_ref_performance,
    ref_data,
    timing_logfile,
    tmp_dir,
)
from tests.helpers.helpers import (
    assert_empty_df,
    assert_empty_list,
    pandas_error,
    run_performance_cli,
    store_timings_as_potential_new_ref,
)
from util.tree import TimingTree


def test_performance_cli_timing(new_ref, timing_logfile, tmp_dir, df_ref_performance):
    timing_database = os.path.join(tmp_dir, "test")
    run_performance_cli(timing_logfile, timing_database)
    df_test = TimingTree.from_json(timing_database)

    store_timings_as_potential_new_ref(tmp_dir, new_ref)

    for i in range(len(df_ref_performance.data)):
        err = pandas_error(df_ref_performance.data[i], df_test.data[i])

        assert_empty_df(err, "Performance datasets are not equal!")


def test_performance_cli_tree(timing_logfile, tmp_dir, df_ref_performance):
    tree_database = os.path.join(tmp_dir, "test")
    run_performance_cli(timing_logfile, tree_database)
    df_test = TimingTree.from_json(tree_database)

    for i in range(len(df_ref_performance.root)):
        ref_nodes = set(df_ref_performance.root[i].to_ancestry_name_list())
        cur_nodes = set(df_test.root[i].to_ancestry_name_list())

        diff_nodes = list(ref_nodes - cur_nodes)

        assert_empty_list(
            diff_nodes,
            "The test tree does not contain the following nodes:",
        )
