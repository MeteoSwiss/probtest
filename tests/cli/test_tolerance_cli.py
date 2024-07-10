import os
import pytest

import pandas as pd

from click.testing import CliRunner

from engine.tolerance import tolerance
from tests.util.fixtures import ref_data, tmp_dir, df_ref_tolerance

def load_pandas(file):
    return pd.read_csv(file, index_col=[0, 1], header=[0, 1])


def pandas_error(df_ref, df_cur):
    diff = (df_ref - df_cur).abs()
    err_mask = (diff > 0).any(axis=1)

    return diff[err_mask]

def run_tolerance_cli(stats_file_name, tolerance_file_name):
    runner = CliRunner()
    result = runner.invoke(
        tolerance,
        [
            "--stats-file-name",
            stats_file_name,
            "--tolerance-file-name",
            tolerance_file_name,
            "--member-num",
            "10",
            "--member-type",
            "dp",
        ],
    )
    if result.exit_code != 0:
        error_message = "Error executing command:\n" + result.output
        if result.exception:
            error_message += "\nException: " + str(result.exception)
        raise Exception(error_message)

def test_tolerance_cli(ref_data,df_ref_tolerance,tmp_dir):
    stats_file_name = os.path.join(ref_data, "stats_{member_id}.csv")
    tolerance_file_name = os.path.join(tmp_dir,"tolerance.csv")
    run_tolerance_cli(stats_file_name, tolerance_file_name)
    df_test = load_pandas(tolerance_file_name)
    err = pandas_error(df_ref_tolerance, df_test)
    assert len(err.values) == 0 , "Tolerance datasets are not equal!\n{}".format(err)
