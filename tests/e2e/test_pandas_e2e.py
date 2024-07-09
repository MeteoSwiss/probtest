import os
import unittest
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
import pytest

from util.tree import TimingTree

from click.testing import CliRunner

from engine.stats import stats
from engine.perturb import perturb
from tests.util.fixtures import df_ref_stats, nc_with_T_U_V, ref_data, df_ref_ensemble_stats

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2e}".format)


def load_pandas(rel_path, index_col=[0, 1], header=[0, 1]):
    input_file_ref = Path(os.environ["PROBTEST_REF_DATA"]) / rel_path
    input_file_cur = Path(os.environ["PROBTEST_CUR_DATA"]) / rel_path
    df_ref = pd.read_csv(input_file_ref, index_col=index_col, header=header)
    df_cur = pd.read_csv(input_file_cur, index_col=index_col, header=header)

    return df_ref, df_cur

def load_pands(file):
    return pd.read_csv(file, index_col=[0, 1, 2], header=[0, 1])


def pandas_error(df_ref, df_cur):
    diff = (df_ref - df_cur).abs()
    err_mask = (diff > 0).any(axis=1)

    return diff[err_mask]

def run_perturb_cli(base_dir, filename, perturb_amplitude):
    runner = CliRunner()
    result = runner.invoke(
        perturb,
        [
            "--model-input-dir",
            base_dir,
            "--perturbed-model-input-dir",
            f"{base_dir}/experiments/" + "{member_id}",
            "--files",
            filename,
            "--member-num",
            "10",
            "--member-type",
            "dp",
            "--variable-names",
            "U,V",
            "--perturb-amplitude",
            f"{perturb_amplitude}",
            "--no-copy-all-files",
        ],
    )
    if result.exit_code != 0:
        error_message = "Error executing command:\n" + result.output
        if result.exception:
            error_message += "\nException: " + str(result.exception)
        raise Exception(error_message)

def run_stats_cli(base_dir, filename, ensemble_flag, perturbed_model_output_dir='./'):
    runner = CliRunner()
    result = runner.invoke(
        stats,
        [
            "--model-output-dir",
            base_dir,
            "--stats-file-name",
            filename,
            "--member-type",
            "dp",
            "--file-id",
            'NetCDF', '*.nc',
            ensemble_flag,
            "--perturbed-model-output-dir",
            perturbed_model_output_dir,
            '--file-specification',
            [{ "NetCDF": { "format": "netcdf", "time_dim": "time", "horizontal_dims": ["lat", "lon"] }}]

        ],
    )
    if result.exit_code != 0:
        error_message = "Error executing command:\n" + result.output
        if result.exception:
            error_message += "\nException: " + str(result.exception)
        raise Exception(error_message)


def generate_ensemble(tmp_path, filename):
    run_perturb_cli(tmp_path, filename, 10e-12)

    return os.path.join(tmp_path, "experiments/{member_id}")

def test_stats_cli_no_ensemble(nc_with_T_U_V, df_ref_stats):
    tmp_path = os.path.dirname(nc_with_T_U_V)
    stats_file = os.path.join(tmp_path, 'test_stats.csv')
    run_stats_cli(tmp_path, stats_file, '--no-ensemble')
    df_test = load_pands(stats_file)
    err = pandas_error(df_ref_stats, df_test)

    assert len(err.values) == 0 , "Stats datasets are not equal!\n{}".format(err)

@pytest.mark.parametrize("member", range(1, 11))
def test_stats_cli_ensemble_for_member(nc_with_T_U_V, df_ref_ensemble_stats, member):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)
    exp_folder = generate_ensemble(tmp_path, initial_condition)
    stats_file = os.path.join(tmp_path, 'stats_{member_id}.csv')
    run_stats_cli(tmp_path, stats_file, '--ensemble', exp_folder)
    df_test = load_pands(stats_file.format(member_id='dp_' + str(member)))
    err = pandas_error(df_ref_ensemble_stats[member], df_test)

    assert len(err.values) == 0 , "Stats datasets are not equal!\n{}".format(err)

#class TestPandasE2E(unittest.TestCase):
#    def setUp(self) -> None:
#        return super().setUp()
#
#    def tearDown(self) -> None:
#        return super().tearDown()
#
#
#    def test_tolerance_e2e(self):
#        exp_name = os.environ["PROBTEST_TEST_EXPERIMENT"]
#        rel_path = "tolerance/{}.csv".format(exp_name)
#        df_ref, df_cur = load_pandas(rel_path)
#
#        err = pandas_error(df_ref, df_cur)
#
#        self.assertEqual(
#            len(err.values), 0, "Tolerance datasets are not equal!\n{}".format(err)
#        )
#
#        return
#
#    def test_cdo_e2e(self):
#        exp_name = os.environ["PROBTEST_TEST_EXPERIMENT"]
#        rel_path = "cdo_table/{}.csv".format(exp_name)
#        df_ref, df_cur = load_pandas(rel_path)
#
#        err = pandas_error(df_ref, df_cur)
#
#        self.assertEqual(
#            len(err.values), 0, "CDO table datasets are not equal!\n{}".format(err)
#        )
#
#        return
#
#    def test_performance_data_e2e(self):
#        exp_name = os.environ["PROBTEST_TEST_EXPERIMENT"]
#
#        df_ref = TimingTree.from_json(
#            Path(os.environ["PROBTEST_REF_DATA"]) / "performance/{}".format(exp_name)
#        )
#        df_cur = TimingTree.from_json(
#            Path(os.environ["PROBTEST_CUR_DATA"]) / "performance/{}".format(exp_name)
#        )
#
#        for i in range(len(df_ref.data)):
#            err = pandas_error(df_ref.data[i], df_cur.data[i])
#
#            self.assertEqual(
#                len(err.values),
#                0,
#                "Performance datasets are not equal for table {}!\n{}".format(err, i),
#            )
#
