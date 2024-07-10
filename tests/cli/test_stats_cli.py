import os
import pytest

import pandas as pd

from click.testing import CliRunner

from engine.stats import stats
from engine.perturb import perturb
from tests.util.fixtures import df_ref_stats, nc_with_T_U_V, ref_data, df_ref_ensemble_stats, tmp_dir

def load_pandas(file):
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


def generate_ensemble(tmp_path, filename, perturb_amplitude=10e-12):
    run_perturb_cli(tmp_path, filename, perturb_amplitude)

    return os.path.join(tmp_path, "experiments/{member_id}")

def test_stats_cli_no_ensemble(nc_with_T_U_V, df_ref_stats):
    tmp_path = os.path.dirname(nc_with_T_U_V)
    stats_file = os.path.join(tmp_path, 'test_stats.csv')
    run_stats_cli(tmp_path, stats_file, '--no-ensemble')
    df_test = load_pandas(stats_file)
    err = pandas_error(df_ref_stats, df_test)

    assert len(err.values) == 0 , "Stats datasets are not equal!\n{}".format(err)

@pytest.mark.xfail(reason="perturb amplitude 10e-14 smaller than perturb amplitude 10e-12 of reference")
@pytest.mark.parametrize("member", range(1, 3))
def test_stats_cli_ensemble_with_too_small_perturb_amplitude_for_member(nc_with_T_U_V, df_ref_ensemble_stats, member):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)
    exp_folder = generate_ensemble(tmp_path, initial_condition,perturb_amplitude=10e-14)
    stats_file = os.path.join(tmp_path, 'stats_{member_id}.csv')
    run_stats_cli(tmp_path, stats_file, '--ensemble', exp_folder)
    df_test = load_pandas(stats_file.format(member_id='dp_' + str(member)))
    err = pandas_error(df_ref_ensemble_stats[member], df_test)


    assert len(err.values) == 0 , "Stats datasets are not equal!\n{}".format(err)

@pytest.mark.parametrize("member", range(1, 11))
def test_stats_cli_ensemble_for_member(nc_with_T_U_V, df_ref_ensemble_stats, member):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)
    exp_folder = generate_ensemble(tmp_path, initial_condition)
    stats_file = os.path.join(tmp_path, 'stats_{member_id}.csv')
    run_stats_cli(tmp_path, stats_file, '--ensemble', exp_folder)
    df_test = load_pandas(stats_file.format(member_id='dp_' + str(member)))
    err = pandas_error(df_ref_ensemble_stats[member], df_test)

    assert len(err.values) == 0 , "Stats datasets are not equal!\n{}".format(err)
