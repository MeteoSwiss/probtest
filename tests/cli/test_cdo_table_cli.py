import os

import pandas as pd
from click.testing import CliRunner

from engine.cdo_table import cdo_table
from engine.perturb import perturb
from tests.util.fixtures import (  # noqa: F401
    df_ref_cdo_table,
    nc_with_T_U_V,
    ref_data,
    tmp_dir,
)


def load_pandas(file):
    return pd.read_csv(file, index_col=[0, 1], header=[0, 1])


def pandas_error(df_ref, df_cur):
    diff = (df_ref - df_cur).abs()
    err_mask = (diff > 0).any(axis=1)

    return diff[err_mask]


def run_perturb_cli_new(base_dir, filename, perturb_amplitude):
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
            "1",
            "--member-type",
            "sp",
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


def run_cdo_table_cli(base_dir, filename, perturbed_model_output_dir):
    runner = CliRunner()
    result = runner.invoke(
        cdo_table,
        [
            "--model-output-dir",
            base_dir,
            "--cdo-table-file",
            filename,
            "--member-type",
            "sp",
            "--file-id",
            "NetCDF",
            "*.nc",
            "--perturbed-model-output-dir",
            perturbed_model_output_dir,
            "--file-specification",
            [
                {
                    "NetCDF": {
                        "format": "netcdf",
                        "time_dim": "time",
                        "horizontal_dims": ["lat", "lon"],
                    }
                }
            ],
        ],
    )
    if result.exit_code != 0:
        error_message = "Error executing command:\n" + result.output
        if result.exception:
            error_message += "\nException: " + str(result.exception)
        raise Exception(error_message)


def generate_ensemble(tmp_path, filename, perturb_amplitude=10e-12):
    run_perturb_cli_new(tmp_path, filename, perturb_amplitude)

    return os.path.join(tmp_path, "experiments/{member_id}")


def test_cdo_table_cli(nc_with_T_U_V, df_ref_cdo_table):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)
    exp_folder = generate_ensemble(
        tmp_path, initial_condition, perturb_amplitude=10e-14
    )
    cdo_table_file = os.path.join(tmp_path, "cdo_table.csv")
    run_cdo_table_cli(tmp_path, cdo_table_file, exp_folder)
    df_test = load_pandas(cdo_table_file)
    err = pandas_error(df_ref_cdo_table, df_test)

    assert len(err.values) == 0, "CDO table datasets are not equal!\n{}".format(err)
