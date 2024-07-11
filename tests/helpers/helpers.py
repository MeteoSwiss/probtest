import numpy as np
import pandas as pd
import xarray as xr
from click.testing import CliRunner

from engine.cdo_table import cdo_table
from engine.perturb import perturb
from engine.stats import stats
from engine.tolerance import tolerance


def load_netcdf(path):
    return xr.load_dataset(path)


def load_pandas(file, index_col=[0, 1, 2], header=[0, 1]):
    return pd.read_csv(file, index_col=index_col, header=header)


def pandas_error(df_ref, df_cur):
    diff = (df_ref - df_cur).abs()
    err_mask = (diff > 0).any(axis=1)

    return diff[err_mask]


def check_netcdf(data_ref, data_cur):
    diff_keys = set(data_ref.keys()) - set(data_cur.keys())
    same_keys = set(data_ref.keys()).intersection(set(data_cur.keys()))

    err = []
    for key in same_keys:
        diff = np.fabs(data_ref[key] - data_cur[key])
        if np.sum(diff) > 0:
            err.append(key)

    return list(diff_keys), err


def assert_empty_list(lst, msg):
    assert lst == [], f"{msg}:\n{lst}"


def assert_empty_df(df, msg):
    assert len(df.values) == 0, f"{msg}:\n{df}"


def run_tolerance_cli(stats_file_name, tolerance_file_name):
    args = [
        "--stats-file-name",
        stats_file_name,
        "--tolerance-file-name",
        tolerance_file_name,
        "--member-num",
        "10",
        "--member-type",
        "dp",
    ]
    run_cli(tolerance, args)


def generate_ensemble(tmp_path, filename, perturb_amplitude=10e-12):
    return run_perturb_cli(tmp_path, filename, perturb_amplitude)


def run_perturb_cli(model_input_dir, files, perturb_amplitude, member_num=10):
    perturbed_model_input_dir = f"{model_input_dir}/experiments/{{member_id}}"
    args = [
        "--model-input-dir",
        model_input_dir,
        "--perturbed-model-input-dir",
        perturbed_model_input_dir,
        "--files",
        files,
        "--member-num",
        str(member_num),
        "--member-type",
        "dp",
        "--variable-names",
        "U,V",
        "--perturb-amplitude",
        f"{perturb_amplitude}",
        "--no-copy-all-files",
    ]
    run_cli(perturb, args)

    return perturbed_model_input_dir


def run_stats_cli(
    model_output_dir, stats_file_name, ensemble, perturbed_model_output_dir=None
):
    args = [
        "--model-output-dir",
        model_output_dir,
        "--stats-file-name",
        stats_file_name,
        "--member-type",
        "dp",
        "--file-id",
        "NetCDF",
        "*.nc",
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
    ]
    args += (
        ["--perturbed-model-output-dir", perturbed_model_output_dir]
        if perturbed_model_output_dir
        else []
    )
    args.append("--ensemble" if ensemble else "--no-ensemble")

    run_cli(stats, args)


def run_cdo_table_cli(model_output_dir, cdo_table_file, perturbed_model_output_dir):
    args = [
        "--model-output-dir",
        model_output_dir,
        "--cdo-table-file",
        cdo_table_file,
        "--member-type",
        "dp",
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
    ]
    run_cli(cdo_table, args)


def run_cli(command, args):
    runner = CliRunner()
    result = runner.invoke(command, args)
    print(result.output)
    if result.exit_code != 0:
        error_message = "Error executing command:\n" + result.output
        if result.exception:
            error_message += "\nException: " + str(result.exception)
        raise Exception(error_message)
