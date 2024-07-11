import numpy as np
import xarray as xr
from click.testing import CliRunner


def load_netcdf(path):
    return xr.load_dataset(path)


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


def run_cli(command, args):
    runner = CliRunner()
    result = runner.invoke(command, args)
    if result.exit_code != 0:
        error_message = "Error executing command:\n" + result.output
        if result.exception:
            error_message += "\nException: " + str(result.exception)
        raise Exception(error_message)
