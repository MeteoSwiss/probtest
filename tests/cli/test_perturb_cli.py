import os
import pytest

import numpy as np
import xarray as xr
import pandas as pd

from click.testing import CliRunner

from engine.perturb import perturb

@pytest.fixture()
def ref_data():
    return 'tests/data'

@pytest.fixture()
def ds_ref_with_T_U_V(ref_data):
    data_ref = pd.read_csv(os.path.join(ref_data,'initial_condition.csv')).set_index(['time','lon','lat'])
    return xr.Dataset.from_dataframe(data_ref)

@pytest.fixture()
def ds_with_T_U_V(nc_with_T_U_V):
    return load_netcdf(nc_with_T_U_V)

@pytest.fixture()
def nc_with_T_U_V(tmp_path):
    # Define dimensions
    time = np.arange(0, 5)
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    # Create a meshgrid for lat and lon
    lon, lat = np.meshgrid(lon, lat)

    # Generate non-random data for variables T,V and U
    T = 20 + 5 * np.sin(np.pi * lat/180)  # Temperature varies sinusoidally with latitude
    V = 100 * np.cos(np.pi * lon/180)  # Velocity varies cosinusoidally with longitude
    U = 100 * np.sin(np.pi * lon/180)  # Velocity varies sinusoidally with longitude

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "T": (("time", "lat", "lon"), np.tile(T[np.newaxis, :, :], (len(time), 1, 1))),
            "V": (("time", "lat", "lon"), np.tile(V[np.newaxis, :, :], (len(time), 1, 1))),
            "U": (("time", "lat", "lon"), np.tile(U[np.newaxis, :, :], (len(time), 1, 1)))
        },
        coords={
            "time": time,
            "lat": ("lat", lat[:, 0]),
            "lon": ("lon", lon[0, :])
        }
    )
    # Save to netcdf
    filename = os.path.join(tmp_path,"initial_condition.nc")
    ds.to_netcdf(filename)

    return filename


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


def run_perturb_cli(base_dir,filename, perturb_amplitude):
    runner = CliRunner()
    result = runner.invoke(perturb, [
        "--model-input-dir", base_dir,
        "--perturbed-model-input-dir", f"{base_dir}/experiments/" + "{member_id}",
        "--files", filename,
        "--member-num", "1",
        "--member-type", "dp",
        "--variable-names", "U,V",
        "--perturb-amplitude", f'{perturb_amplitude}',
        "--no-copy-all-files"
    ])
    if result.exit_code != 0:
        error_message = "Error executing command:\n" + result.output
        if result.exception:
            error_message += "\nException: " + str(result.exception)
        raise Exception(error_message)


@pytest.mark.xfail
def test_compare_ref_with_data_from_fixture_V_missing(ds_ref_with_T_U_V,ds_with_T_U_V):
    ds_with_T_U_V = ds_with_T_U_V.drop_vars('V')
    diff_keys, err = check_netcdf(ds_ref_with_T_U_V, ds_with_T_U_V)
    assert diff_keys == [], f"The following variables are not contained in both files:\n{err}"

def test_compare_ref_with_data_from_fixture(ds_ref_with_T_U_V,ds_with_T_U_V):
    diff_keys, err = check_netcdf(ds_ref_with_T_U_V, ds_with_T_U_V)
    assert err == [], f"The following variables contain errors:\n{err}"
    assert diff_keys == [], f"The following variables are not contained in both files:\n{err}"

def test_perturb_cli_amplitude_0_0(nc_with_T_U_V,ds_ref_with_T_U_V):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)

    run_perturb_cli(tmp_path, initial_condition, 0.0)

    data_test = load_netcdf(os.path.join(tmp_path,f"experiments/dp_1/initial_condition.nc"))

    diff_keys, err = check_netcdf(ds_ref_with_T_U_V, data_test)

    assert err == [], f"The following variables contain errors:\n{err}"
    assert diff_keys == [], f"The following variables are not contained in both files:\n{err}"

def test_perturb_cli_amplitude_0_2(nc_with_T_U_V,ds_ref_with_T_U_V):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)

    run_perturb_cli(tmp_path, initial_condition, 0.2)

    data_test = load_netcdf(os.path.join(tmp_path,f"experiments/dp_1/initial_condition.nc"))

    diff_keys, err = check_netcdf(ds_ref_with_T_U_V, data_test)

    # Remove U and V from the list of variables with errors because they are perturbed
    err.remove('U')
    err.remove('V')

    assert err == [], f"The following variables contain errors:\n{err}"
    assert diff_keys == [], f"The following variables are not contained in both files:\n{err}"