import os
import pytest

import numpy as np
import xarray as xr

from click.testing import CliRunner

from engine.perturb import perturb

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

@pytest.fixture(scope="function")
def nc_with_T_U_V(tmp_path):
    # Define dimensions
    time = np.arange(0, 10)
    lat = np.linspace(-90, 90, 19)
    lon = np.linspace(-180, 180, 37)

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

def cli_run_perturb(base_dir,filename,member,id):
    runner = CliRunner()
    result = runner.invoke(perturb, [
        "--model-input-dir", base_dir,
        "--perturbed-model-input-dir", f"{base_dir}/experiments/{id}_" + "{member_id}",
        "--files", filename,
        "--member-num", f"1,{str(member)}",
        "--member-type", "dp",
        "--variable-names", "U,V",
        "--perturb-amplitude", "0.05",
        "--no-copy-all-files"
    ])
    if result.exit_code != 0:
        error_message = "Error executing command:\n" + result.output
        if result.exception:
            error_message += "\nException: " + str(result.exception)
        raise Exception(error_message)


@pytest.mark.parametrize("member", [3,5,13,33,50,73,87,99])
def test_perturb_cli_for_member(nc_with_T_U_V,tmp_path,member):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)

    cli_run_perturb(tmp_path,initial_condition,member, "ref")
    cli_run_perturb(tmp_path, initial_condition,member, "test")

    data_ref = load_netcdf(os.path.join(tmp_path,f"experiments/ref_dp_{member}/initial_condition.nc"))
    data_test = load_netcdf(os.path.join(tmp_path,f"experiments/test_dp_{member}/initial_condition.nc"))

    diff_keys, err = check_netcdf(data_ref, data_test)

    assert err == [], f"The following variables contain errors:\n{err}"
    assert diff_keys == [], f"The following variables are not contained in both files:\n{err}"
