import os
import pytest
from pathlib import Path

import numpy as np
import xarray as xr

from click.testing import CliRunner

from engine.perturb import perturb


def load_netcdf(rel_path):
    nc_ref_path = Path(os.environ["PROBTEST_REF_DATA"]) / rel_path
    nc_cur_path = Path(os.environ["PROBTEST_CUR_DATA"]) / rel_path

    data_ref = xr.load_dataset(nc_ref_path)
    data_cur = xr.load_dataset(nc_cur_path)

    return data_ref, data_cur


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
def tmp_folder(tmpdir):
    return tmpdir.mkdir("data")

@pytest.fixture(scope="function")
def nc_with_T_U_V(tmp_folder):
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
    filename = os.path.join(tmp_folder,"initial_condition.nc")
    ds.to_netcdf(filename)
    return filename

def test_perturb(nc_with_T_U_V,tmp_folder):
    data_cur = nc_with_T_U_V

    print(data_cur)
    print(tmp_folder)
    print(type(data_cur))
    print(type(tmp_folder))
    runner = CliRunner()
    result = runner.invoke(perturb, ["--experiment-name", "test1", "--model-input-dir", 
                                     tmp_folder, "--perturbed-model-input-dir", "./", 
                                     "--files", "initial_condition.nc", "--member-num", "3", 
                                     "--member-type", "sp"])

    print(result.output)
    #data_ref, data_cur = load_netcdf("perturb/initial_condition.nc")
    #diff_keys, err = check_netcdf(data_ref, data_cur)
    #assert len(err) == 0
    #assert len(diff_keys) == 0


#class TestNcE2E(unittest.TestCase):
#    def setUp(self) -> None:
#        return super().setUp()
#
#    def tearDown(self) -> None:
#        return super().tearDown()
#
#    def test_perturb_e2e(self):
#        exp = os.environ["PROBTEST_TEST_EXPERIMENT"]
#
#        # Get a list of directories that include the experiment name
#        perturb_path = Path(os.getcwd(), os.environ["PROBTEST_REF_DATA"], "perturb")
#        exp_dirs = [d for d in os.listdir(perturb_path) if exp in d]
#        self.assertNotEqual(
#            len(exp_dirs),
#            0,
#            msg="No experiment folders found in {}".format(perturb_path),
#        )
#
#        for exp_dir in exp_dirs:
#            data_ref, data_cur = load_netcdf(
#                "perturb/{}/initial_condition.nc".format(exp_dir)
#            )
#
#            diff_keys, err = check_netcdf(data_ref, data_cur)
#            self.assertEqual(
#                err, [], msg="The following variables contain errors:\n{}".format(err)
#            )
#            self.assertEqual(
#                diff_keys,
#                [],
#                msg="The following variables are not contained in both files:\n"
#                + "{}".format(err),
#            )

