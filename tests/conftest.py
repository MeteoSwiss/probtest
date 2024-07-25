import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.helpers import create_random_stats_file, generate_ensemble, load_pandas
from util.tree import TimingTree


@pytest.fixture(autouse=True, scope="session")
def new_ref() -> str:
    new_ref = os.path.join(tempfile.mkdtemp())
    print(f"\nNew reference data will be stored in {new_ref}")
    return new_ref


@pytest.fixture()
def ref_data() -> str:
    return "tests/data"


@pytest.fixture()
def timing_logfile(ref_data) -> str:
    return os.path.join(ref_data, "timing_example_1.txt")


@pytest.fixture()
def df_ref_performance(ref_data) -> TimingTree:
    return TimingTree.from_json(os.path.join(ref_data, "ref"))


@pytest.fixture(scope="module")
def tmp_dir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture(scope="module")
def ensemble(tmp_dir, nc_with_T_U_V) -> str:
    initial_condition = os.path.basename(nc_with_T_U_V)
    return generate_ensemble(tmp_dir, initial_condition, perturb_amplitude=10e-12)


@pytest.fixture(scope="module")
def too_small_ensemble(tmp_dir, nc_with_T_U_V) -> str:
    initial_condition = os.path.basename(nc_with_T_U_V)
    return generate_ensemble(tmp_dir, initial_condition, perturb_amplitude=10e-14)


@pytest.fixture()
def ds_ref_with_T_U_V(ref_data) -> xr.Dataset:
    data_ref = pd.read_csv(os.path.join(ref_data, "initial_condition.csv")).set_index(
        ["time", "lon", "lat"]
    )
    return xr.Dataset.from_dataframe(data_ref)


@pytest.fixture()
def ds_with_T_U_V(nc_with_T_U_V) -> xr.Dataset:
    return xr.load_dataset(nc_with_T_U_V)


@pytest.fixture()
def df_ref_stats(ref_data) -> pd.DataFrame:
    return load_pandas(
        os.path.join(ref_data, "ref_stats.csv"), index_col=[0, 1, 2], header=[0, 1]
    )


@pytest.fixture()
def df_ref_tolerance(ref_data) -> pd.DataFrame:
    return load_pandas(
        os.path.join(ref_data, "ref_tolerance.csv"), index_col=[0, 1], header=[0, 1]
    )


@pytest.fixture()
def df_ref_cdo_table(ref_data) -> pd.DataFrame:
    return load_pandas(
        os.path.join(ref_data, "ref_cdo_table.csv"), index_col=[0, 1], header=[0, 1]
    )


@pytest.fixture()
def df_ref_ensemble_stats(ref_data) -> dict:
    # dict comprehension for 1-10 members containing ref_data + stats_dp_number.csv
    return {
        i: load_pandas(
            os.path.join(ref_data, f"stats_dp_{i}.csv"),
            index_col=[0, 1, 2],
            header=[0, 1],
        )
        for i in range(1, 11)
    }


@pytest.fixture(scope="module")
def nc_with_T_U_V(tmp_dir) -> str:
    """
    Create a netcdf file with variables T, U and V.
    The variables are 3D with dimensions time, lat and lon.
    """
    # Define dimensions
    time = np.arange(0, 5)
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    # Create a meshgrid for lat and lon
    lon, lat = np.meshgrid(lon, lat)

    # Generate non-random data for variables T,V and U
    T = 20 + 5 * np.sin(
        np.pi * lat / 180
    )  # Temperature varies sinusoidally with latitude
    V = 100 * np.cos(np.pi * lon / 180)  # Velocity varies cosinusoidally with longitude
    U = 100 * np.sin(np.pi * lon / 180)  # Velocity varies sinusoidally with longitude

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "T": (
                ("time", "lat", "lon"),
                np.tile(T[np.newaxis, :, :], (len(time), 1, 1)),
            ),
            "V": (
                ("time", "lat", "lon"),
                np.tile(V[np.newaxis, :, :], (len(time), 1, 1)),
            ),
            "U": (
                ("time", "lat", "lon"),
                np.tile(U[np.newaxis, :, :], (len(time), 1, 1)),
            ),
        },
        coords={"time": time, "lat": ("lat", lat[:, 0]), "lon": ("lon", lon[0, :])},
    )
    # Save to netcdf
    filename = os.path.join(tmp_dir, "initial_condition.nc")
    ds.to_netcdf(filename)

    return filename


@pytest.fixture(scope="module")
def stats_file_set(tmp_dir):
    """
    Create a set of stats files for testing the selection of members
    For convenience also the filenames for the members and the tolerance
    are provided.
    """
    configurations = [
        {
            "time_dim": 3,
            "height_dim": 5,
            "variable": "v1",
            "file_format": "Format1:*test_3d*.nc",
        },
        {
            "time_dim": 3,
            "height_dim": 2,
            "variable": "v2",
            "file_format": "Format2:*test_2d*.nc",
        },
        {
            "time_dim": 2,
            "height_dim": 4,
            "variable": "v3",
            "file_format": "Format3:*test_2d*.nc",
        },
    ]
    seed = 42
    stats_pattern = os.path.join(tmp_dir, "stats_{member_id}.csv")
    create_random_stats_file(
        stats_pattern.format(member_id="ref"), configurations, seed - 1, 0.0
    )
    for i in range(1, 50):
        create_random_stats_file(
            stats_pattern.format(member_id=i), configurations, seed + i, 1e-3
        )
    create_random_stats_file(
        stats_pattern.format(member_id=50), configurations, seed + 50, 1e2
    )
    files = {}
    files["stats"] = stats_pattern
    files["members"] = os.path.join(tmp_dir, "selected_members.csv")
    # not possible to move to /tmp because of
    # [Errno 18] Invalid cross-device link from i.e. /scratch to /tmp
    files["tol"] = os.path.join("tolerance.csv")
    yield files
    if os.path.exists(files["tol"]):
        os.remove(files["tol"])
