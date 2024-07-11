import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests.helpers.helpers import generate_ensemble


@pytest.fixture()
def ref_data():
    return "tests/data"


@pytest.fixture(scope="module")
def tmp_dir():
    return tempfile.mkdtemp()


@pytest.fixture(scope="module")
def ensemble(tmp_dir, nc_with_T_U_V):
    initial_condition = os.path.basename(nc_with_T_U_V)
    return generate_ensemble(tmp_dir, initial_condition, perturb_amplitude=10e-12)


@pytest.fixture(scope="module")
def wrong_ensemble(tmp_dir, nc_with_T_U_V):
    initial_condition = os.path.basename(nc_with_T_U_V)
    return generate_ensemble(tmp_dir, initial_condition, perturb_amplitude=10e-14)


@pytest.fixture()
def ds_ref_with_T_U_V(ref_data):
    data_ref = pd.read_csv(os.path.join(ref_data, "initial_condition.csv")).set_index(
        ["time", "lon", "lat"]
    )
    return xr.Dataset.from_dataframe(data_ref)


@pytest.fixture()
def ds_with_T_U_V(nc_with_T_U_V):
    return xr.load_dataset(nc_with_T_U_V)


@pytest.fixture()
def df_ref_stats(ref_data):
    return pd.read_csv(
        os.path.join(ref_data, "ref_stats.csv"), index_col=[0, 1, 2], header=[0, 1]
    )


@pytest.fixture()
def df_ref_tolerance(ref_data):
    return pd.read_csv(
        os.path.join(ref_data, "ref_tolerance.csv"), index_col=[0, 1], header=[0, 1]
    )


@pytest.fixture()
def df_ref_cdo_table(ref_data):
    return pd.read_csv(
        os.path.join(ref_data, "ref_cdo_table.csv"), index_col=[0, 1], header=[0, 1]
    )


@pytest.fixture()
def df_ref_ensemble_stats(ref_data):
    # dict comprehension for 1-10 members containing ref_data + stats_dp_number.csv
    return {
        i: pd.read_csv(
            os.path.join(ref_data, f"stats_dp_{i}.csv"),
            index_col=[0, 1, 2],
            header=[0, 1],
        )
        for i in range(1, 11)
    }


@pytest.fixture(scope="module")
def nc_with_T_U_V(tmp_dir) -> str:
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
