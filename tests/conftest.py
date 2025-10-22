"""
Test Fixtures for Performance and Perturbation Testing

This module defines various pytest fixtures for setting up test environments,
data files, and configurations required for performance testing and perturbation
experiments.
These fixtures create necessary temporary files and directories, generate test
data, and provide setup and teardown mechanisms for efficient testing.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import random
import string

from tests.helpers import create_random_stats_file, generate_ensemble, load_pandas
from util.tree import TimingTree


@pytest.fixture(autouse=True, scope="session")
def new_ref() -> str:
    ref = os.path.join(tempfile.mkdtemp())
    print(f"\nNew reference data will be stored in {ref}")
    return ref


@pytest.fixture(name="ref_data")
def fixture_ref_data() -> str:
    return "tests/data"


@pytest.fixture(name="timing_logfile")
def fixture_timing_logfile(ref_data) -> str:
    return os.path.join(ref_data, "timing_example_1.txt")


@pytest.fixture()
def df_ref_performance(ref_data) -> TimingTree:
    return TimingTree.from_json(os.path.join(ref_data, "ref"))


@pytest.fixture(name="tmp_dir", scope="module")
def fixture_tmp_dir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture(scope="module")
def ensemble(tmp_dir, nc_with_t_u_v) -> str:
    initial_condition = os.path.basename(nc_with_t_u_v)
    return generate_ensemble(tmp_dir, initial_condition, perturb_amplitude=10e-12)


@pytest.fixture(scope="module")
def too_small_ensemble(tmp_dir, nc_with_t_u_v) -> str:
    initial_condition = os.path.basename(nc_with_t_u_v)
    return generate_ensemble(tmp_dir, initial_condition, perturb_amplitude=10e-14)


@pytest.fixture()
def ds_ref_with_t_u_v(ref_data) -> xr.Dataset:
    data_ref = pd.read_csv(os.path.join(ref_data, "initial_condition.csv")).set_index(
        ["time", "lat", "lon"]
    )
    return xr.Dataset.from_dataframe(data_ref)


@pytest.fixture()
def ds_with_t_u_v(nc_with_t_u_v) -> xr.Dataset:
    return xr.load_dataset(nc_with_t_u_v)


@pytest.fixture()
def df_ref_stats(ref_data) -> pd.DataFrame:
    return load_pandas(
        os.path.join(ref_data, "ref_stats.csv"), index_col=[0, 1, 2], header=[0, 1]
    )


@pytest.fixture()
def df_ref_tolerance(ref_data) -> pd.DataFrame:
    return load_pandas(
        os.path.join(ref_data, "tolerance.csv"), index_col=[0, 1], header=[0, 1]
    )


@pytest.fixture()
def df_ref_cdo_table(ref_data) -> pd.DataFrame:
    return load_pandas(
        os.path.join(ref_data, "cdo_table.csv"), index_col=[0, 1], header=[0, 1]
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


@pytest.fixture(name="nc_with_t_u_v", scope="module")
def fixture_nc_with_t_u_v(tmp_dir) -> str:
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
    t = 20 + 5 * np.sin(
        np.pi * lat / 180
    )  # Temperature varies sinusoidally with latitude
    v = 100 * np.cos(np.pi * lon / 180)  # Velocity varies cosinusoidally with longitude
    u = 100 * np.sin(np.pi * lon / 180)  # Velocity varies sinusoidally with longitude

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "T": (
                ("time", "lat", "lon"),
                np.tile(t[np.newaxis, :, :], (len(time), 1, 1)).astype("float64"),
            ),
            "V": (
                ("time", "lat", "lon"),
                np.tile(v[np.newaxis, :, :], (len(time), 1, 1)).astype("float64"),
            ),
            "U": (
                ("time", "lat", "lon"),
                np.tile(u[np.newaxis, :, :], (len(time), 1, 1)).astype("float64"),
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
        stats_pattern.format(member_id="ref"), configurations, seed, 0.0
    )
    for i in range(1, 21):
        amp = 1.0e-3
        if i in {4, 9, 15}:
            amp = 1.0e1
        create_random_stats_file(
            stats_pattern.format(member_id=i), configurations, i, amp
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


@pytest.fixture(name="setup_csv_files")
def fixture_setup_csv_files(tmp_path):
    # Create sample CSV files for testing
    tolerance_data = pd.DataFrame(
        {"A": [0.1, 0.2], "B": [0.3, 0.4]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b"), ("c", "d")], names=["col1", "col2"]
        ),
    )
    ref_data = pd.DataFrame(
        {"A": [1, 2], "B": [3, 4]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b", "c"), ("d", "e", "f")], names=["col1", "col2", "col3"]
        ),
    )
    cur_data = pd.DataFrame(
        {"A": [2, 3], "B": [4, 5]},
        index=pd.MultiIndex.from_tuples(
            [("a", "b", "c"), ("d", "e", "f")], names=["col1", "col2", "col3"]
        ),
    )

    tolerance_file = tmp_path / "tolerance_test.csv"
    ref_file = tmp_path / "input_ref_test.csv"
    cur_file = tmp_path / "input_cur_test.csv"

    tolerance_data.to_csv(tolerance_file)
    ref_data.to_csv(ref_file)
    cur_data.to_csv(cur_file)

    return {
        "tolerance_file": tolerance_file,
        "ref_file": ref_file,
        "cur_file": cur_file,
    }

def create_random_sample_dataset(seed=None):
    n_hdr_size = 5
    n_body_size = 6

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # --- Header-level variables ---
    lat = np.random.uniform(0, 90, size=n_hdr_size)
    lon = np.random.uniform(0, 180, size=n_hdr_size)
    statid = ["".join(random.choices(string.ascii_lowercase, k=1)) for _ in range(n_hdr_size)]
    time_nomi = np.random.choice([0, 30, 60, 90], size=n_hdr_size)
    codetype = np.full(n_hdr_size, 5)
    lbody = np.random.randint(1, 3, size=n_hdr_size)
    ibody = np.arange(1, n_hdr_size + 1)

    # --- Body-level variables ---
    varno = np.random.randint(1, 10, size=n_body_size)
    level = np.linspace(1000, 700, n_body_size)
    veri_data = np.random.uniform(0, 100, size=n_body_size)
    obs = np.random.rand(n_body_size)
    bcor = np.random.rand(n_body_size)
    level_typ = np.random.rand(n_body_size)
    level_sig = np.random.rand(n_body_size)
    state = np.random.rand(n_body_size)
    flags = np.random.rand(n_body_size)
    check = np.random.rand(n_body_size)
    e_o = np.random.rand(n_body_size)
    qual = np.random.rand(n_body_size)
    plevel = np.random.rand(n_body_size)

    # --- Assemble dataset ---
    data = xr.Dataset(
        {
            "lat": (("d_hdr",), lat),
            "lon": (("d_hdr",), lon),
            "statid": (("d_hdr",), statid),
            "time_nomi": (("d_hdr",), time_nomi),
            "codetype": (("d_hdr",), codetype),
            "l_body": (("d_hdr",), lbody),
            "i_body": (("d_hdr",), ibody),

            "varno": (("d_body",), varno),
            "level": (("d_body",), level),
            "veri_data": (("d_body",), veri_data),
            "obs": (("d_body",), obs),
            "bcor": (("d_body",), bcor),
            "level_typ": (("d_body",), level_typ),
            "level_sig": (("d_body",), level_sig),
            "state": (("d_body",), state),
            "flags": (("d_body",), flags),
            "check": (("d_body",), check),
            "e_o": (("d_body",), e_o),
            "qual": (("d_body",), qual),
            "plevel": (("d_body",), plevel),
        },
        coords={
            "d_hdr": np.arange(n_hdr_size),
            "d_body": np.arange(n_body_size),
        },
        attrs={"n_hdr": n_hdr_size, "n_body": n_body_size},
    )


    return data


@pytest.fixture(scope="module")
def fof_file_set(tmp_dir):
    fof_types = ["AIREP", "PILOT"]  # i tipi di FOF da creare
    seed = 42
    files = {}

    fof_patterns = []
    tol_patterns = []

    print("FFFF")

    for fof_type in fof_types:
        stats_pattern = os.path.join(tmp_dir, f"fof{fof_type}_{{member_id}}.nc")
        fof_patterns.append(stats_pattern)


        # crea il file di riferimento#
        ds_ref = create_random_sample_dataset(seed=seed)

        ds_ref.to_netcdf(stats_pattern.format(member_id="ref"))



    #     # genera 20 membri con piccole variazioni
        for i in range(1, 11):
            amp = 1.0e-3
            if i in {4, 9, 15}:
                amp = 1.0e1

            ds = create_random_sample_dataset(seed=i)

            # applica rumore solo a "veri_data"
            if "veri_data" in ds.data_vars:
                noise = np.random.normal(0, amp, ds["veri_data"].shape)
                ds["veri_data"] = ds["veri_data"] * (1 + noise)

            ds.to_netcdf(stats_pattern.format(member_id=i))
            print(ds)

        tol_file = os.path.join(tmp_dir, f"tolerance_{fof_type}.csv")
        with open(tol_file, "w") as f:
            f.write("variable,tolerance\nveri_data,0.01\n")
        tol_patterns.append(tol_file)

    #print(tol_patterns)
    files["fof"] = fof_patterns
    files["members"] = os.path.join(tmp_dir, "selected_members.csv")
    files["tol"] = tol_patterns


    yield files

    # cleanup
    if os.path.exists(files["tol"]):
        os.remove(files["tol"])
