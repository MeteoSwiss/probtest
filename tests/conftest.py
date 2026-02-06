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
def df_ref_tolerance_fof():
    index = [0, 1, 2, 3, 4, 5]
    values = [0, 1, 1e-3, 3, 0.5, 0]
    df = pd.DataFrame({"value": values}, index=index)
    return df


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


@pytest.fixture(name="sample_dataset_fof")
def fixture_sample_dataset_fof():
    n_hdr_size = 5
    n_body_size = 6

    lat = [1, 3, 2, 4, 5]
    lon = [5, 9, 7, 8, 3]
    varno = [3, 3, 4, 4, 4, 4]
    statid = ["a", "b", "c", "d", "e"]
    time_nomi = [0, 30, 0, 30, 60]
    codetype = [5, 5, 5, 5, 5]
    lbody = np.array([1, 1, 1, 1, 2])
    ibody = [1, 2, 3, 4, 5]
    level = [1000, 950, 900, 850, 800, 750]
    veri_data = [45, 34, 45, 56, 67, 78]
    obs = np.array([0.374, 0.950, 0.731, 0.598, 0.156, 0.155])
    bcor = np.array([0.058, 0.866, 0.601, 0.708, 0.020, 0.969])
    level_typ = np.array([0.832, 0.212, 0.181, 0.183, 0.304, 0.524])
    level_sig = np.array([0.431, 0.291, 0.611, 0.139, 0.292, 0.366])
    state = np.array([1, 1, 1, 1, 1, 1])
    flags = np.array([9, 9, 9, 9, 9, 9])
    check = np.array([13, 13, 13, 13, 13, 13])
    e_o = np.array([0.034, 0.909, 0.258, 0.662, 0.311, 0.520])
    qual = np.array([0.796, 0.509, 0.810, 0.163, 0.425, 0.138])
    plevel = np.array([0.801, 0.406, 0.077, 0.847, 0.320, 0.755])

    data = xr.Dataset(
        {
            "lat": (("d_hdr",), lat),
            "lon": (("d_hdr",), lon),
            "varno": (("d_body",), varno),
            "statid": (("d_hdr",), statid),
            "time_nomi": (("d_hdr",), time_nomi),
            "codetype": (("d_hdr",), codetype),
            "level": (("d_body",), level),
            "l_body": (("d_hdr",), lbody),
            "i_body": (("d_hdr",), ibody),
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
        coords={"d_hdr": np.arange(n_hdr_size), "d_body": np.arange(n_body_size)},
        attrs={"n_hdr": n_hdr_size, "n_body": n_body_size, "plevel": 4},
    )

    return data


@pytest.fixture(scope="function")
def fof_file_set(tmp_dir, sample_dataset_fof):
    """
    Create a set of FOF files from sample_dataset_fof,
    modifying only veri_data and returning the actual paths
    of the files created (ref + members).
    """

    fof_types = ["AIREP"]
    files = {"fof": [], "tol": [], "path": []}

    veri_data_sets = [
        [45, 34, 45 + 1e-3, 56, 67, 78],
        [45, 34, 45 + 1e-3, 57, 67, 78.5],
        [45, 34, 45 + 1e-4, 58, 67, 78.4],
        [45, 35, 45 + 1e-6, 59, 67, 78.1],
    ]

    for fof_type in fof_types:
        fof_files_for_type = []

        ref_path = os.path.join(tmp_dir, f"fof{fof_type}_.nc")
        files["path"] = os.path.join(tmp_dir, f"fof{fof_type}_{{member_id}}.nc")
        ds_ref = sample_dataset_fof.copy(deep=True)
        ds_ref.to_netcdf(ref_path)
        fof_files_for_type.append(ref_path)

        for i, veri_data_values in enumerate(veri_data_sets, start=1):
            ds_member = sample_dataset_fof.copy(deep=True)
            ds_member["veri_data"] = (
                ("d_body",),
                veri_data_values[: ds_member.sizes["d_body"]],
            )

            member_path = os.path.join(tmp_dir, f"fof{fof_type}_{i}.nc")
            ds_member.to_netcdf(member_path)
            fof_files_for_type.append(member_path)

        files["fof"].extend(fof_files_for_type)

        tol_file = os.path.join(tmp_dir, f"tolerance_{fof_type}.csv")
        with open(tol_file, "w", encoding="utf-8") as f:
            f.write("variable,tolerance\nveri_data,0.01\n")
        files["tol"].append(tol_file)

    members_csv = os.path.join(tmp_dir, "selected_members.csv")
    with open(members_csv, "w", encoding="utf-8") as f:
        f.write("member_id\n1\n2\n3\n4\n")
    files["members"] = members_csv

    yield files

    for path in files["fof"] + files["tol"] + [files["members"]]:
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(name="stats_dataframes", scope="function")
def fixture_dataframes(tmp_dir):
    """
    Create stats dataframes and reference tolerances.
    """

    file_ids = ["NetCDF:*atm_3d*.nc"]
    variables = ["var_1", "var_2"]
    times = range(4)
    stats = ["max", "mean", "min"]

    index = pd.MultiIndex.from_arrays(
        [file_ids * 4, ["var_1"] * 3 + ["var_2"], list(range(3)) + [0]],
        names=["file_ID", "variable", "height"],
    )
    columns = pd.MultiIndex.from_product([times, stats], names=["time", "statistic"])

    array1 = np.linspace(0.9, 1.9, 4 * 12).reshape(4, 12)
    array1[:2] *= -1
    df1 = pd.DataFrame(array1, index=index, columns=columns)

    array2 = np.linspace(1.1, 2.1, 4 * 12).reshape(4, 12)
    array2[:2] *= -1
    df2 = pd.DataFrame(array2, index=index, columns=columns)

    base_index = pd.Index(variables, name="variable")
    tol_large_base = pd.DataFrame(
        np.ones((2, 12)) * 0.21, index=base_index, columns=columns
    )
    tol_small_base = pd.DataFrame(
        np.ones((2, 12)) * 0.06, index=base_index, columns=columns
    )

    tol_large = pd.concat(
        [tol_large_base] * len(file_ids), keys=file_ids, names=["file_ID", "variable"]
    )
    tol_small = pd.concat(
        [tol_small_base] * len(file_ids), keys=file_ids, names=["file_ID", "variable"]
    )

    df1_stats = os.path.join(tmp_dir, "stats1.csv")
    df2_stats = os.path.join(tmp_dir, "stats2.csv")
    tol_large_stats = os.path.join(tmp_dir, "tol_large.csv")
    tol_small_stats = os.path.join(tmp_dir, "tol_small.csv")

    df1.to_csv(df1_stats)
    df2.to_csv(df2_stats)
    tol_large.to_csv(tol_large_stats)
    tol_small.to_csv(tol_small_stats)

    yield df1_stats, df2_stats, tol_large_stats, tol_small_stats
