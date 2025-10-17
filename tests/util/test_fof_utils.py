"""
This module contains unit tests for the `util/fof_utils.py` module.
"""

import numpy as np
import pytest
import xarray as xr

from util.fof_utils import (
    FileType,
    clean_value,
    compare_arrays,
    compare_var_and_attr_ds,
    expand_zip,
    get_file_type,
    get_observation_variables,
    get_report_variables,
    prepare_array,
    primary_check,
    print_entire_line,
    split_feedback_dataset,
    to_list,
    value_list,
    write_lines,
)


@pytest.fixture(name="ds1", scope="function")
def sample_dataset():
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
    state = np.array([0.456, 0.785, 0.199, 0.514, 0.592, 0.046])
    flags = np.array([0.607, 0.170, 0.065, 0.948, 0.965, 0.808])
    check = np.array([0.304, 0.097, 0.684, 0.440, 0.122, 0.495])
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


def test_get_report_variables(ds1):
    """
    Test that one gets the variable names of reports.
    """
    variables = get_report_variables(ds1)

    assert variables == [
        "lat",
        "lon",
        "statid",
        "time_nomi",
        "codetype",
        "l_body",
        "i_body",
    ]


def test_get_observation_variable(ds1):
    """
    Test that one gets the variable names of variables.
    """
    variables = get_observation_variables(ds1)

    assert variables == [
        "varno",
        "level",
        "veri_data",
        "obs",
        "bcor",
        "level_typ",
        "level_sig",
        "state",
        "flags",
        "check",
        "e_o",
        "qual",
        "plevel",
    ]


@pytest.fixture(name="ds_report", scope="function")
def sample_dataset_report():
    n_hdr_size = 5
    n_body_size = 6

    lat = [1, 2, 3, 4, 5]
    lon = [5, 7, 9, 8, 3]
    statid = ["a", "c", "b", "d", "e"]
    time_nomi = [0, 0, 30, 30, 60]
    codetype = [5, 5, 5, 5, 5]
    l_body = [1, 1, 1, 1, 2]
    i_body = [1, 3, 2, 4, 5]

    data = xr.Dataset(
        {
            "lat": (("y",), lat),
            "lon": (("y",), lon),
            "statid": (("y",), statid),
            "time_nomi": (("y",), time_nomi),
            "codetype": (("y",), codetype),
            "l_body": (("y",), l_body),
            "i_body": (("y",), i_body),
        },
        coords={"d_hdr": np.arange(n_hdr_size)},
        attrs={"n_hdr": n_hdr_size, "n_body": n_body_size},
    )

    return data


@pytest.fixture(name="ds_obs", scope="function")
def sample_dataset_obs():
    n_hdr_size = 6
    n_body_size = 6

    lat = np.array([1, 2, 3, 4, 5, 5])
    lon = np.array([5, 7, 9, 8, 3, 3])
    statid = np.array(["a", "c", "b", "d", "e", "e"])
    time_nomi = np.array([0, 0, 30, 30, 60, 60])
    varno = np.array([3, 3, 4, 4, 4, 4])
    level = np.array([950, 1000, 750, 800, 850, 900])

    state = np.array([0.456, 0.785, 0.199, 0.514, 0.592, 0.046])
    flags = np.array([0.607, 0.170, 0.065, 0.948, 0.965, 0.808])
    check = np.array([0.304, 0.097, 0.684, 0.440, 0.122, 0.495])
    e_o = np.array([0.034, 0.909, 0.258, 0.662, 0.311, 0.520])
    qual = np.array([0.796, 0.509, 0.810, 0.163, 0.425, 0.138])
    plevel = np.array([0.801, 0.406, 0.077, 0.847, 0.320, 0.755])

    ds = xr.Dataset(
        {
            "lat": (("d_hdr",), lat),
            "lon": (("d_hdr",), lon),
            "varno": (("d_body",), varno),
            "statid": (("d_hdr",), statid),
            "time_nomi": (("d_hdr",), time_nomi),
            "level": (("d_body",), level),
            "state": (("d_body",), state),
            "flags": (("d_body",), flags),
            "check": (("d_body",), check),
            "e_o": (("d_body",), e_o),
            "qual": (("d_body",), qual),
            "plevel": (("d_body",), plevel),
        },
        coords={"d_hdr": np.arange(n_hdr_size), "d_body": np.arange(n_body_size)},
        attrs={"n_hdr": n_hdr_size, "n_body": n_body_size},
    )
    return ds


@pytest.fixture(name="ds_veri", scope="function")
def sample_dataset_veri():
    n_hdr_size = 6
    n_body_size = 6

    lat = np.array([1, 2, 3, 4, 5, 5])
    lon = np.array([5, 7, 9, 8, 3, 3])
    statid = np.array(["a", "c", "b", "d", "e", "e"])
    time_nomi = np.array([0, 0, 30, 30, 60, 60])
    varno = np.array([3, 3, 4, 4, 4, 4])
    level = np.array([950, 1000, 750, 800, 850, 900])
    veri_data = np.array([34, 45, 78, 67, 56, 45])

    ds = xr.Dataset(
        {
            "lat": (("d_hdr",), lat),
            "lon": (("d_hdr",), lon),
            "varno": (("d_body",), varno),
            "statid": (("d_hdr",), statid),
            "time_nomi": (("d_hdr",), time_nomi),
            "level": (("d_body",), level),
            "veri_data": (("d_body",), veri_data),
        },
        coords={
            "d_hdr": np.arange(n_hdr_size),
            "d_body": np.arange(n_body_size),
        },
        attrs={"n_hdr": n_hdr_size, "n_body": n_body_size},
    )

    return ds


def test_split_report(ds1, ds_report, ds_obs, ds_veri):
    """
    Test that the dataset is correctly split into reports, observations
    and veri data according to their dimensions.
    """
    reports, observations, veri_data = split_feedback_dataset(ds1)

    assert reports == ds_report and observations == ds_obs and veri_data == ds_veri


@pytest.fixture(name="arr1", scope="function")
def fixture_arr1():
    return np.array([1.0, 5.0, 3.0, 4.0, 7.0], dtype=np.float32)


@pytest.fixture(name="arr2", scope="function")
def fixture_arr2():
    return np.array([1.0, 5.0, 3.0, 4.0, 7.0], dtype=np.float32)


@pytest.fixture(name="arr3", scope="function")
def fixture_arr3():
    return np.array([23.0, 5.0, 3.0, 22.0, 7.0], dtype=np.float32)


@pytest.fixture(name="arr1_nan", scope="function")
def fixture_arr1_nan():
    return np.array([1.0, 5.0, 3.0, np.nan, 7.0], dtype=np.float32)


@pytest.fixture(name="arr2_nan", scope="function")
def fixture_arr2_nan():
    return np.array([1.0, 5.0, 3.0, np.nan, 7.0], dtype=np.float32)


def test_compare_array_equal(arr1, arr2, arr1_nan, arr2_nan):
    """
    Test that two arrays are considered equal if:
    - they have the same content
    - they have nan values in the same positions
    """
    total, equal, diff = compare_arrays(arr1, arr2, "var_name")
    total_nan, equal_nan, diff_nan = compare_arrays(arr1_nan, arr2_nan, "var_name")

    assert (total, equal, total_nan, equal_nan, diff.size, diff_nan.size) == (
        5,
        5,
        5,
        5,
        0,
        0,
    )


def test_compare_array_diff(arr1, arr3):
    """
    Test that if I compare two different arrays I get the number of total and equal
    vales and the number of the position where values are different."""
    total, equal, diff = compare_arrays(arr1, arr3, "var_name")

    assert (total, equal, diff.tolist()) == (5, 3, [0, 3])


@pytest.fixture(name="arr_nan", scope="function")
def fixture_arr():
    return np.array([1.0, np.nan, 3.0, 4.0, np.nan], dtype=np.float32)


def test_prepare_array_nan(arr_nan):
    """
    Test that if an array containign nan is given, these values are replaced
    by -9.99999e05.
    """
    array = prepare_array(arr_nan)
    expected = np.array([1.0, -9.99999e05, 3.0, 4.0, -9.99999e05], dtype=np.float32)
    assert np.array_equal(array, expected)


def test_prepare_array(arr1):
    """
    Test that if an array without nan is given, the output of the function
    is the same as the input.
    """
    array = prepare_array(arr1)
    assert np.array_equal(array, arr1)


@pytest.fixture(name="arr_with_spaces", scope="function")
def fixture_arr_with_spaces():
    return np.array([1.0, "ABF   '", 3.0, 4.0, b"ABF   '"], dtype=object)


def test_clean_value(arr_with_spaces):
    """
    Test that if values with unnecessary spaces are provided,
    they are removed and cleaned up.
    """
    cleaned = np.array([clean_value(x) for x in arr_with_spaces])
    expected = np.array(["1.0", "ABF", "3.0", "4.0", "ABF"])
    np.testing.assert_array_equal(cleaned, expected)


@pytest.fixture(name="ds2", scope="function")
def sample_dataset_2():
    n_hdr_size = 5
    n_body_size = 6

    lat = [1, 3, 2, 4, 5]
    lon = [5, 9, 7, 8, 3]
    varno = [3, 3, 4, 4, 4, 4]
    statid = ["a", "b", "c", "d", "e"]
    time_nomi = [0, 30, 0, 30, 60]
    codetype = [5, 5, 5, 5, 3]
    lbody = np.array([1, 1, 1, 1, 2])
    ibody = [1, 2, 3, 4, 5]
    level = [1000, 950, 900, 850, 800, 750]
    veri_data = [45, 34, 45, 56, 67, 78]
    obs = np.array([0.374, 0.950, 0.731, 0.598, 0.156, 0.155])
    bcor = np.array([0.058, 0.866, 0.601, 0.708, 0.020, 0.969])
    level_typ = np.array([0.832, 0.212, 0.181, 0.183, 0.304, 0.524])
    level_sig = np.array([0.431, 0.291, 0.611, 0.139, 0.292, 0.366])
    state = np.array([0.456, 0.785, 0.199, 0.514, 0.592, 0.046])
    flags = np.array([0.607, 0.170, 0.065, 0.948, 0.965, 0.808])
    check = np.array([0.304, 0.097, 0.684, 0.440, 0.122, 0.495])
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


def test_print_entire_line(ds1, ds2, capsys):
    """
    Test that in case of differences, these are printed correctly.
    """
    diff = np.array([5])
    print_entire_line(ds1, ds2, diff)
    captured = capsys.readouterr()
    output = captured.out.splitlines()

    assert output[0] == (
        "\x1b[1mid\x1b[0m  : d_hdr        |d_body       |lat          |lon          "
        "|varno        |statid       |time_nomi    |codetype     |level        "
        "|l_body       |i_body       |veri_data    |obs          |bcor         "
        "|level_typ    |level_sig    |state        |flags        |check        "
        "|e_o          |qual         |plevel       "
    )
    assert output[1] == (
        "\x1b[1mref\x1b[0m : 0            |5            |1            |5            "
        "|4            |a            |0            |5            |750          "
        "|1            |1            |78           |0.155        |0.969        "
        "|0.524        |0.366        |0.046        |0.808        |0.495        "
        "|0.52         |0.138        |0.755        "
    )
    assert output[2] == (
        "\x1b[1mcur\x1b[0m : 0            |5            |1            |5            "
        "|4            |a            |0            |5            |750          "
        "|1            |1            |78           |0.155        |0.969        "
        "|0.524        |0.366        |0.046        |0.808        |0.495        "
        "|0.52         |0.138        |0.755        "
    )
    assert output[3] == (
        "\x1b[1mdiff\x1b[0m: 0            |0            |0            |0            "
        "|0            |nan          |0            |0            |0            "
        "|0            |0            |0            |0.0          |0.0          "
        "|0.0          |0.0          |0.0          |0.0          |0.0          "
        "|0.0          |0.0          |0.0          "
    )


def test_write_lines(ds1, ds2, tmp_path):
    """
    Test that if there are any differences, they are saved in a separate csv file.
    """
    file_path = tmp_path / "differences.csv"
    diff = np.array([5])
    write_lines(ds1, ds2, diff, file_path)

    content = file_path.read_text(encoding="utf-8")

    expected = (
        "id  : d_hdr        |d_body       |lat          |lon          |varno        "
        "|statid       |time_nomi    |codetype     |level        |l_body       "
        "|i_body       |veri_data    |obs          |bcor         |level_typ    "
        "|level_sig    |state        |flags        |check        |e_o          "
        "|qual         |plevel       \n"
        "ref  : 0            |5            |1            |5            |4            "
        "|a            |0            |5            |750          |1            "
        "|1            |78           |0.155        |0.969        |0.524        "
        "|0.366        |0.046        |0.808        |0.495        |0.52         "
        "|0.138        |0.755        \n"
        "cur  : 0            |5            |1            |5            |4            "
        "|a            |0            |5            |750          |1            "
        "|1            |78           |0.155        |0.969        |0.524        "
        "|0.366        |0.046        |0.808        |0.495        |0.52         "
        "|0.138        |0.755        \n"
        "diff : 0            |0            |0            |0            |0            "
        "|nan          |0            |0            |0            |0            "
        "|0            |0            |0.0          |0.0          |0.0          "
        "|0.0          |0.0          |0.0          |0.0          |0.0          "
        "|0.0          |0.0          \n"
    )
    assert content == expected


def test_compare_var_and_attr_ds(ds1, ds2, tmp_path):
    """
    Test that, given two datasets, returns the number of elements in which
    the variables are the same and in which they differ.
    """

    file_path = tmp_path / "differences.csv"

    total1, equal1 = compare_var_and_attr_ds(
        ds1, ds2, nl=0, output=True, location=file_path
    )
    total2, equal2 = compare_var_and_attr_ds(ds1, ds2, nl=4, output=True, location=None)

    assert (total1, equal1) == (104, 103)
    assert (total2, equal2) == (104, 103)


@pytest.fixture(name="ds3", scope="function")
def sample_dataset_3():
    n_hdr_size = 5
    n_body_size = 5

    lat = [1, 3, 2, 4, 5]
    lon = [5, 9, 7, 8, 3]
    varno = [3, 3, 4, 4, 4]
    statid = ["a", "b", "c", "d", "e"]
    time_nomi = [0, 30, 0, 30, 60]
    codetype = [5, 5, 5, 5, 3]
    lbody = np.array([1, 1, 1, 1, 2])
    ibody = [1, 2, 3, 4, 5]
    level = [1000, 950, 900, 850, 800]
    veri_data = [45, 34, 45, 56, 67]
    obs = np.array([0.374, 0.950, 0.731, 0.598, 0.156])
    bcor = np.array([0.058, 0.866, 0.601, 0.708, 0.020])
    level_typ = np.array([0.832, 0.212, 0.181, 0.183, 0.304])
    level_sig = np.array([0.431, 0.291, 0.611, 0.139, 0.292])
    state = np.array([0.456, 0.785, 0.199, 0.514, 0.592])
    flags = np.array([0.607, 0.170, 0.065, 0.948, 0.965])
    check = np.array([0.304, 0.097, 0.684, 0.440, 0.122])
    e_o = np.array([0.034, 0.909, 0.258, 0.662, 0.311])
    qual = np.array([0.796, 0.509, 0.810, 0.163, 0.425])
    plevel = np.array([0.801, 0.406, 0.077, 0.847, 0.320])

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
        attrs={
            "n_hdr": n_hdr_size,
            "n_body": n_body_size,
            "plevel": np.array([0.374, 0.950, 0.731, 0.598, 0.156]),
        },
    )

    return data


def test_compare_var_and_attr_ds_different_attr(ds2, ds3):
    """
    Test that, given two datasets, returns the number of elements in which
    the attributes are the same and in which they differ.
    """

    total, equal = compare_var_and_attr_ds(ds2, ds3, nl=0, output=True, location=None)

    assert (total, equal) == (108, 25)


def test_value_list():
    """
    Test that the correct placeholder is given according to the different cases.
    """
    placeholders = {"fof": False}
    assert value_list("fof", [1, 2, 3], placeholders) == [""]
    placeholders = {"fof": True}
    assert value_list("fof", [1, 2, 3], placeholders) == [1, 2, 3]


def test_get_file_type(tmp_path):
    """
    Test that the file is recognized as FOF or STAT if the corresponding keyword
    is present in the file name; otherwise, an error will be raised.
    """

    test_file_fof = tmp_path / "fofexample.nc"
    str_fof = str(test_file_fof)
    file_type_fof = get_file_type(str_fof)

    test_file_stats = tmp_path / "statsexample.nc"
    str_stats = str(test_file_stats)
    file_type_stats = get_file_type(str_stats)

    with pytest.raises(ValueError):
        get_file_type("random_file.nc")

    assert file_type_fof == FileType.FOF
    assert file_type_stats == FileType.STATS


def test_primary_check(tmp_path):
    """
    Note that if two fof files are not of the same type, then the primary_check fails.
    """
    test_fof1 = tmp_path / "fofAIREP.nc"
    test_fof2 = tmp_path / "fofAIREP.nc"
    test_fof3 = tmp_path / "fofPILOT.nc"

    assert primary_check(test_fof1, test_fof2)

    false_result = primary_check(test_fof1, test_fof3)
    assert false_result is False


@pytest.fixture(name="none_value", scope="function")
def fixture_none_value():
    return


@pytest.fixture(name="string", scope="function")
def fixture_string():
    return "a,b,c,  ,e"


@pytest.fixture(name="simple_list", scope="function")
def fixture_list():
    return [1, 2, 3, 4]


def test_to_list_empty(none_value):
    """
    Test that the function returns an empty list if no value is given.
    """
    empty_return = to_list(none_value)
    assert empty_return == []


def test_to_list_full(string):
    """
    Test that if the function receives a string, it is split at each comma and
    unnecessary spaces are removed.
    """
    string_list_to_list = to_list(string)
    assert string_list_to_list == ["a", "b", "c", "e"]


def test_to_list_no_changes(simple_list):
    """
    Tests that if the function receives a value that is neither a string
    nor None, nothing changes.
    """
    to_list_no_changes = to_list(simple_list)
    assert to_list_no_changes == [1, 2, 3, 4]


def test_expand_zip():
    """
    Test that the zip is expanded correctly.
    """
    zipped = [
        "test_{fof_type}.nc",
        "test_{fof_type}_{member_id}.nc",
        "test_{member_id}.nc",
    ]
    fof_type = ["AIREP", "PILOT"]
    member_ids = [1, 2]
    expanded_zip1 = expand_zip(zipped, fof_type, member_ids, member_type=None)
    expanded_zip2 = expand_zip(zip(zipped), fof_type, member_ids, member_type=None)
    assert expanded_zip1, expanded_zip2 == [
        "test_AIREP.nc",
        "test_PILOT.nc",
        "test_AIREP_1.nc",
        "test_AIREP_2.nc",
        "test_PILOT_1.nc",
        "test_PILOT_2.nc",
        "test_1.nc",
        "test_2.nc",
    ]

    expanded_zip3 = expand_zip(
        "test_{fof_type}.nc", fof_type, member_ids, member_type=None
    )
    assert expanded_zip3 == ["test_AIREP.nc", "test_PILOT.nc"]
