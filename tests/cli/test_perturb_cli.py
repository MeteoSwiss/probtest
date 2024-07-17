import os

import pytest

from tests.helpers.helpers import (
    assert_empty_list,
    check_netcdf,
    load_netcdf,
    run_perturb_cli,
)


@pytest.mark.xfail
def test_compare_ref_with_data_from_fixture_V_missing(ds_ref_with_T_U_V, ds_with_T_U_V):
    # Remove V from the reference dataset so test fails
    ds_with_T_U_V = ds_with_T_U_V.drop_vars("V")

    diff_keys, _ = check_netcdf(ds_ref_with_T_U_V, ds_with_T_U_V)

    assert_empty_list(
        diff_keys, "The following variables are not contained in both files"
    )


def test_compare_ref_with_data_from_fixture(ds_ref_with_T_U_V, ds_with_T_U_V):
    diff_keys, err = check_netcdf(ds_ref_with_T_U_V, ds_with_T_U_V)

    assert_empty_list(err, "The following variables contain errors")
    assert_empty_list(
        diff_keys, "The following variables are not contained in both files"
    )


def test_perturb_cli_amplitude_0_0(nc_with_T_U_V, ds_ref_with_T_U_V):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)

    run_perturb_cli(tmp_path, initial_condition, member_num=1, perturb_amplitude=0.0)

    data_test = load_netcdf(
        os.path.join(tmp_path, "experiments/dp_1/initial_condition.nc")
    )

    diff_keys, err = check_netcdf(ds_ref_with_T_U_V, data_test)

    assert_empty_list(err, "The following variables contain errors")
    assert_empty_list(
        diff_keys, "The following variables are not contained in both files"
    )


def test_perturb_cli_amplitude_0_2(nc_with_T_U_V, ds_ref_with_T_U_V):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)

    run_perturb_cli(tmp_path, initial_condition, member_num=1, perturb_amplitude=0.2)

    data_test = load_netcdf(
        os.path.join(tmp_path, "experiments/dp_1/initial_condition.nc")
    )

    diff_keys, err = check_netcdf(ds_ref_with_T_U_V, data_test)

    # Remove U and V from the list of variables with errors because they are perturbed
    err.remove("U")
    err.remove("V")

    assert_empty_list(err, "The following variables contain errors")
    assert_empty_list(
        diff_keys, "The following variables are not contained in both files"
    )
