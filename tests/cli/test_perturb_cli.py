import os

import pytest

from engine.perturb import perturb
from tests.helpers.fixtures import (  # noqa: F401
    ds_ref_with_T_U_V,
    ds_with_T_U_V,
    nc_with_T_U_V,
    ref_data,
    tmp_dir,
)
from tests.helpers.helpers import assert_empty_list, check_netcdf, load_netcdf, run_cli


def run_perturb_cli(base_dir, filename, perturb_amplitude):
    args = [
        "--model-input-dir",
        base_dir,
        "--perturbed-model-input-dir",
        f"{base_dir}/experiments/" + "{member_id}",
        "--files",
        filename,
        "--member-num",
        "1",
        "--member-type",
        "dp",
        "--variable-names",
        "U,V",
        "--perturb-amplitude",
        f"{perturb_amplitude}",
        "--no-copy-all-files",
    ]
    run_cli(perturb, args)


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

    run_perturb_cli(tmp_path, initial_condition, perturb_amplitude=0.0)

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

    run_perturb_cli(tmp_path, initial_condition, perturb_amplitude=0.2)

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
