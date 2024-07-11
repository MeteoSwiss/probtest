import os

import pytest

from tests.helpers.fixtures import (  # noqa: F401
    df_ref_ensemble_stats,
    df_ref_stats,
    nc_with_T_U_V,
    ref_data,
    tmp_dir,
)
from tests.helpers.helpers import (
    assert_empty_df,
    generate_ensemble,
    load_pandas,
    pandas_error,
    run_stats_cli,
)


def test_stats_cli_no_ensemble(nc_with_T_U_V, df_ref_stats):
    tmp_path = os.path.dirname(nc_with_T_U_V)
    stats_file = os.path.join(tmp_path, "stats_{member_id}.csv")
    run_stats_cli(tmp_path, stats_file, ensemble=False)
    df_test = load_pandas(
        stats_file.format(member_id="ref"), index_col=[0, 1, 2], header=[0, 1]
    )
    err = pandas_error(df_ref_stats, df_test)

    assert_empty_df(err, "Stats datasets are not equal!")


@pytest.mark.xfail(
    reason="perturb amplitude 10e-14 smaller than perturb amplitude 10e-12 of reference"
)
@pytest.mark.parametrize("member", range(1, 3))
def test_stats_cli_ensemble_with_too_small_perturb_amplitude_for_member(
    nc_with_T_U_V, df_ref_ensemble_stats, member
):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)
    exp_folder = generate_ensemble(
        tmp_path, initial_condition, perturb_amplitude=10e-14
    )
    stats_file = os.path.join(tmp_path, "stats_{member_id}.csv")
    run_stats_cli(tmp_path, stats_file, "--ensemble", exp_folder)
    df_test = load_pandas(
        stats_file.format(member_id="dp_" + str(member)),
        index_col=[0, 1, 2],
        header=[0, 1],
    )
    err = pandas_error(df_ref_ensemble_stats[member], df_test)

    assert_empty_df(err, "Stats datasets are not equal!")


@pytest.mark.parametrize("member", range(1, 11))
def test_stats_cli_ensemble_for_member(nc_with_T_U_V, df_ref_ensemble_stats, member):
    initial_condition = os.path.basename(nc_with_T_U_V)
    tmp_path = os.path.dirname(nc_with_T_U_V)
    exp_folder = generate_ensemble(tmp_path, initial_condition)
    stats_file = os.path.join(tmp_path, "stats_{member_id}.csv")
    run_stats_cli(
        tmp_path, stats_file, ensemble=True, perturbed_model_output_dir=exp_folder
    )
    df_test = load_pandas(
        stats_file.format(member_id="dp_" + str(member)),
        index_col=[0, 1, 2],
        header=[0, 1],
    )
    err = pandas_error(df_ref_ensemble_stats[member], df_test)

    assert_empty_df(err, "Stats datasets are not equal!")
