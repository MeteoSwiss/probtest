import os

import pytest

from tests.helpers.fixtures import (  # noqa: F401
    df_ref_cdo_table,
    ensemble,
    nc_with_T_U_V,
    ref_data,
    tmp_dir,
    wrong_ensemble,
)
from tests.helpers.helpers import (
    assert_empty_df,
    load_pandas,
    pandas_error,
    run_cdo_table_cli,
)


def test_cdo_table_cli(tmp_dir, wrong_ensemble, df_ref_cdo_table):
    cdo_table_file = os.path.join(tmp_dir, "cdo_table.csv")
    run_cdo_table_cli(tmp_dir, cdo_table_file, wrong_ensemble)
    df_test = load_pandas(cdo_table_file, index_col=[0, 1])
    err = pandas_error(df_ref_cdo_table, df_test)

    assert_empty_df(err, "CDO table datasets are not equal!")


@pytest.mark.xfail(
    reason="perturb amplitude 10e-14 < 10e-12",
    strict=True,
)
def test_cdo_table_cli_with_too_small_perturb_amplitude(
    tmp_dir, ensemble, df_ref_cdo_table
):
    cdo_table_file = os.path.join(tmp_dir, "cdo_table.csv")
    run_cdo_table_cli(tmp_dir, cdo_table_file, ensemble)
    df_test = load_pandas(cdo_table_file, index_col=[0, 1])
    err = pandas_error(df_ref_cdo_table, df_test)

    assert_empty_df(err, "CDO table datasets are not equal!")
