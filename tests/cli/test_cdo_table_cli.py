import os

from tests.helpers.fixtures import (  # noqa: F401
    df_ref_cdo_table,
    ref_data,
    test_with_T_U_V,
    tmp_dir,
)
from tests.helpers.helpers import (
    assert_empty_df,
    generate_ensemble,
    load_pandas,
    pandas_error,
    run_cdo_table_cli,
)


def test_cdo_table_cli(test_with_T_U_V, df_ref_cdo_table):
    initial_condition = os.path.basename(test_with_T_U_V)
    tmp_path = os.path.dirname(test_with_T_U_V)
    exp_folder = generate_ensemble(
        tmp_path, initial_condition, perturb_amplitude=10e-14
    )
    cdo_table_file = os.path.join(tmp_path, "cdo_table.csv")
    run_cdo_table_cli(tmp_path, cdo_table_file, exp_folder)
    df_test = load_pandas(cdo_table_file, index_col=[0, 1])
    err = pandas_error(df_ref_cdo_table, df_test)

    assert_empty_df(err, "CDO table datasets are not equal!")
