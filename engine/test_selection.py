import click
from util.click_util import CommaSeperatedInts, cli_help
from engine.check import check_intersection, check_variable
from util.dataframe_ops import parse_probtest_csv, compute_rel_diff_dataframe, compute_div_dataframe
from util.log_handler import logger


@click.command()
@click.option(
    "--stats-file-name",
    help=cli_help["stats_file_name"],
)
@click.option(
    "--tolerance-file-name",
    help=cli_help["tolerance_file_name"],
)
@click.option(
    "--member-num",
    type=CommaSeperatedInts(),
    default="10",
    help=cli_help["member_num"],
)
@click.option(
    "--member-type",
    type=str,
    default="",
    help=cli_help["member_type"],
)
@click.option(
    "--factor",
    type=float,
    help=cli_help["factor"],
)


def test_selection(stats_file_name, tolerance_file_name, member_num, member_type, factor):

    if len(member_num) == 1:
        member_num = [i for i in range(1, member_num[0] + 1)]

    input_file_ref = stats_file_name.format(member_id="ref")
    passed = 0
    df_tol = parse_probtest_csv(tolerance_file_name, index_col=[0, 1])
    df_tol *= factor
    df_ref = parse_probtest_csv(input_file_ref, index_col=[0, 1, 2])

    for m_num in member_num:
        m_id = (str(m_num) if not member_type else member_type + "_" + str(m_num))

        df_cur = parse_probtest_csv(stats_file_name.format(member_id=m_id), index_col=[0, 1, 2])

        # check if variables are available in reference file
        skip_test, df_ref, df_cur = check_intersection(df_ref, df_cur)
        if skip_test:  # No intersection
            logger.info(
                "ERROR: No intersection between variables in input and reference file."
            )
            exit(1)

        # compute relative difference
        diff_df = compute_rel_diff_dataframe(df_ref, df_cur)
        # take maximum over height
        diff_df = diff_df.groupby(["file_ID", "variable"]).max()

        out, err, tol = check_variable(diff_df, df_tol)

        div = compute_div_dataframe(err, tol)

        if out:
            passed = passed + 1

    logger.info("The tolerance test passed for {} out of {} references.".format(passed, len(member_num)))
    return
