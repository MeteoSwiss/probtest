import click

from util.click_util import cli_help
from util.constants import CHECK_THRESHOLD
from util.dataframe_ops import (
    compute_div_dataframe,
    compute_rel_diff_dataframe,
    parse_probtest_csv,
)
from util.log_handler import logger


def check_variable(diff_df, df_tol):
    out = diff_df - df_tol

    selector = (out > CHECK_THRESHOLD).any(axis=1)

    return len(out[selector].index) == 0, diff_df[selector], df_tol[selector]


@click.command()
@click.option(
    "--input-file-ref",
    help=cli_help["input_file_ref"],
)
@click.option(
    "--input-file-cur",
    help=cli_help["input_file_cur"],
)
@click.option(
    "--tolerance-file-name",
    help=cli_help["tolerance_file_name"],
)
@click.option(
    "--factor",
    type=float,
    help=cli_help["factor"],
)
def check(input_file_ref, input_file_cur, tolerance_file_name, factor):
    df_tol = parse_probtest_csv(tolerance_file_name, index_col=[0, 1])

    logger.info("applying a factor of {} to the spread".format(factor))
    df_tol *= factor

    df_ref = parse_probtest_csv(input_file_ref, index_col=[0, 1, 2])
    df_cur = parse_probtest_csv(input_file_cur, index_col=[0, 1, 2])

    logger.info(
        "checking {} against {} using tolerances from {}".format(
            input_file_cur, input_file_ref, tolerance_file_name
        )
    )

    # compute relative difference
    diff_df = compute_rel_diff_dataframe(df_ref, df_cur)
    # take maximum over height
    diff_df = diff_df.groupby(["file_ID", "variable"]).max()

    out, err, tol = check_variable(diff_df, df_tol)

    div = compute_div_dataframe(err, tol)

    if out:
        logger.info("RESULT: check PASSED!")
    else:
        logger.info("RESULT: check FAILED")
        logger.info("Differences")
        print(err)
        logger.info("\nTolerance")
        print(tol)
        logger.info("\nError relative to tolerance")
        print(div)

    exit(0 if out else 1)
