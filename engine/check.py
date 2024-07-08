import click

from util.click_util import cli_help
from util.dataframe_ops import compute_div_dataframe, tolerance_test
from util.log_handler import logger


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

    out, err, tol = tolerance_test(
        tolerance_file_name, input_file_ref, input_file_cur, factor
    )

    div = compute_div_dataframe(err, tol)

    if out:
        logger.info("RESULT: check PASSED!")
    else:
        logger.info("RESULT: check FAILED")
        logger.info("Differences")
        logger.info(err)
        logger.info("\nTolerance")
        logger.info(tol)
        logger.info("\nError relative to tolerance")
        logger.info(div)

    exit(0 if out else 1)
