import click
import pandas as pd

from util.click_util import CommaSeperatedInts, CommaSeperatedStrings, cli_help
from util.dataframe_ops import parse_probtest_csv

pd.set_option("expand_frame_repr", False)


@click.command()
@click.option(
    "--variables",
    type=CommaSeperatedStrings(),
    help=cli_help["variables"],
)
@click.option(
    "--file-ids",
    type=CommaSeperatedStrings(),
    help=cli_help["file_ids"],
)
@click.option(
    "--times",
    type=CommaSeperatedInts(),
    help=cli_help["times"],
)
@click.option(
    "--histogram/--no-histogram",
    is_flag=True,
    help=cli_help["histogram"],
)
@click.option(
    "--cdo-table-file",
    help=cli_help["cdo_table_file"],
)
def cdo_table_reader(variables, file_ids, times, histogram, cdo_table_file):
    variables = variables if variables else slice(None)
    file_ids = file_ids if file_ids else slice(None)
    times = times if times else slice(None)
    hist = slice(None) if histogram else "rel_diff"

    df = parse_probtest_csv(cdo_table_file, index_col=[0, 1])
    sub_df = df.loc[(file_ids, variables), (times, hist)]

    print(sub_df)
