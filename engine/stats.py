"""
CLI for computing stats

This command line tool provides functionality for:
- Creating and saving statistics dataframes from specified model output files.
- Generating statistics for both ensemble and reference model runs.
"""

import os
from multiprocessing import Pool
from pathlib import Path

import click

from util.click_util import CommaSeperatedInts, cli_help
from util.dataframe_ops import df_from_file_ids
from util.log_handler import logger
from util.utils import prepend_type_to_member_id


def create_stats_dataframe(input_dir, file_id, stats_file_name, file_specification):
    df = df_from_file_ids(file_id, input_dir, file_specification)

    logger.info("writing stats file to %s", stats_file_name)

    Path(stats_file_name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stats_file_name)

    return df


@click.command()
@click.option(
    "--ensemble/--no-ensemble",
    is_flag=True,
    help=cli_help["ensemble"],
)
@click.option(
    "--stats-file-name",
    help=cli_help["stats_file_name"],
)
@click.option(
    "--model-output-dir",
    help=cli_help["model_output_dir"],
)
@click.option(
    "--file-id",
    nargs=2,
    type=str,
    multiple=True,
    metavar="FILE_TYPE FILE_PATTERN",
    help=cli_help["file_id"],
)
@click.option(
    "--member-ids",
    type=CommaSeperatedInts(),
    default="1,2,3,4,5,6,7,8,9,10",
    help=cli_help["member_ids"],
)
@click.option(
    "--member-type",
    type=str,
    default="",
    help=cli_help["member_type"],
)
@click.option(
    "--perturbed-model-output-dir",
    help=cli_help["perturbed_model_output_dir"],
)
@click.option(
    "--file-specification",
    type=list,
    help=cli_help["file_specification"],
)
def stats(
    ensemble,
    stats_file_name,
    model_output_dir,
    file_id,
    member_ids,
    member_type,
    perturbed_model_output_dir,
    file_specification,
):  # pylint: disable=too-many-positional-arguments
    file_specification = file_specification[0]  # can't store dicts as defaults in click
    assert isinstance(file_specification, dict), "must be dict"

    # compute stats for the ensemble and the reference run
    if ensemble:
        df_args = []
        
        member_ids.append(0)
        for member_id in member_ids:
            if member_id == 0:
                complete_member_id = "ref"
                output_dir = model_output_dir
            else:
                complete_member_id = prepend_type_to_member_id(member_type, member_id)
                output_dir = perturbed_model_output_dir.format(member_id=complete_member_id)
            
            df_args.append(
                (
                    output_dir,
                    file_id,
                    stats_file_name.format(member_id=complete_member_id),
                    file_specification
                )
            )
            
        with Pool() as p:
            p.starmap(create_stats_dataframe, df_args)

    else:
        create_stats_dataframe(
            model_output_dir,
            file_id,
            stats_file_name.format(member_id=os.path.basename(model_output_dir)),
            file_specification,
        )
