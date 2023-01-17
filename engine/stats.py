import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

from util.click_util import CommaSeperatedStrings, cli_help
from util.constants import compute_statistics
from util.dataframe_ops import df_from_file_ids
from util.log_handler import logger
from util.xarray_ops import statistics_over_horizontal_dim


def monotonically_increasing(L):
    return all(x <= y for x, y in zip(L[:-1], L[1:]))


def dataframe_from_ncfile(
    file_id, filename, varname, time_dim, horizontal_dims, xarray_ds
):
    statistics = statistics_over_horizontal_dim(
        xarray_ds[varname], horizontal_dims, compute_statistics
    )

    first_stat = statistics[0]
    if len(first_stat.dims) == 2:
        height_name = (
            first_stat.dims[0] if first_stat.dims[0] != time_dim else first_stat.dims[1]
        )  # might be 'height', 'height_2', 'alt', 'plev', ...
        height = xarray_ds[height_name].values
        matrix = np.empty(
            (first_stat.shape[0] * len(statistics), first_stat.shape[1])
        )  # shape: (mean_max_min of each height, time)

        # weave mean max min into time dimension
        for i, stat in enumerate(statistics):
            matrix[i :: len(statistics), :] = stat.values
    elif len(first_stat.dims) == 1:
        height = np.array([-1])
        # matrix needs to have 2 dimensions for DataFrame constructor
        matrix = np.empty((statistics[0].size * len(statistics), 1))
        # weave mean max min into time dimension
        for i, stat in enumerate(statistics):
            matrix[i :: len(statistics), 0] = stat.values
    elif len(first_stat.dims) == 0:
        height = np.array([-1])
        # matrix needs to have 2 dimensions for DataFrame constructor
        matrix = np.empty((len(statistics), 1))
        # weave mean max min into time dimension
        for i, stat in enumerate(statistics):
            matrix[i :: len(statistics), 0] = stat.values
    else:
        logger.error(
            (
                "Unknown number of dimension for first_stat of variable '{}'. "
                + "Dims: {}"
            ).format(varname, first_stat.dims)
        )
        sys.exit(1)

    if "None" != time_dim:
        time = xarray_ds[time_dim].values
    else:
        time = [0]

    index = pd.MultiIndex.from_product(
        [[file_id], [varname], height], names=("file_ID", "variable", "height")
    )
    columns = pd.MultiIndex.from_product(
        [time, compute_statistics], names=("time", "statistic")
    )

    return pd.DataFrame(matrix.T, index=index, columns=columns)


def create_stats_dataframe(
    input_dir, file_ids, time_dim, horizontal_dims, stats_file_name
):
    df = df_from_file_ids(
        dataframe_from_ncfile, file_ids, input_dir, time_dim, horizontal_dims
    )

    logger.info("writing stats file to {}".format(stats_file_name))

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
    "--time-dim",
    help=cli_help["time_dim"],
)
@click.option(
    "--horizontal-dims",
    type=CommaSeperatedStrings(),
    help=cli_help["horizontal_dims"],
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
    "--file-ids",
    type=CommaSeperatedStrings(),
    help=cli_help["file_ids"],
)
@click.option(
    "--member_ids",
    type=CommaSeperatedStrings(),
    help=cli_help["member_ids"],
)
@click.option(
    "--perturbed-model-output-dir",
    help=cli_help["perturbed_model_output_dir"],
)
def stats(
    ensemble,
    time_dim,
    horizontal_dims,
    stats_file_name,
    model_output_dir,
    file_ids,
    member_ids,
    perturbed_model_output_dir,
):
    # compute stats for the ensemble run
    if ensemble:
        for m_id in member_ids:
            input_dir = perturbed_model_output_dir.format(member_id=m_id)
            create_stats_dataframe(
                input_dir,
                file_ids,
                time_dim,
                horizontal_dims,
                stats_file_name.format(member_id=m_id),
            )

    # compute the stats for the reference.
    # For ensembles, this file is named
    # stats_{member_id} -> stats_ref (used again in "tolerance")
    create_stats_dataframe(
        model_output_dir,
        file_ids,
        time_dim,
        horizontal_dims,
        stats_file_name.format(member_id="ref"),
    )
