import tempfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from util.click_util import CommaSeperatedStrings, cli_help
from util.constants import cdo_bins
from util.dataframe_ops import df_from_file_ids
from util.file_system import file_names_from_regex
from util.log_handler import logger
from util import model_output_parser


def rel_diff(var1, var2):
    rel_diff = np.zeros_like(var1)

    mask_0 = np.logical_and(np.abs(var1) < 1e-15, np.abs(var2) < 1e-15)
    mask_inf = np.logical_and(np.abs(var1) < 1e-15, np.abs(var2) > 1e-15)
    mask_valid = np.logical_and(np.abs(var1) > 1e-15, np.abs(var2) > 1e-15)

    rel_diff[mask_0] = 0
    rel_diff[mask_inf] = 999
    rel_diff[mask_valid] = np.abs(
        (var1[mask_valid] - var2[mask_valid]) / var1[mask_valid]
    )

    return rel_diff


def rel_diff_stats(file_id, filename, varname, time_dim, horizontal_dims, xarray_ds):
    dims = xarray_ds[varname].dims
    dataarray = xarray_ds[varname]
    time = xarray_ds[time_dim].values

    # compute maximum over all axis except time
    dims_without_time = [d for d in dims if d != time_dim]
    amax = dataarray.max(dim=dims_without_time)

    # allocate empty matrix
    ncol = 9
    matrix = np.empty((amax.size * ncol))

    # compute histogram of relative differences
    for i in range(amax.size):
        hist, edges = np.histogram(
            dataarray[{time_dim: i}].values.flatten(), bins=[0] + cdo_bins
        )
        matrix[i * ncol] = amax[i]
        matrix[i * ncol + 1 : i * ncol + ncol] = hist

    index = pd.MultiIndex.from_product(
        [[file_id], [varname]], names=("file_ID", "variable")
    )
    columns = pd.MultiIndex.from_product(
        [time, ["rel_diff"] + cdo_bins], names=["time", "statistic"]
    )

    return pd.DataFrame(matrix[np.newaxis, :], index=index, columns=columns)


@click.command()
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
@click.option(
    "--cdo-table-file",
    help=cli_help["cdo_table_file"],
)
@click.option(
    "--file-specification",
    type=list,
    help=cli_help["file_specification"],
)
def cdo_table(
    model_output_dir,
    file_ids,
    member_ids,
    perturbed_model_output_dir,
    cdo_table_file,
    file_specification,
):
    # TODO: A single perturbed run provides enough data to make proper statistics.
    #       refactor cdo_table interface to reflect that
    if len(member_ids) > 1:
        logger.warning(
            "Only a single member_id can be specified, using {}".format(member_ids[0])
        )
    member_id = member_ids[0]

    # modify netcdf parse method:
    model_output_parser.dataframe_from_ncfile = rel_diff_stats

    # step 1: compute rel-diff netcdf files
    with tempfile.TemporaryDirectory() as tmpdir:
        for fid in file_ids:
            file_regex = "*{}*.nc".format(fid)
            ref_files, err = file_names_from_regex(model_output_dir, file_regex)
            if err > 0:
                logger.info("did not find any files for ID {}. Continue.".format(fid))
                continue
            ref_files.sort()
            perturb_files, err = file_names_from_regex(
                perturbed_model_output_dir.format(member_id=member_id), file_regex
            )
            if err > 0:
                logger.info("did not find any files for ID {}. Continue.".format(fid))
                continue
            perturb_files.sort()

            for rf, pf in zip(ref_files, perturb_files):
                ref_data = xr.open_dataset("{}/{}".format(model_output_dir, rf))
                perturb_data = xr.open_dataset(
                    "{}/{}".format(
                        perturbed_model_output_dir.format(member_id=member_id), pf
                    )
                )
                diff_data = ref_data.copy()
                varnames = [
                    v
                    for v in list(ref_data.keys())
                    if "time" in ref_data.variables.get(v).dims
                ]

                for v in varnames:
                    diff_data.variables.get(v).values = rel_diff(
                        ref_data.variables.get(v).values,
                        perturb_data.variables.get(v).values,
                    )

                diff_data.to_netcdf("{}/{}".format(tmpdir, rf))
                ref_data.close()
                perturb_data.close()
                diff_data.close()

        # step 2: generate dataframe from precomputed relative differences
        df = df_from_file_ids(file_ids, tmpdir, file_specification)

        # normalize histogram component of DataFrame
        times = np.array(df.columns.levels[0], dtype=int)
        for t in times:
            df.loc[:, (t, cdo_bins)] = df.loc[:, (t, cdo_bins)].div(
                df.loc[:, (t, cdo_bins)].sum(axis=1), axis=0
            )

        logger.info("writing cdo table to {}...".format(cdo_table_file))

        Path(cdo_table_file).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cdo_table_file)
