"""
CLI for performance checks

This module provides a command-line interface (CLI) tool for evaluating the
performance of a current experiment by comparing its timing data with a
reference.
It assesses whether the current experiment's runtime is within acceptable limits
based on specified parameters and provides feedback on whether the performance
check has passed or failed.
"""

import sys

import click
import numpy as np
import xarray as xr

from util.click_util import cli_help
from util.log_handler import logger
from util.tree import TimingTree


@click.command()
@click.option(
    "--timing-current",
    help=cli_help["timing_current"],
)
@click.option("--i-table", type=int, help=cli_help["i_table"], default=-1)
def performance_stats(
    timing_current,
    i_table,
):  # pylint: disable=too-many-positional-arguments



    timing_tree = TimingTree.from_json(timing_current)

    timer_names = ['model_init', 'total', 'integrate_nh', 'nh_solve', 'nh_hdiff', 'transport', 'physics']

    aggregate_timer_names = {'dycore':('nh_solve', 'nh_hdiff', 'transport')}

    all_timer_names = timer_names + list(aggregate_timer_names.keys())

    dims = ('name', 'metric')

    coords = {'name': all_timer_names, 'metric': ['mean', 'std']}

    print(all_timer_names)

    # Create an empty DataArray with np.nan
    timer_stats = xr.DataArray(
        data=np.full((len(coords['name']), len(coords['metric'])), np.nan),  # Fill with NaN
        coords=coords,
        dims=dims
    )


    for timer_name in timer_names:
      times = np.asarray(
          timing_tree.data[i_table].loc[(timer_name, slice(None)), "total max (s)"].values
      )
      timer_stats.loc[timer_name, 'mean'] = np.mean(times)
      timer_stats.loc[timer_name, 'std'] = np.std(times)



    for aggregated_timer_name in aggregate_timer_names.keys():

      times = np.zeros_like(np.asarray(
          timing_tree.data[i_table].loc[(aggregate_timer_names[aggregated_timer_name][0], slice(None)), "total max (s)"].values
      ))
      for timer_name in aggregate_timer_names[aggregated_timer_name]:
        times += np.asarray(
            timing_tree.data[i_table].loc[(timer_name, slice(None)), "total max (s)"].values
        )
      timer_stats.loc[aggregated_timer_name, 'mean'] = np.mean(times)
      timer_stats.loc[aggregated_timer_name, 'std'] = np.std(times)


    # Save the dataset to a NetCDF file
    timer_stats.to_netcdf('timer_stats.nc')

    # Load it back
    loaded_dataarray = xr.open_dataarray('timer_stats.nc')

    print(timer_stats)
    print(loaded_dataarray)
    print(timer_stats-loaded_dataarray)