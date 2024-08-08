"""
CLI for performance meta data plotting script

This script provides a CLI for plotting performance meta data based on timing logs.
It reads timing data from a specified JSON database and generates plots showing
runtime changes across different revisions for specified timers.
The script can save the plots to a specified directory or display them directly.
"""

from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from probtest.util.click_util import CommaSeperatedStrings, cli_help
from probtest.util.constants import DATETIME_FORMAT
from probtest.util.log_handler import logger
from probtest.util.tree import TimingTree
from probtest.util.utils import unique_elements


def plot_meta_data_timer(timer, data, revs, ax, experiment_name, savedir):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))

    # Selecting and plotting the data
    y = np.diff(data.loc[(timer, slice(None)), "total max (s)"].values)
    y = np.insert(y, 0, 0)

    nonnan = ~np.isnan(y)

    y = y[nonnan]
    revs = [r for r, n in zip(revs, nonnan) if n]
    x = range(len(revs))
    ax.bar(x, y)
    srevs = [s[:8] for s in revs]
    ax.set_xticks(x)
    ax.set_xticklabels(srevs, rotation=90)

    ax.set_xlabel("revision")
    ax.set_ylabel("runtime change / s")
    ax.set_title(timer)

    plt.tight_layout()
    if savedir:
        Path(savedir).mkdir(exist_ok=True, parents=True)
        perf_name = f"perf_meta_{experiment_name}_{timer}.png"
        path = f"{savedir}/{perf_name}"
        logger.info("saving figure to %s", path)
        fig.savefig(path)
    else:
        plt.show()


@click.command()
@click.option(
    "--timing-database",
    help=cli_help["timing_database"],
)
@click.option(
    "--savedir",
    help=cli_help["savedir"],
)
@click.option(
    "--timing-names",
    type=CommaSeperatedStrings(),
    help=cli_help["timing_names"],
)
@click.option(
    "--experiment-name",
    help=cli_help["experiment_name"],
)
@click.option("--i-table", type=int, help=cli_help["i_table"], default=-1)
def performance_meta_data(
    timing_database, savedir, timing_names, experiment_name, i_table
):
    tt = TimingTree.from_json(timing_database)

    dates = tt.meta_data["finish_time"]
    if isinstance(dates, str):
        dates = [dates]
    dates = sorted([datetime.strptime(s, DATETIME_FORMAT) for s in dates])

    # create dataframe for revisions and align to tt.data
    index = tt.data[i_table].index
    df = pd.DataFrame(np.nan, index=index, columns=["revision"])
    for i, time in enumerate(tt.meta_data["finish_time"]):
        df.loc[(slice(None), time), :] = tt.meta_data["revision"][i]

    # include history from last 20 weeks
    last_measurement = dates[-1]
    week = timedelta(days=7)
    day = timedelta(days=1)
    reftime = last_measurement + day - week * 20
    seltime = [
        t
        for t in tt.meta_data["finish_time"]
        if datetime.strptime(t, DATETIME_FORMAT) > reftime
    ]
    data = pd.concat(
        [
            tt.data[i_table].loc[(slice(None), seltime), :],
            df.loc[(slice(None), seltime), :],
        ],
        axis=1,
    )
    data.index.set_names(["timer", "sample"], inplace=True)

    # take mean over timers and revisions
    data_mean = data.groupby(["timer", "revision"]).mean()

    # reorder revision index in chronological order
    revs = unique_elements(tt.meta_data["revision"])
    # not every timer will have all the revisions. This will result in NaN
    ref_by_time = pd.MultiIndex.from_product(
        [index.levels[0], revs], names=["timer", "revision"]
    )
    data_mean = data_mean.reindex(ref_by_time)

    # plotting
    ncols = min(len(timing_names), 3)
    nrows = int(np.ceil(len(timing_names) / 3.0))
    _, ax = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(5 * ncols, 3.5 * nrows),
        sharey=True,
        sharex=True,
    )
    if nrows == 1:
        ax = np.expand_dims(ax, axis=0)
    if ncols == 1:
        ax = np.expand_dims(ax, axis=1)

    for i, t in enumerate(timing_names):
        plot_meta_data_timer(
            t, data_mean, revs, ax[i // 3, i % 3], experiment_name, savedir
        )
