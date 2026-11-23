"""
CLI for plotting a model run in comparison to an ensemble

For each variable (and level), the horizontal mean of every ensemble member
and of the current run is plotted over time as the difference to the
unperturbed reference. The current run passes the extreme-values test of the
check command if it stays within the range covered by the members (for a
factor of one).
"""

import sys
from pathlib import Path

import click
import numpy as np
from matplotlib import lines as mlines
from matplotlib import pyplot as plt

from util.click_util import CommaSeparatedInts, CommaSeparatedStrings, cli_help
from util.dataframe_ops import parse_probtest_stats
from util.log_handler import logger
from util.utils import expand_members, validate_single_stats_file

STATISTIC = "mean"
MAX_NSUBPLOTS = 30
MAX_NCOL = 3
MEMBER_STYLE = {"color": "0.6", "lw": 0.8}
CURRENT_STYLE = {"color": "k", "lw": 2.5}


def mean_differences(df, df_ref):
    """
    Difference of the horizontal means to the ones of the reference, with the
    rows of the reference.
    """
    mean = df.xs(STATISTIC, axis=1, level=1)
    mean_ref = df_ref.xs(STATISTIC, axis=1, level=1)
    return mean.reindex(index=mean_ref.index, columns=mean_ref.columns) - mean_ref


def subplot_titles(index):
    """
    Titles of the rows (file_ID, variable, height): the level is only given
    for variables with several levels.
    """
    n_levels = {}
    for file_id, variable, _ in index:
        n_levels[(file_id, variable)] = n_levels.get((file_id, variable), 0) + 1

    titles = []
    for file_id, variable, height in index:
        title = f"{file_id}: {variable}"
        if n_levels[(file_id, variable)] > 1:
            title += f" (level {height:g})"
        titles.append(title)
    return titles


def plot_differences(member_diffs, current_diff=None):
    """
    One subplot per row with the differences of the members and, if given, of
    the current run. Returns the figure.
    """
    index = member_diffs[0].index
    times = np.asarray(member_diffs[0].columns)

    ncols = min(len(index), MAX_NCOL)
    nrows = int(np.ceil(len(index) / ncols))
    legend_height = 1  # height in inches
    fig_height = 3 * nrows + legend_height

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, fig_height),
        sharex=True,
        squeeze=False,
    )

    for i, (row, title) in enumerate(zip(index, subplot_titles(index))):
        axis = ax[i // ncols, i % ncols]
        limit = 0.0
        for diff in member_diffs:
            values = diff.loc[row].to_numpy(dtype=np.float64)
            axis.plot(times, values, **MEMBER_STYLE)
            limit = np.nanmax([limit, np.nanmax(np.abs(values), initial=0.0)])
        if current_diff is not None and row in current_diff.index:
            values = current_diff.loc[row].to_numpy(dtype=np.float64)
            axis.plot(times, values, **CURRENT_STYLE)
            limit = np.nanmax([limit, np.nanmax(np.abs(values), initial=0.0)])

        if limit > 0:
            axis.set_ylim(-1.1 * limit, 1.1 * limit)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axis.axhline(0.0, color="0.3", lw=0.5, ls=":")
        axis.set_title(title, fontsize="medium")
        if i % ncols == 0:
            axis.set_ylabel(f"{STATISTIC} - reference")

    for i in range(len(index), nrows * ncols):
        ax[i // ncols, i % ncols].set_visible(False)
    # label the lowest subplot of each column
    for col in range(ncols):
        last_row = (len(index) - 1 - col) // ncols
        ax[last_row, col].set_xlabel("timestep")
        ax[last_row, col].tick_params(labelbottom=True)

    handles = [mlines.Line2D([], [], label="ensemble members", **MEMBER_STYLE)]
    if current_diff is not None:
        handles.append(mlines.Line2D([], [], label="current run", **CURRENT_STYLE))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles))
    fig.tight_layout()
    # use an absolute margin for the legend
    fig.subplots_adjust(bottom=legend_height / fig_height)

    return fig


@click.command()
@click.option(
    "--ensemble-files",
    type=CommaSeparatedStrings(),
    default=[],
    help=cli_help["ensemble_files"]
    + "\nNote: this option accepts exactly one stats file.",
)
@click.option(
    "--current-files",
    type=CommaSeparatedStrings(),
    default=[],
    help=cli_help["current_files"]
    + "\nNote: this option accepts at most one stats file.",
)
@click.option(
    "--member-ids",
    type=CommaSeparatedInts(),
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
    "--variables",
    type=CommaSeparatedStrings(),
    default=[],
    help=cli_help["plot_variables"],
)
@click.option(
    "--savedir",
    help=cli_help["savedir"],
)
def ensemble_plot(
    ensemble_files, current_files, member_ids, member_type, variables, savedir
):  # pylint: disable=too-many-positional-arguments
    errors = []
    ensemble_file = validate_single_stats_file(ensemble_files, "ensemble", errors)
    current_file = None
    if current_files:
        current_file = validate_single_stats_file(current_files, "current", errors)
    if not member_ids:
        errors.append("No member ids given.")
    if errors:
        for msg in errors:
            logger.error("ERROR: %s", msg)
        sys.exit(1)

    df_ref = parse_probtest_stats(
        ensemble_file.format(member_id="ref", member_type=""), index_col=[0, 1, 2]
    )
    if variables:
        df_ref = df_ref[df_ref.index.get_level_values("variable").isin(variables)]
    if df_ref.empty:
        logger.error("ERROR: No variables to plot.")
        sys.exit(1)
    if len(df_ref.index) > MAX_NSUBPLOTS:
        logger.info(
            "Limit the subplots to the first %s of %s variables and levels. "
            "Use --variables to select others.",
            MAX_NSUBPLOTS,
            len(df_ref.index),
        )
        df_ref = df_ref.iloc[:MAX_NSUBPLOTS]

    member_diffs = [
        mean_differences(parse_probtest_stats(file, index_col=[0, 1, 2]), df_ref)
        for file in expand_members(
            ensemble_file, member_ids=member_ids, member_type=member_type
        )
    ]
    current_diff = None
    if current_file:
        current_diff = mean_differences(
            parse_probtest_stats(current_file, index_col=[0, 1, 2]), df_ref
        )

    fig = plot_differences(member_diffs, current_diff)

    if savedir:
        Path(savedir).mkdir(exist_ok=True, parents=True)
        path = f"{savedir}/ensemble_plot.png"
        logger.info("saving figure to %s", path)
        fig.savefig(path)
    else:
        plt.show()
