import itertools
import os
import sys

import click
import pandas as pd

from util.click_util import CommaSeperatedInts, cli_help
from util.dataframe_ops import (
    compute_rel_diff_dataframe,
    force_monotonic,
    parse_probtest_csv,
)
from util.log_handler import logger
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


def plot_rankings(rankings, variables, total_member_num):

    # Initialize a dictionary to store the frequency of rankings for each entry in each variable
    ranking_frequency = {index: [0 for _ in range(total_member_num)] for index in range(total_member_num)}
    # Count the frequency of each ranking for each entry in each variable
    for variable, ranking in rankings.items():
        for index, rank in enumerate(ranking):
            ranking_frequency[index][rank-1] = ranking_frequency[index][rank-1] + 1

    # Restructure for plotting
    rf = []
    for key, val in ranking_frequency.items():
        rf.append(val)
    rf = np.array(rf)
    ranks = list(map(str,ranking_frequency.keys()))

    colors = plt.cm.viridis(np.linspace(0, 1, total_member_num))
    for i in range(total_member_num):
        if i==0:
            plt.bar(range(total_member_num), rf[:,i], label=f'Rank {i+1}',color=colors[i])
        else:
            plt.bar(range(total_member_num), rf[:,i], bottom=np.sum(rf[:,0:i],axis=1), label=f'Rank {i+1}',color=colors[i])
    # Add labels and title
    plt.xticks(range(total_member_num), ranks)
    plt.xlabel('Ensemble Member')
    plt.ylabel('Frequency')
    plt.title('Frequency of Rankings')
    plt.legend(title='Rankings', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    print("Saving figure to ranking_frequency_stacked_bar_entries.png")
    plt.savefig('ranking_frequency_stacked_bar_entries.png', dpi=300)


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
    "--total-member-num",
    type=int,
    default=100,
    help=cli_help["total_member_num"],
)
@click.option(
    "--plot-rank-dist/--no-plot-rank-dist",
    is_flag=True,
    help=cli_help["plot_rank_dist"],
)


def optimal_member_sel(stats_file_name, tolerance_file_name, member_num, member_type, total_member_num, plot_rank_dist):
    if len(member_num) == 1:
        member_num = [i for i in range(1, member_num[0] + 1)]
    # read in stats files
    dfs = [
        parse_probtest_csv(stats_file_name.format(member_id=m_id), index_col=[0, 1, 2])
        for m_id in (
            (str(m_num) if not member_type else member_type + "_" + str(m_num))
            for m_num in range(1,total_member_num+1)
        )
    ]
    ndata = len(dfs)
    if ndata < 2:
        logger.critical(
            "not enough ensemble members to find optimal spread, got {} dataset. Abort.".format(ndata)
        )
        sys.exit(1)

    df_ref = parse_probtest_csv(stats_file_name.format(member_id="ref"), index_col=[0, 1, 2])
    vars=set(index[1] for index in df_ref.index)

    # Initialize a dictionary to store maximum values for each variable
    max_values = {var: [] for var in vars}

    # Iterate through each df_diff
    for i in range(total_member_num):
        df_diff = abs(df_ref - dfs[i])

        # Iterate through the variable names
        for var in vars:
            # Extract maximum value for the current variable in the current df_diff
            max_value = df_diff.loc[(slice(None), var), :].max().max()
            # Append the maximum value to the array corresponding to the variable
            max_values[var].append(max_value)

    # Initialize a dictionary to store rankings for each variable
    ranking = {}

    # Iterate through each variable
    # Some indices may have same rank
    for var, max_values in max_values.items():
        # Use rankdata to rank max values in descending order
        rankings = rankdata([-value for value in max_values], method='min')
        # Convert ranks to integers and store them for the current variable
        ranking[var] = [int(rank) for rank in rankings]

    if plot_rank_dist:
        plot_rankings(ranking, vars, total_member_num)

    return
