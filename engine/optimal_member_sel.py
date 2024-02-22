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


def angle_between(A, B):
    # Convert DataFrame to NumPy array
    A_array = A.to_numpy()
    B_array = B.to_numpy()

    # Compute the dot product of the flattened arrays of A and B
    dot_product = np.dot(A_array.flatten(), B_array.flatten())

    # Compute the norms of A and B
    norm_A = np.linalg.norm(A_array)
    norm_B = np.linalg.norm(B_array)

    # Compute the angle in radians
    if norm_A * norm_B == 0:
        cosine_angle = 1.0  # Return 1.0 when norm_A * norm_B is 0
    else:
        cosine_angle = dot_product / (norm_A * norm_B)

    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

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
    if len(member_num) != 1:
        member_num = len(member_num)
    else:
        member_num = member_num[0]

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
    vars=set(index[1] for index in dfs[0].index)

    dfs_rel = [compute_rel_diff_dataframe(dfs[i], df_ref) for i in range(total_member_num)]
    # Only one value per variables and height for each statistic
    dfs_rel = [r.groupby(["file_ID", "variable"]).max() for r in dfs_rel]

    # Find first member: maximum norm
    index = 0
    value = 0
    for i in range(total_member_num):
        if np.linalg.norm(dfs_rel[i]) > value:
            index = i
            value = np.linalg.norm(dfs_rel[i])
    indices = np.arange(total_member_num)
    indices = np.concatenate((indices[:index], indices[index+1:]))
    selection = [index]

    # TODO: file_ID needs to be flexible in case more than one file_ID is used
    file_ID = dfs[0].index.get_level_values('file_ID')[0]
    angles = np.zeros(total_member_num) # selected members will be set to 0
    for m in range(member_num-1):
        for i in indices:
            for var in vars:
                angles[i] = angles[i] + angle_between(dfs_rel[selection[-1]].loc[(file_ID,var)],dfs_rel[i].loc[(file_ID,var)])
        max_angle = np.argmax(angles)
        mask = indices != max_angle
        angles[max_angle] = 0
        indices = indices[mask]
        selection.append(max_angle)

    selection = [x+1 for x in selection] # Members start counting at 1
    print(selection)

#    # Initialize a dictionary to store maximum values for each variable
#    max_values = {var: [] for var in vars}
#
#    # get all possible combinations of the input data
#    combs = list(itertools.product(range(ndata), range(ndata)))
#    # do not use the i==j combinations
#    combs = [(i, j) for i, j in combs if j < i]
#    # compute relative differences for all combinations
#    rdiff = [compute_rel_diff_dataframe(dfs[i], dfs[j]) for i, j in combs]
#
#    file_ID = dfs[0].index.get_level_values('file_ID')[0]
#    for member in range(total_member_num):
#        rdiff_member = [rdiff[i] for i in range(len(rdiff)) if member in combs[i]]
#        rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff_member]
#        df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()
#        for var in vars:
#            max_values[var].append(df_max.loc[(file_ID, var)].max().max())
#
#    # Initialize a dictionary to store rankings for each variable
#    ranking = {}
#
#    # Iterate through each variable
#    # Some indices may have same rank
#    for var, values in max_values.items():
#        # Use rankdata to rank max values in descending order
#        rankings = rankdata([-value for value in values], method='min')
#        # Convert ranks to integers and store them for the current variable
#        ranking[var] = [int(rank) for rank in rankings]
#
#    # Find best members (most ranking=1)
#    sum_ranks = np.zeros(total_member_num)
#    for i in range(total_member_num):
#        sum_ranks[i] = sum(ranking[var][i] for var in vars if ranking[var][i]==1)
#    best_members = np.argsort(-sum_ranks)[:member_num]+1 # members start at 1
#    print(best_members)
#
#    if plot_rank_dist:
#        plot_rankings(ranking, vars, total_member_num)

    return
