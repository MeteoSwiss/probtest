import sys

import click
import numpy as np

from engine.check import check_intersection, check_variable
from engine.tolerance import tolerance
from util.click_util import CommaSeperatedInts, cli_help
from util.dataframe_ops import compute_rel_diff_dataframe, parse_probtest_csv
from util.log_handler import logger


def angle_between(A, B):
    # Convert DataFrame to NumPy array
    A_array = A.to_numpy()
    B_array = B.to_numpy()

    # Find indices where both arrays are not NaN
    not_nan_indices = ~np.isnan(A_array) & ~np.isnan(B_array)
    A_array = A_array[not_nan_indices]
    B_array = B_array[not_nan_indices]

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


def select_members(stats_file_name, member_num, member_type, total_member_num):
    # read in stats files
    dfs = [
        parse_probtest_csv(stats_file_name.format(member_id=m_id), index_col=[0, 1, 2])
        for m_id in (
            (str(m_num) if not member_type else member_type + "_" + str(m_num))
            for m_num in range(1, total_member_num + 1)
        )
    ]
    ndata = len(dfs)
    if ndata < 2:
        logger.critical(
            (
                "not enough ensemble members to find optimal spread, got {} dataset."
                + "Abort."
            ).format(ndata)
        )
        sys.exit(1)

    df_ref = parse_probtest_csv(
        stats_file_name.format(member_id="ref"), index_col=[0, 1, 2]
    )
    dfs_rel = [
        compute_rel_diff_dataframe(dfs[i], df_ref) for i in range(total_member_num)
    ]
    # Only one value per variables and height for each statistic
    dfs_rel = [r.groupby(["file_ID", "variable"]).max() for r in dfs_rel]
    # Find first member: maximum norm
    index = 0
    value = 0
    for i in range(total_member_num):
        norm = 0
        dfs_rel_array = dfs_rel[i].to_numpy()
        nan_indices=np.isnan(dfs_rel_array)
        dfs_rel_array[nan_indices] = 0
        norm = np.linalg.norm(dfs_rel_array)
        if norm > value:
            index = i
            value = norm
    indices = np.arange(total_member_num)
    indices = np.concatenate((indices[:index], indices[index + 1 :]))
    selection = [index]

    angles = np.zeros(total_member_num)  # selected members will be set to 0
    for m in range(member_num - 1):
        for i in indices:
            angles[i] = angles[i] + angle_between(dfs_rel[selection[-1]], dfs_rel[i])**2
        max_angle = np.argmax(angles)
        mask = indices != max_angle
        angles[max_angle] = 0
        indices = indices[mask]
        selection.append(max_angle)

    selection = [x + 1 for x in selection]  # Members start counting at 1

    return selection


def test_selection(
    stats_file_name, tolerance_file_name, total_member_num, member_type, factor
):


    input_file_ref = stats_file_name.format(member_id="ref")
    passed = 0
    df_tol = parse_probtest_csv(tolerance_file_name, index_col=[0, 1])
    df_tol *= factor
    df_ref = parse_probtest_csv(input_file_ref, index_col=[0, 1, 2])

    for m_num in range(1,total_member_num+1):
        m_id = str(m_num) if not member_type else member_type + "_" + str(m_num)

        df_cur = parse_probtest_csv(
            stats_file_name.format(member_id=m_id), index_col=[0, 1, 2]
        )

        # check if variables are available in reference file
        skip_test, df_ref, df_cur = check_intersection(df_ref, df_cur)
        if skip_test:  # No intersection
            logger.info(
                "ERROR: No intersection between variables in input and reference file."
            )
            exit(1)

        # compute relative difference
        diff_df = compute_rel_diff_dataframe(df_ref, df_cur)
        # take maximum over height
        diff_df = diff_df.groupby(["file_ID", "variable"]).max()

        out, err, tol = check_variable(diff_df, df_tol)

        if out:
            passed = passed + 1

    logger.info(
        "The tolerance test passed for {} out of {} references.".format(
            passed, total_member_num
        )
    )
    return


@click.command()
@click.option(
    "--test-tolerance/--no-test-tolerance",
    is_flag=True,
    help=cli_help["test_tolerance"],
)
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
    default="15",
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
    "--factor",
    type=float,
    help=cli_help["factor"],
)
def optimal_member_sel(
    test_tolerance,
    stats_file_name,
    tolerance_file_name,
    member_num,
    member_type,
    total_member_num,
    factor,
):

    if len(member_num) != 1:
        logger.info(
            "ERROR: The optimal member selection needs a single value for member_num."
        )
        exit(1)
    else:
        member_num = member_num[0]

    if test_tolerance:
        # Test selection
        test_selection(
            stats_file_name, tolerance_file_name, total_member_num, member_type, factor
        )
    else:
        selection = select_members(
            stats_file_name, member_num, member_type, total_member_num
        )
        print(selection)

        # Create tolerances from selection
        context = click.Context(tolerance)
        context.invoke(
            tolerance,
            stats_file_name=stats_file_name,
            tolerance_file_name=tolerance_file_name,
            member_num=selection,
            member_type=member_type,
        )

    return
