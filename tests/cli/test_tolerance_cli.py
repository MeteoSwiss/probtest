"""
This module contains tests for verifying the correctness of the tolerance CLI
command.
The tests compare the output of the tolerance CLI with reference tolerance data
and check for discrepancies.
"""

import os

from tests.conftest import fof_file_set
from tests.helpers import (
    assert_empty_df,
    load_pandas,
    pandas_error,
    run_tolerance_cli,
    store_as_potential_new_ref,
)


def test_tolerance_cli(ref_data, df_ref_tolerance, tmp_path, new_ref, fof_file_set):

    stats_file_pattern = os.path.join(ref_data, "stats_{member_id}.csv")



    # fof_file_pattern = os.path.join(
    #     ref_data, "experiments/mch_kenda-ch1_small_member_id_{member_id}/fof{fof_type}.nc"
    # )

    tolerance_file_name = os.path.join(tmp_path, "tolerance.csv")
    #tolerance_file_name2 = os.path.join(tmp_dir, "tolerance{fof_type}.csv")

    # run_tolerance_cli(
    #    ensemble_files=[stats_file_pattern] + fof_file_set["fof"],
    #    tolerance_files=[tolerance_file_name] + fof_file_set["tol"])

    # df_test = load_pandas(tolerance_file_name, index_col=[0, 1])
    # err = pandas_error(df_ref_tolerance, df_test)
    # store_as_potential_new_ref(tolerance_file_name, new_ref)

    # assert_empty_df(err, "Tolerance datasets are not equal!")







# def create_random_stats_file(filename, configurations, seed, perturbation):
#     random.seed(seed)
#     max_time_dim = max(config["time_dim"] for config in configurations)
#     time_header = ",".join(f"{t},{t},{t}" for t in range(max_time_dim))
#     header = [
#         f"time,,,{time_header}",
#         "statistic,," + ",max,mean,min" * max_time_dim,
#         "file_ID,variable,height,,,,,,,,,",
#     ]

#     data = []
#     for config in configurations:
#         time_dim = config["time_dim"]
#         height_dim = config["height_dim"]
#         variable = config["variable"]
#         file_format = config["file_format"]

#         for h in range(height_dim):
#             row = f"{file_format},{variable},{h}.0"
#             for t in range(time_dim):
#                 base_mean = round((h - 2.0) * sin(t), 5)
#                 mean = base_mean + (t + 1.0) * ((seed % 2) == 0) * round(
#                     random.uniform(-perturbation, perturbation), 5
#                 )
#                 max_val = 2.0 * abs(base_mean) + (t + 1.0) * ((seed % 3) == 0) * round(
#                     random.uniform(-perturbation, perturbation), 5
#                 )
#                 min_val = -2.0 * abs(base_mean) + (t + 1.0) * ((seed % 5) == 0) * round(
#                     random.uniform(-perturbation, perturbation), 5
#                 )
#                 row += f",{max_val},{mean},{min_val}"
#             for _ in range(time_dim, max_time_dim):
#                 row += ",,,"
#             data.append(row)

#     with open(filename, "w", encoding="utf-8") as f:
#         for line in header:
#             f.write(line + "\n")
#         for row in data:
#             f.write(row + "\n")


# def stats_file_set(tmp_dir):
#     """
#     Create a set of stats files for testing the selection of members
#     For convenience also the filenames for the members and the tolerance
#     are provided.
#     """
#     configurations = [
#         {
#             "time_dim": 3,
#             "height_dim": 5,
#             "variable": "v1",
#             "file_format": "Format1:*test_3d*.nc",
#         },
#         {
#             "time_dim": 3,
#             "height_dim": 2,
#             "variable": "v2",
#             "file_format": "Format2:*test_2d*.nc",
#         },
#         {
#             "time_dim": 2,
#             "height_dim": 4,
#             "variable": "v3",
#             "file_format": "Format3:*test_2d*.nc",
#         },
#     ]
#     seed = 42

# stats_pattern = os.path.join(tmp_dir, "stats_{member_id}.csv")
# create_random_stats_file(
#         stats_pattern.format(member_id="ref"), configurations, seed, 0.0
#     )
