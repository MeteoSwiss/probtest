"""
Tolerance methods: how an ensemble is turned into a tolerance file and how a
current run is checked against it.

Two methods are available:

``relative``
    The original probtest method. The tolerance is the largest relative
    difference between any ensemble member and the reference, taken over all
    levels and made monotonic in time.

``extreme_values``
    The min/max test. The raw ensemble statistics (unperturbed reference,
    minimum and maximum over the perturbed members) are stored per variable,
    level and time step. A current run passes if its deviation from the
    reference stays within the deviations seen in the ensemble, scaled by a
    factor. It is applied to the horizontal mean only.
"""

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from util.dataframe_ops import (
    check_intersection,
    compute_rel_diff_dataframe,
    force_monotonic,
    parse_probtest_stats,
)
from util.log_handler import logger
from util.utils import FileType

METHOD_RELATIVE = "relative"
METHOD_EXTREME_VALUES = "extreme_values"
TOLERANCE_METHODS = [METHOD_RELATIVE, METHOD_EXTREME_VALUES]

# statistic of the stats files the extreme-values method is applied to
EXTREME_VALUES_STATISTIC = "mean"
# column labels of an ensemble-statistics file
ENSEMBLE_STATISTICS = ["ref", "min", "max"]

_STATISTIC_LEVEL = 1


def parse_stats_exact(path):
    """
    Read a stats file without rounding errors. The extreme-values method
    compares values down to the machine epsilon, so every file it reads has to
    be parsed exactly.
    """
    return parse_probtest_stats(path, index_col=[0, 1, 2], float_precision="round_trip")


def build_relative_tolerance(df_ref, dfs, file_type, minimum_tolerance):
    """
    Largest relative difference between the reference and any member.
    """
    rdiff = [compute_rel_diff_dataframe(df_ref, df) for df in dfs]

    if file_type is FileType.STATS:
        rdiff_max = [r.groupby(["file_ID", "variable"]).max() for r in rdiff]
        df_max = pd.concat(rdiff_max).groupby(["file_ID", "variable"]).max()
        df_max = df_max.map(lambda x: minimum_tolerance if x < minimum_tolerance else x)

        force_monotonic(df_max)

    else:
        df_max = pd.concat(rdiff, axis=1).max(axis=1)
        df_max = df_max.map(lambda x: minimum_tolerance if x < minimum_tolerance else x)

    return df_max


def _select_statistic(df, statistic):
    return df.xs(statistic, axis=1, level=_STATISTIC_LEVEL)


def build_extreme_values_statistics(df_ref, dfs):
    """
    Ensemble statistics for the extreme-values method.

    Returns a DataFrame with the index of the stats files
    (file_ID, variable, height) and the columns (time, statistic), where
    statistic is one of ``ref``, ``min`` and ``max``. Minimum and maximum are
    taken over the perturbed members only; the reference is not part of them.
    A value missing in any member yields NaN, which fails the check later.
    """
    ref = _select_statistic(df_ref, EXTREME_VALUES_STATISTIC)
    members = np.stack(
        [
            _select_statistic(df, EXTREME_VALUES_STATISTIC)
            .reindex(index=ref.index, columns=ref.columns)
            .to_numpy(dtype=np.float64)
            for df in dfs
        ],
        axis=0,
    )

    parts = {
        "ref": ref,
        "min": pd.DataFrame(
            np.min(members, axis=0), index=ref.index, columns=ref.columns
        ),
        "max": pd.DataFrame(
            np.max(members, axis=0), index=ref.index, columns=ref.columns
        ),
    }

    df_out = pd.concat(parts, axis=1, names=["statistic", "time"])
    df_out = df_out.swaplevel(axis=1)
    columns = pd.MultiIndex.from_product(
        [ref.columns, ENSEMBLE_STATISTICS], names=["time", "statistic"]
    )
    return df_out[columns]


@dataclass
class ExtremeValuesResult:
    """
    Outcome of an extreme-values check.

    summary: one row per (file_ID, variable) with the columns ``failed`` and
        ``total`` (number of level/time-step values).
    failures: one row per failing (file_ID, variable, height, time) with the
        observed value, the reference, their difference and the allowed range.
    """

    passed: bool
    summary: pd.DataFrame
    failures: pd.DataFrame


def extreme_values_bounds(df_ens, factor):
    """
    Allowed range of (current - reference) derived from ensemble statistics.

    Where the ensemble has no spread on one side, the bound falls back to the
    machine epsilon scaled by the magnitude of the reference.
    """
    ref = _select_statistic(df_ens, "ref")
    eps_ref = np.finfo(np.float64).eps * ref.abs()

    upper = factor * np.maximum(_select_statistic(df_ens, "max") - ref, eps_ref)
    lower = factor * np.minimum(_select_statistic(df_ens, "min") - ref, -eps_ref)
    return ref, lower, upper


def check_extreme_values(df_ens, df_cur, factor):
    """
    Check the horizontal means of a current run against ensemble statistics.

    df_ens: ensemble statistics as built by build_extreme_values_statistics
    df_cur: stats DataFrame of the current run
    """
    skip_test, df_ens, df_cur = check_intersection(df_ens, df_cur)
    if skip_test:
        logger.error("RESULT: check FAILED")
        sys.exit(1)

    ref, lower, upper = extreme_values_bounds(df_ens, factor)
    cur = _select_statistic(df_cur, EXTREME_VALUES_STATISTIC).reindex(
        index=ref.index, columns=ref.columns
    )
    diff = cur - ref

    # comparisons with NaN are False: a NaN in any operand fails
    ok = (diff >= lower) & (diff <= upper)

    by_variable = (~ok).groupby(["file_ID", "variable"], sort=False)
    summary = pd.DataFrame(
        {
            "failed": by_variable.sum().sum(axis=1).astype(int),
            "total": by_variable.count().sum(axis=1).astype(int),
        }
    )

    def _long(df, name):
        return df.stack(future_stack=True).rename(name)

    table = pd.concat(
        [
            _long(cur, "observed"),
            _long(ref, "reference"),
            _long(diff, "difference"),
            _long(lower, "lower"),
            _long(upper, "upper"),
        ],
        axis=1,
    )
    failures = table[~_long(ok, "ok")]

    return ExtremeValuesResult(
        passed=bool(ok.to_numpy().all()), summary=summary, failures=failures
    )


def check_file_with_extreme_values(tolerance_file_name, input_file_cur, factor):
    """
    Check a current stats file against an ensemble-statistics file.
    """
    if input_file_cur.file_type is not FileType.STATS:
        logger.critical(
            "The %s method only supports stats files. Abort.", METHOD_EXTREME_VALUES
        )
        sys.exit(1)

    df_ens = parse_ensemble_statistics(tolerance_file_name)
    df_cur = parse_stats_exact(input_file_cur.path)

    logger.info("applying a factor of %s to the spread", factor)
    logger.info(
        "checking %s against the ensemble statistics from %s",
        input_file_cur.path,
        tolerance_file_name,
    )

    return check_extreme_values(df_ens, df_cur, factor)


def parse_ensemble_statistics(path):
    """
    Read an ensemble-statistics file written by the extreme-values method.
    """
    df = parse_stats_exact(path)

    labels = set(df.columns.get_level_values(_STATISTIC_LEVEL))
    if labels != set(ENSEMBLE_STATISTICS):
        logger.critical(
            "%s is not an ensemble-statistics file of the %s method "
            "(expected the statistics %s, found %s). "
            "Was it created with 'tolerance --method %s'? Abort.",
            path,
            METHOD_EXTREME_VALUES,
            ENSEMBLE_STATISTICS,
            sorted(labels),
            METHOD_EXTREME_VALUES,
        )
        sys.exit(1)

    return df
