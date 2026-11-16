"""
Reading and writing of ensemble-statistics files of the extreme-values method.

Ensemble statistics are handled as a DataFrame with the index
(file_ID, variable, height) and the columns (time, statistic), where statistic
is one of ``ref``, ``min`` and ``max``. Two storage formats are supported and
selected by the file extension:

``.csv``
    The DataFrame as written by pandas, like the stats files.

``.npz``
    The layout of chilitest (utils/chilitest in ICON), a set of files:

    - one ``.npz`` file per dimension (``2d``, ``3d``). The file name contains
      the placeholder ``{dim}``, e.g. ``exp_stats_{dim}.npz``. Each file holds
      the arrays ``times``, ``plev`` (3d only) and ``<variable>_ref``,
      ``<variable>_min``, ``<variable>_max`` with the shape (time) for 2d and
      (time, level) for 3d.
    - ``exp_stats_meta.json`` with the variables and the kind of spatial
      average per dimension.

    A dimension corresponds to one file_ID of the stats files: ``2d`` to a
    file_ID whose variables have a single level, ``3d`` to one whose variables
    share a set of levels.
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from util.dataframe_ops import parse_probtest_stats
from util.log_handler import logger

# column labels of an ensemble-statistics file
ENSEMBLE_STATISTICS = ["ref", "min", "max"]

DIM_PLACEHOLDER = "{dim}"
DIM_2D = "2d"
DIM_3D = "3d"
SPATIAL_AVERAGE_TYPES = {
    DIM_2D: "global_mean",
    DIM_3D: "surface_mean_per_p_level",
}
# level label of the stats files for variables without a vertical dimension
NO_LEVEL = -1.0

_INDEX_NAMES = ["file_ID", "variable", "height"]
_COLUMN_NAMES = ["time", "statistic"]


def _abort(message, *args):
    logger.critical(message + " Abort.", *args)
    sys.exit(1)


def is_npz(path):
    return str(path).lower().endswith(".npz")


def write_ensemble_statistics(df_ens, path, dimensions=None, times=None):
    """
    Write ensemble statistics in the format given by the extension of path.

    dimensions: only for .npz, see write_npz_set
    times: only for .npz, see write_npz_set
    """
    if is_npz(path):
        write_npz_set(df_ens, path, dimensions=dimensions, times=times)
    else:
        df_ens.to_csv(path)


def read_ensemble_statistics(path, df_cur=None, dimensions=None):
    """
    Read ensemble statistics in the format given by the extension of path.

    df_cur, dimensions: only for .npz, see read_npz_set
    """
    if is_npz(path):
        return read_npz_set(path, df_cur=df_cur, dimensions=dimensions)
    return parse_ensemble_statistics(path)


def parse_ensemble_statistics(path):
    """
    Read an ensemble-statistics file in the csv format.
    """
    df = parse_probtest_stats(path, index_col=[0, 1, 2])

    labels = set(df.columns.get_level_values(1))
    if labels != set(ENSEMBLE_STATISTICS):
        _abort(
            "%s is not an ensemble-statistics file of the extreme_values method "
            "(expected the statistics %s, found %s). "
            "Was it created with 'tolerance --method extreme_values'?",
            path,
            ENSEMBLE_STATISTICS,
            sorted(labels),
        )

    # the csv parser sorts the statistics by name
    columns = pd.MultiIndex.from_product(
        [df.columns.get_level_values(0).unique(), ENSEMBLE_STATISTICS],
        names=df.columns.names,
    )
    return df[columns]


def meta_file_name(path):
    """
    Name of the meta file belonging to a set of .npz files.
    """
    return os.path.splitext(str(path).replace(DIM_PLACEHOLDER, "meta"))[0] + ".json"


def _check_placeholder(path):
    if DIM_PLACEHOLDER not in str(path):
        _abort(
            "The name of .npz ensemble-statistics files has to contain the "
            "placeholder %s (one file is used per dimension), got %s.",
            DIM_PLACEHOLDER,
            path,
        )


def _levels_by_variable(df_file):
    """
    Variables of the rows of one file_ID in their order of appearance, each
    with the list of its levels.
    """
    levels = {}
    for variable, height in df_file.index:
        levels.setdefault(variable, []).append(height)
    return levels


def _dimension_of(file_id, levels):
    level_lists = list(levels.values())
    if all(len(lev) == 1 for lev in level_lists):
        return DIM_2D
    if all(lev == level_lists[0] for lev in level_lists):
        return DIM_3D
    return _abort(
        "Cannot store %s in the .npz format: its variables neither all have a "
        "single level nor all share the same levels.",
        file_id,
    )


def _file_ids_by_dimension(df_ens, dimensions):
    """
    Map each dimension to the file_ID stored under it.
    """
    file_ids = list(dict.fromkeys(df_ens.index.get_level_values("file_ID")))

    if dimensions is not None:
        unknown = set(dimensions.values()) - set(file_ids)
        if unknown or set(dimensions.values()) != set(file_ids):
            _abort(
                "The dimensions %s do not match the file IDs of the ensemble "
                "statistics %s.",
                dimensions,
                file_ids,
            )
        return dict(dimensions)

    by_dimension = {}
    for file_id in file_ids:
        levels = _levels_by_variable(df_ens.xs(file_id, level="file_ID"))
        dim = _dimension_of(file_id, levels)
        if dim in by_dimension:
            _abort(
                "Both %s and %s would be stored as the dimension %s. "
                "Assign the dimensions explicitly.",
                by_dimension[dim],
                file_id,
                dim,
            )
        by_dimension[dim] = file_id
    return by_dimension


def write_npz_set(df_ens, path, dimensions=None, times=None):
    """
    Write ensemble statistics as a set of .npz files and a meta file.

    path: file name containing the placeholder {dim}
    dimensions: dict mapping a dimension ("2d", "3d") to the file_ID stored
        under it. By default, a file_ID whose variables have a single level is
        stored as "2d" and one with several levels as "3d".
    times: dict mapping a dimension to the array stored as ``times``. By
        default, the indices of the time steps are stored.
    """
    _check_placeholder(path)
    by_dimension = _file_ids_by_dimension(df_ens, dimensions)
    n_time = len(df_ens.columns.get_level_values(0).unique())

    meta: dict = {"variables": {}, "spatial_average_type": {}}

    for dim, file_id in by_dimension.items():
        df_file = df_ens.xs(file_id, level="file_ID")
        levels = _levels_by_variable(df_file)
        single_level = dim == DIM_2D or all(len(lev) == 1 for lev in levels.values())
        if dim == DIM_2D and not single_level:
            _abort("%s has several levels and cannot be stored as %s.", file_id, dim)

        arrays = {
            "times": (
                np.asarray(times[dim])
                if times is not None and dim in times
                else np.arange(n_time, dtype=np.float64)
            )
        }
        if dim != DIM_2D:
            arrays["plev"] = np.asarray(next(iter(levels.values())), dtype=np.float64)

        for variable in levels:
            for statistic in ENSEMBLE_STATISTICS:
                # (level, time) -> (time, level)
                values = (
                    df_file.loc[[variable]]
                    .xs(statistic, axis=1, level=1)
                    .to_numpy(dtype=np.float64)
                    .T
                )
                arrays[f"{variable}_{statistic}"] = (
                    values[:, 0] if dim == DIM_2D else values
                )

        file_name = str(path).replace(DIM_PLACEHOLDER, dim)
        np.savez(file_name, **arrays)  # type: ignore[arg-type]

        meta["variables"][dim] = list(levels)
        meta["spatial_average_type"][dim] = SPATIAL_AVERAGE_TYPES.get(
            dim, SPATIAL_AVERAGE_TYPES[DIM_3D]
        )

    with open(meta_file_name(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _npz_files(path):
    """
    The existing files of a set, by dimension.
    """
    pattern = re.compile(
        re.escape(str(path)).replace(re.escape(DIM_PLACEHOLDER), "(?P<dim>[^/]+)") + "$"
    )
    files = {}
    for file_name in sorted(glob.glob(str(path).replace(DIM_PLACEHOLDER, "*"))):
        match = pattern.match(file_name)
        if match:
            files[match.group("dim")] = file_name
    return files


def _load_npz(file_name):
    """
    Variables of one .npz file with their arrays in the shape (time, level),
    and the levels (None for a file without levels).
    """
    with np.load(file_name) as data:
        arrays = {key: data[key] for key in data.files}

    variables = [key[: -len("_ref")] for key in arrays if key.endswith("_ref")]
    missing = [
        f"{variable}_{statistic}"
        for variable in variables
        for statistic in ENSEMBLE_STATISTICS
        if f"{variable}_{statistic}" not in arrays
    ]
    if not variables or missing:
        _abort(
            "%s does not contain the statistics %s for all variables (missing: %s).",
            file_name,
            ENSEMBLE_STATISTICS,
            missing or "everything",
        )

    plev = arrays.get("plev")
    values = {}
    for variable in variables:
        for statistic in ENSEMBLE_STATISTICS:
            array = np.asarray(arrays[f"{variable}_{statistic}"], dtype=np.float64)
            if array.ndim == 1:
                array = array[:, np.newaxis]
            if array.ndim != 2 or (
                array.shape[1] != 1 and (plev is None or len(plev) != array.shape[1])
            ):
                _abort(
                    "Unexpected shape %s of %s_%s in %s.",
                    array.shape,
                    variable,
                    statistic,
                    file_name,
                )
            values[(variable, statistic)] = array

    n_level = values[(variables[0], "ref")].shape[1]
    return variables, values, (plev if n_level > 1 or plev is not None else None)


def _matching_file_id(df_cur, dim, variables, n_level):
    """
    The file_ID of the current stats that corresponds to a .npz file: the one
    containing all its variables with the same number of levels.
    """
    candidates = []
    for file_id in dict.fromkeys(df_cur.index.get_level_values("file_ID")):
        levels = _levels_by_variable(df_cur.xs(file_id, level="file_ID"))
        if all(len(levels.get(variable, [])) == n_level for variable in variables):
            candidates.append(file_id)

    if len(candidates) != 1:
        _abort(
            "Cannot assign the dimension %s of the ensemble statistics to a file "
            "ID of the current stats file: %s contain(s) all its variables %s on "
            "%s level(s). Assign the dimensions explicitly.",
            dim,
            candidates or "no file IDs",
            variables,
            n_level,
        )
    return candidates[0]


def read_npz_set(path, df_cur=None, dimensions=None):
    """
    Read ensemble statistics from a set of .npz files.

    path: file name containing the placeholder {dim}
    df_cur: stats DataFrame the statistics are going to be compared to. It is
        used to find the file_ID of each dimension, and the level label of
        variables without levels.
    dimensions: dict mapping a dimension to its file_ID. Overrides the
        assignment derived from df_cur. Without both, the dimension itself is
        used as file_ID.
    """
    _check_placeholder(path)
    files = _npz_files(path)
    if not files:
        _abort("No ensemble-statistics files found for %s.", path)

    frames = []
    for dim, file_name in files.items():
        variables, values, plev = _load_npz(file_name)
        n_time, n_level = values[(variables[0], "ref")].shape

        if dimensions is not None and dim in dimensions:
            file_id = dimensions[dim]
        elif df_cur is not None:
            file_id = _matching_file_id(df_cur, dim, variables, n_level)
        else:
            file_id = dim

        rows = []
        data = []
        for variable in variables:
            if plev is not None:
                heights = [float(p) for p in plev]
            else:
                heights = [_single_level_label(df_cur, file_id, variable)]
            for i_level, height in enumerate(heights):
                rows.append((file_id, variable, height))
                data.append(
                    np.stack(
                        [
                            values[(variable, statistic)][:, i_level]
                            for statistic in ENSEMBLE_STATISTICS
                        ],
                        axis=1,
                    ).reshape(-1)
                )

        frames.append(
            pd.DataFrame(
                np.asarray(data),
                index=pd.MultiIndex.from_tuples(rows, names=_INDEX_NAMES),
                columns=pd.MultiIndex.from_product(
                    [range(n_time), ENSEMBLE_STATISTICS], names=_COLUMN_NAMES
                ),
            )
        )

    return pd.concat(frames)


def _single_level_label(df_cur, file_id, variable):
    """
    Level label of a variable without levels. The .npz format does not store
    it, so it is taken from the stats the statistics are compared to.
    """
    if df_cur is not None:
        try:
            heights = df_cur.xs((file_id, variable), level=["file_ID", "variable"])
        except KeyError:
            return NO_LEVEL
        if len(heights.index) == 1:
            return float(heights.index[0])
    return NO_LEVEL
