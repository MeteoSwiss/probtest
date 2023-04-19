import collections
import pathlib
import sys

import numpy as np
import pandas as pd

from util.constants import CHECK_THRESHOLD
from util.file_system import file_names_from_regex
from util.log_handler import logger
from util.model_output_parser import model_output_parser

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2e}".format)


def force_monotonic(dataframe):
    stats = list(dataframe.columns.levels[1])
    for s in stats:
        dataframe.loc[:, (slice(None), s)] = dataframe.loc[:, (slice(None), s)].cummax(
            axis=1
        )


def compute_rel_diff_dataframe(df1, df2):
    out = (df1 - df2).abs() / df1.abs()
    smalls = df1.abs() < CHECK_THRESHOLD
    out[smalls] = 0.0
    return out


def compute_div_dataframe(df1, df2):
    return df1 / df2.replace({0: np.nan})


def parse_probtest_csv(path, index_col):
    df = pd.read_csv(path, index_col=index_col, header=[0, 1])

    times = df.columns.levels[0].astype(int)
    df.columns = df.columns.set_levels(times, level=0)

    # the dataframe's time column will be read as string,
    # thus ordered like "0", "1", "10", "11", .. "2", ...
    new_cols = pd.MultiIndex.from_product(
        [sorted(df.columns.levels[0]), df.columns.levels[1]],
        names=df.columns.names,
    )

    return pd.DataFrame(df, columns=new_cols)


# reduce the amount of logging by logging each message only once.
logged_parser_fid_combination = []


def try_to_read_input_file(fid, file_name, file_specification):
    """Try to read input file file_name using the file_specification."""
    for spec_label, spec_glob, specification in file_specification:
        if pathlib.Path(file_name).match(spec_glob):
            try:
                file_parser = model_output_parser[specification["format"].lower()]
            except KeyError:
                logger.error(
                    "No parser defined for format `{}` of file `{}`.".format(
                        specification["format"], file_name
                    )
                )
                sys.exit(1)
            logger.debug(
                "Use file_parser `{}` for file `{}`".format(spec_label, file_name)
            )
            message = "Use file_parser `{}` for fid `{}`".format(spec_label, fid)
            if message not in logged_parser_fid_combination:
                logged_parser_fid_combination.append(message)
                logger.info(message)
            var_dfs = file_parser(
                "{}:{}".format(spec_label, fid), file_name, specification
            )
            break
    else:
        logger.error("No specification to read file `{}` found.".format(file_name))
        sys.exit(1)

    if var_dfs is None:
        # the do_nothing parser returns None
        return spec_label, None

    if len(var_dfs) == 0:
        logger.error("Could not find any variables in `{}`".format(file_name))
        logger.error(
            "Used file_parser `{}` with glob `{}` ".format(spec_label, spec_glob)
        )
        sys.exit(1)

    return spec_label, pd.concat(var_dfs, axis=0)

    # different variables in a file have same timestamps:
    # concatenate along variable axis


def df_from_file_ids(file_ids, input_dir, file_specification):
    """
    file_specification: [list(spec_label, spec_glob, specification), ...]
        spec_label: str
            Name of the file specification
        spec_glob: str
            Glob to match specific files via pathlib.Path.match
        specification: dict(format: str, **kwargs)
            dictionary that specifies the file format defined by the key.
            format: str
                The format defines the corresponding parser function.
            **kwargs:
                Other keyword arguments can be used by the corresponding parser.
                Examples:
                time_dim: str
                    The time dimension in the model output files
                horizontal_dims: str
                    List possible horizontal dimensions. If multiple horizontal
                    dimensions are used their names must be combined in one string
                    separated by ":".
    """

    # Collect data frames for each combination of file id (fid) and
    # specification (spec). Frames for the same fid and spec represent
    # different timestamps and have to be concatenated along time-axis (axis=1).
    # Time-concatenated frames from different ids and/or specifications will be
    # concatenated along variable-axis (axis=0).
    dfs_of_fid_and_spec = collections.defaultdict(list)

    for fid in file_ids:
        input_files, err = file_names_from_regex(input_dir, fid)
        if err > 0:
            logger.info("did not find any files for ID glob {}. Continue.".format(fid))
            continue

        for f in input_files:
            spec_label, var_df = try_to_read_input_file(
                fid, "{}/{}".format(input_dir, f), file_specification
            )
            if var_df is not None:
                dfs_of_fid_and_spec[(fid, spec_label)].append(var_df)

    fid_dfs = []
    for file_dfs in dfs_of_fid_and_spec.values():
        # same file IDs and file type specification will have same variables but
        # with different timestamps: concatenate along time axis
        fid_dfs.append(pd.concat(file_dfs, axis=1))

    fid_dfs = unify_time_index(fid_dfs)
    # different file IDs will have different variables but with same timestamps:
    # concatenate along variable axis
    df = pd.concat(fid_dfs, axis=0)

    return df


def unify_time_index(fid_dfs):
    """
    Unify the column index of all data frames by replacing the time with a
    continuous range from 0 to N-1

    fid_dfs: Iterable(pd.DataFrame)
    """
    fid_dfs_out = []
    for df in fid_dfs:
        # Find out number of time steps in the column MultiIndex.
        # Is there an easier way to do so without assuming the order of indices?
        time_multiindex_index = df.columns.names.index("time")
        ntime = len(df.columns.levels[time_multiindex_index])

        unique_times = list(df.columns.levels[time_multiindex_index])
        unique_times.sort()
        df = df.reindex(columns=unique_times, level=time_multiindex_index)

        df.columns = df.columns.set_levels(range(ntime), level="time")

        fid_dfs_out.append(df)

    return fid_dfs_out
