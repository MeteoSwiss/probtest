import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray

from util.constants import CHECK_THRESHOLD
from util.file_system import file_names_from_regex
from util.log_handler import logger

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


def get_variables(data, time_dim, horizontal_dims):
    # return a list of variable names from the dataset data that have a time dimension
    # and horizaontal dimension or in case there is no time dimension just the variables
    #  with horizontal dimension
    all_variables = data.variables.keys()

    if any(v for v in all_variables if "None" in data.variables[v].dims):
        logger.error('"None" as variable dimension leads to undefined behavior.')
        sys.exit(1)

    variables = []

    if isinstance(horizontal_dims, Iterable):
        horizontal_dims_unpacked = [x for hd in horizontal_dims for x in hd.split(":")]

    if "None" != time_dim:
        if isinstance(horizontal_dims, Iterable):
            for v in all_variables:
                if (
                    any(hd in horizontal_dims_unpacked for hd in data.variables[v].dims)
                    and time_dim in data.variables[v].dims
                ):
                    variables.append(v)
        else:
            variables = [
                v
                for v in all_variables
                if time_dim in data.variables[v].dims and v != time_dim
            ]
    else:
        if isinstance(horizontal_dims, Iterable):
            for v in all_variables:
                if any(hd in horizontal_dims_unpacked for hd in data.variables[v].dims):
                    variables.append(v)
        else:
            logger.error(
                '"None" == time_dim and hasattr(horizontal_dims, "__iter__") '
                + "does not work."
            )
            sys.exit(1)

    return variables


def parse_csv(path, index_col):
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


def df_from_file_ids(func, file_ids, input_dir, time_dim, horizontal_dims=None):
    fid_dfs = []

    for fid in file_ids:
        file_regex = "*{}*.nc".format(fid)
        input_files, err = file_names_from_regex(input_dir, file_regex)
        if err > 0:
            logger.info("did not find any files for ID {}. Continue.".format(fid))
            continue

        file_dfs = []

        for f in input_files:
            ds = xarray.open_dataset("{}/{}".format(input_dir, f), decode_cf=False)
            logger.debug("processing file {}".format(f))
            var_tmp = get_variables(ds, time_dim, horizontal_dims)

            var_dfs = []

            for v in var_tmp:
                kwargs = dict(
                    file_id=fid, filename=f, varname=v, time_dim=time_dim, xarray_ds=ds
                )
                if horizontal_dims is not None:
                    kwargs["horizontal_dims"] = horizontal_dims
                sub_df = func(**kwargs)
                var_dfs.append(sub_df)

            ds.close()
            # different variables in a file have same timestamps:
            # add along variable axis
            file_dfs.append(pd.concat(var_dfs, axis=0))

        # same file IDs will have same variables but with different timestamps:
        # add along time axis
        fid_dfs.append(pd.concat(file_dfs, axis=1))

    # different file IDs will have different variables but with same timestamps:
    # add along variable axis
    df = pd.concat(fid_dfs, axis=0)

    # sort and relabel times so that they are 0, 1, 2, 3, 4, ...
    unique_times = list(df.columns.levels[0])
    unique_times.sort()
    df = df.reindex(columns=unique_times, level=0)
    time_to_int = {t: i for i, t in enumerate(unique_times)}
    df.rename(columns=time_to_int, inplace=True)

    return df
