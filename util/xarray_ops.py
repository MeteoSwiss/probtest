import sys

from util.log_handler import logger


def statistics_over_horizontal_dim(xarray_da, horizontal_dims, compute_statistics):
    """
    Calculate the horizontal statistics like mean, max and min of a xarray DataArray

    xarray_da: xarray.DataArray
    horizontal_dims: Iterable of possible horizontal dimensions. If multiple
       horizontal dimensions are used their names must be in one string
        separated by ":".
    """

    dims = xarray_da.dims
    for hor_dim in horizontal_dims:
        hor_dim = hor_dim.split(":")
        if all([d in dims for d in hor_dim]):
            return [
                getattr(xarray_da, s)(dim=hor_dim, skipna=False)
                for s in compute_statistics
            ]

    logger.error(
        "Could not find horizontal dimension for variable '{}'. Dims: {}".format(
            xarray_da.name, dims
        )
    )
    sys.exit(1)
