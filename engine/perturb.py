"""
CLI for perturbing NetCDF files.

This command line tool provides functionality for:
- Creating copies of specified model files and optionally copying all files from
  a source directory to a destination directory.
- Applying perturbations to arrays in NetCDF files based on a specified
  perturbation amplitude and random seed.
"""

import os
import shutil

import click
import numpy as np

from util.click_util import CommaSeperatedInts, CommaSeperatedStrings, cli_help
from util.log_handler import logger
from util.netcdf_io import nc4_get_copy
from util.utils import get_seed_from_member_id, prepend_type_to_member_id


def perturb_array(array, seed, perturb_amplitude):
    np.random.seed(seed)
    # *2-1: map to [-1,1)
    # *perturb_amplitude: rescale to perturbation amplitude
    # +1 perturb around 1
    perturbation = (np.random.rand(*array.shape) * 2 - 1) * perturb_amplitude + 1
    return np.copy(array * perturbation)


@click.command()
@click.option("--experiment-name", help=cli_help["experiment_name"], default="")
@click.option(
    "--model-input-dir",
    help=cli_help["model_input_dir"],
)
@click.option(
    "--perturbed-model-input-dir",
    help=cli_help["perturbed_model_input_dir"],
)
@click.option(
    "--files",
    type=CommaSeperatedStrings(),
    help=cli_help["files"],
)
@click.option(
    "--member-ids",
    type=CommaSeperatedInts(),
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
    "--variable-names",
    type=CommaSeperatedStrings(),
    help=cli_help["variable_names"],
)
@click.option(
    "--perturb-amplitude",
    type=float,
    help=cli_help["perturb_amplitude"],
)
@click.option(
    "--copy-all-files/--no-copy-all-files",
    is_flag=True,
    help=cli_help["copy_all_files"],
)
def perturb(
    experiment_name,
    model_input_dir,
    perturbed_model_input_dir,
    files,
    member_ids,
    member_type,
    variable_names,
    perturb_amplitude,
    copy_all_files,
):  # pylint: disable=unused-argument, too-many-positional-arguments

    for _member_id in member_ids:

        perturbed_dir = perturbed_model_input_dir.format(
            member_id=prepend_type_to_member_id(member_type, _member_id)
        )

        model_input_dir_abspath = os.path.abspath(model_input_dir)

        # Create directory for perturbed ensemble member
        if not os.path.exists(perturbed_dir):
            logger.info("creating new directory: %s", perturbed_dir)
            os.makedirs(perturbed_dir)

        # Add perturbed files to member directory
        data = [
            nc4_get_copy(f"{model_input_dir_abspath}/{f}", f"{perturbed_dir}/{f}")
            for f in files
        ]

        for d in data:
            for vn in variable_names:
                d.variables[vn][:] = perturb_array(
                    array=d.variables[vn][:],
                    seed=get_seed_from_member_id(_member_id),
                    perturb_amplitude=perturb_amplitude,
                )
            d.close()

        # Copy rest of the files in `model_input_dir` to the perturbed ensemble
        # member dir (`perturbed_dir`)
        if copy_all_files:
            for f in os.listdir(model_input_dir_abspath):
                if (
                    f not in files
                ):  # files added manually via `files` flag already copied above
                    shutil.copy(os.path.join(model_input_dir, f), perturbed_dir)
