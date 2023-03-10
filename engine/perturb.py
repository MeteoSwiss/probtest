import os
import shutil

import click
import numpy as np

from util.click_util import CommaSeperatedStrings, cli_help
from util.log_handler import logger
from util.netcdf_io import nc4_get_copy
from util.utils import generate_seed_from_member_id


def create_perturb_files(in_path, in_files, out_path, copy_all_files=False):
    path = os.path.abspath(in_path)
    if not os.path.exists(out_path):
        logger.info("creating new directory: {}".format(out_path))
        os.makedirs(out_path)
    data = [
        nc4_get_copy("{}/{}".format(path, f), "{}/{}".format(out_path, f))
        for f in in_files
    ]

    if copy_all_files:
        all_files = os.listdir(path)
        # disregard the input files which are copied above
        other_files = [f for f in all_files if f not in in_files]
        # copy all other files
        for f in other_files:
            shutil.copy("{}/{}".format(in_path, f), out_path)

    return data


def perturb_array(array, s, a):
    shape = array.shape
    np.random.seed(s)
    p = (
        np.random.rand(*shape) * 2 - 1
    ) * a + 1  # *2-1: map to [-1,1), *a: rescale to amplitude, +1 perturb around 1
    parray = np.copy(array * p)
    return parray


@click.command()
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
    "--member_ids",
    type=CommaSeperatedStrings(),
    help=cli_help["member_ids"],
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
    model_input_dir,
    perturbed_model_input_dir,
    files,
    member_ids,
    variable_names,
    perturb_amplitude,
    copy_all_files,
):
    for m_id in member_ids:
        perturbed_model_input_dir_member_id = perturbed_model_input_dir.format(
            member_id=m_id
        )
        data = create_perturb_files(
            model_input_dir,
            files,
            perturbed_model_input_dir_member_id,
            copy_all_files,
        )
        for d in data:
            for vn in variable_names:
                d.variables[vn][:] = perturb_array(
                    d.variables[vn][:],
                    generate_seed_from_member_id(m_id, use_64_bits=False),
                    perturb_amplitude,
                )
            d.close()
