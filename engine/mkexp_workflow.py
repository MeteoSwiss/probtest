"""
CLI for running the tolerance test workflow of chilitest for experiments that
are set up with mkexp.

Two commands take the application configuration files of chilitest
(utils/chilitest in ICON) and chain the commands run-ensemble, stats,
tolerance and check:

mkexp-reference
    Run an ensemble with the reference model and store its statistics.

mkexp-test
    Run the experiment with the model to test and check it against the
    reference statistics with the extreme-values (min/max) test.
"""

import copy
import os
import subprocess
import sys

import click

from engine.check import check
from engine.run_ensemble import run_ensemble
from engine.stats import stats
from engine.tolerance import tolerance
from util.chilitest_config import (
    APPLICATION_REFERENCE,
    APPLICATION_TEST,
    ConfigurationError,
    RuntimeOptions,
    probtest_config,
    runtime_options,
)
from util.click_util import merge_defaults
from util.log_handler import logger


def run_command(command, config):
    """
    Run a probtest command with the defaults from config, as if config was the
    json config file and the command was called without options.
    """
    # a copy, because commands may modify their arguments (stats appends to
    # member_ids)
    defaults = copy.deepcopy(merge_defaults(config, command.name.split("-")))
    logger.info("--- probtest %s", command.name)
    try:
        command.main(args=[], standalone_mode=False, default_map=defaults)
    except SystemExit as e:
        if e.code not in (0, None):
            raise


def save_version_info(command, version_file):
    """
    Store the version information of the model next to the statistics.
    """
    if not os.path.isfile(command[0]):
        logger.warning(
            "%s not found, the model version is not stored in %s",
            command[0],
            version_file,
        )
        return

    process = subprocess.run(command, check=True, capture_output=True, text=True)
    os.makedirs(os.path.dirname(version_file), exist_ok=True)
    with open(version_file, "w", encoding="utf-8") as f:
        # ICON prints its version information to stderr
        f.write(process.stderr or process.stdout)
    logger.info("model version written to %s", version_file)


def _config(application_config_file, expected_kind, **options):
    config, kind = probtest_config(application_config_file, RuntimeOptions(**options))
    if kind != expected_kind:
        other = "mkexp-test" if kind == APPLICATION_TEST else "mkexp-reference"
        raise ConfigurationError(
            f"{application_config_file} is the configuration of the other "
            f"application, use '{other}'."
        )
    return config


@click.command()
@click.argument("application_config_file", type=click.Path(exists=True))
@runtime_options
def mkexp_reference(application_config_file, **options):
    """
    Run an ensemble of an experiment set up with mkexp and store its
    statistics as the reference for mkexp-test.

    APPLICATION_CONFIG_FILE is a chilitest application configuration for the
    update of the reference statistics (<experiment>_update_ref_statistics.json).
    """
    config = _config(application_config_file, APPLICATION_REFERENCE, **options)

    run_command(run_ensemble, config)
    run_command(stats, config)
    run_command(tolerance, config)

    workflow = config["mkexp_workflow"]
    save_version_info(workflow["version_command"], workflow["version_file"])
    logger.info(
        "reference statistics written to %s",
        os.path.dirname(config["default"]["tolerance_files"]),
    )


@click.command()
@click.argument("application_config_file", type=click.Path(exists=True))
@runtime_options
def mkexp_test(application_config_file, **options):
    """
    Run an experiment set up with mkexp and check it against the reference
    statistics written by mkexp-reference.

    APPLICATION_CONFIG_FILE is a chilitest application configuration for the
    tolerance test (<experiment>_tolerance_test.json).
    """
    config = _config(application_config_file, APPLICATION_TEST, **options)

    run_command(run_ensemble, config)
    run_command(stats, config)
    try:
        run_command(check, config)
    except SystemExit as e:
        sys.exit(e.code)
