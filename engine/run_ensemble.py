"""
CLI for running ensembles

This module provides a command-line interface (CLI) for running ensemble
simulations with perturbed configurations.
It allows users to execute a series of simulation runs by modifying the run
scripts with specified perturbations and managing job submissions.
"""

import os
import subprocess
import time
from pathlib import Path

import click

from util.click_util import CommaSeparatedInts, CommaSeparatedStrings, cli_help
from util.log_handler import logger
from util.run_backends import BACKEND_MKEXP, BACKEND_RUNSCRIPT, RUN_BACKENDS
from util.run_backends.mkexp import (
    check_success_marker,
    generate_mkexp_runscript,
    perturbation_arguments,
    run_log_file,
)
from util.run_backends.runscript import prepare_perturbed_run_script
from util.utils import get_seed_from_member_id, prepend_type_to_member_id


def append_job(job, job_list, parallel):
    p = subprocess.Popen(job)  # pylint: disable=consider-using-with
    if not parallel:
        try:
            time.sleep(5)
            p.wait()
            check_job_returncode(p)
        finally:
            p.kill()
    else:
        job_list.append(p)


def finalize_jobs(job_list, dry, parallel):
    if parallel and not dry:
        last_exception = None
        for job in job_list:
            job.communicate()
            try:
                check_job_returncode(job)
            except subprocess.CalledProcessError as e:
                logger.error(e)
                last_exception = e
        if last_exception:
            raise last_exception


def check_job_returncode(job):
    """Test job return code."""
    if job.returncode != 0:
        raise subprocess.CalledProcessError(returncode=job.returncode, cmd=job.args)


@click.command()
@click.option(
    "--run-dir",
    help=cli_help["run_dir"],
)
@click.option(
    "--perturbed-run-dir",
    help=cli_help["perturbed_run_dir"],
)
@click.option(
    "--run-script-name",
    help=cli_help["run_script_name"],
)
@click.option(
    "--perturbed-run-script-name",
    help=cli_help["perturbed_run_script_name"],
)
@click.option(
    "--experiment-name",
    help=cli_help["experiment_name"],
)
@click.option(
    "--perturbed-experiment-name",
    help=cli_help["perturbed_run_script_name"],
)
@click.option(
    "--submit-command",
    help=cli_help["submit_command"],
)
@click.option(
    "--member-ids",
    default="1,2,3,4,5,6,7,8,9,10",
    type=CommaSeparatedInts(),
    help=cli_help["member_ids"],
)
@click.option(
    "--member-type",
    type=str,
    default="",
    help=cli_help["member_type"],
)
@click.option(
    "--parallel/--no-parallel",
    is_flag=True,
    help=cli_help["parallel"],
)
@click.option(
    "--dry/--no-dry",
    is_flag=True,
    help=cli_help["dry"],
)
@click.option(
    "--lhs",
    type=CommaSeparatedStrings(),
    help=cli_help["lhs"],
)
@click.option(
    "--rhs-new",
    type=CommaSeparatedStrings(),
    help=cli_help["rhs_new"],
)
@click.option(
    "--rhs-old",
    type=CommaSeparatedStrings(),
    help=cli_help["rhs_old"],
)
@click.option(
    "--backend",
    type=click.Choice(RUN_BACKENDS),
    default=BACKEND_RUNSCRIPT,
    help=cli_help["run_backend"],
)
@click.option(
    "--mkexp-path",
    help=cli_help["mkexp_path"],
)
@click.option(
    "--mkexp-config",
    help=cli_help["mkexp_config"],
)
@click.option(
    "--mkexp-option",
    multiple=True,
    metavar="KEY=VALUE",
    help=cli_help["mkexp_option"],
)
@click.option(
    "--restart/--no-restart",
    is_flag=True,
    help=cli_help["restart"],
)
@click.option(
    "--success-marker",
    default=None,
    help=cli_help["success_marker"],
)
@click.option(
    "--run-log-name",
    default="{experiment_name}.run.log",
    help=cli_help["run_log_name"],
)
def run_ensemble(
    run_dir,
    perturbed_run_dir,
    run_script_name,
    perturbed_run_script_name,
    experiment_name,
    perturbed_experiment_name,
    submit_command,
    member_ids,
    member_type,
    parallel,
    dry,
    lhs,
    rhs_new,
    rhs_old,
    backend,
    mkexp_path,
    mkexp_config,
    mkexp_option,
    restart,
    success_marker,
    run_log_name,
):  # pylint: disable=too-many-positional-arguments
    if backend == BACKEND_MKEXP:
        run_ensemble_mkexp(
            run_dir=run_dir,
            run_script_name=run_script_name,
            perturbed_run_script_name=perturbed_run_script_name,
            experiment_name=experiment_name,
            perturbed_experiment_name=perturbed_experiment_name,
            submit_command=submit_command,
            member_ids=member_ids,
            member_type=member_type,
            parallel=parallel,
            dry=dry,
            lhs=lhs,
            rhs_new=rhs_new,
            mkexp_path=mkexp_path,
            mkexp_config=mkexp_config,
            mkexp_options=list(mkexp_option),
            restart=restart,
            success_marker=success_marker,
            run_log_name=run_log_name,
        )
        return

    perturbed_run_dir = perturbed_run_dir if perturbed_run_dir else run_dir
    os.chdir(run_dir)
    job_list = []

    # run the reference
    if not dry:
        logger.info("running unperturbed reference")
        job = submit_command.split() + [run_script_name]
        append_job(job, job_list, parallel)

    # run the ensemble
    for member_id in member_ids:

        Path(perturbed_run_dir.format(member_id=str(member_id))).mkdir(
            exist_ok=True, parents=True
        )
        os.chdir(perturbed_run_dir.format(member_id=str(member_id)))

        runscript = f"{run_dir}/{run_script_name}"

        typed_member_id = prepend_type_to_member_id(member_type, member_id)

        perturbed_run_dir_path = perturbed_run_dir.format(member_id=typed_member_id)
        perturbed_run_script_path = perturbed_run_script_name.format(
            member_id=typed_member_id
        )
        perturbed_runscript = f"{perturbed_run_dir_path}/{perturbed_run_script_path}"

        prepare_perturbed_run_script(
            runscript,
            perturbed_runscript,
            experiment_name,
            perturbed_experiment_name.format(member_id=typed_member_id),
            lhs,
            rhs_new,
            rhs_old,
            get_seed_from_member_id(member_id),
        )

        if not dry:
            job = submit_command.split() + [perturbed_runscript]
            logger.info("running the model with '%s'", " ".join(job))
            append_job(job, job_list, parallel)

    finalize_jobs(job_list, dry, parallel)

    logger.info("model finished!")


def run_ensemble_mkexp(
    *,
    run_dir,
    run_script_name,
    perturbed_run_script_name,
    experiment_name,
    perturbed_experiment_name,
    submit_command,
    member_ids,
    member_type,
    parallel,
    dry,
    lhs,
    rhs_new,
    mkexp_path,
    mkexp_config,
    mkexp_options,
    restart,
    success_marker,
    run_log_name,
):
    """
    Generate the run scripts of the unperturbed reference and of the members
    with mkexp and submit them.

    run_script_name and perturbed_run_script_name are the scripts mkexp
    generates, relative to run_dir or absolute.
    """
    missing = [
        name
        for name, value in [
            ("--mkexp-path", mkexp_path),
            ("--mkexp-config", mkexp_config),
            ("--run-dir", run_dir),
        ]
        if not value
    ]
    if missing:
        raise click.UsageError(
            f"The mkexp backend requires the option(s) {', '.join(missing)}."
        )

    run_dir = os.path.abspath(run_dir)
    os.chdir(run_dir)

    # experiment name, run script and the mkexp arguments of the perturbation
    experiments = [(experiment_name, os.path.join(run_dir, run_script_name), [])]
    for member_id in member_ids:
        typed_member_id = prepend_type_to_member_id(member_type, member_id)
        experiments.append(
            (
                perturbed_experiment_name.format(member_id=typed_member_id),
                os.path.join(
                    run_dir, perturbed_run_script_name.format(member_id=typed_member_id)
                ),
                perturbation_arguments(
                    lhs, rhs_new, get_seed_from_member_id(member_id)
                ),
            )
        )

    job_list = []
    for name, runscript, perturbation in experiments:
        generate_mkexp_runscript(
            mkexp_path=mkexp_path,
            mkexp_config=mkexp_config,
            experiment_name=name,
            options=mkexp_options + perturbation,
            mkexp_search_path=run_dir,
            runscript=runscript,
            restart=restart,
        )

        if not dry:
            job = submit_command.split() + [runscript]
            logger.info("running the model with '%s'", " ".join(job))
            append_job(job, job_list, parallel)

    finalize_jobs(job_list, dry, parallel)

    if success_marker and not dry:
        failed = [
            log_file
            for name, runscript, _ in experiments
            for log_file in [run_log_file(runscript, run_log_name, name)]
            if not check_success_marker(log_file, success_marker)
        ]
        if failed:
            raise RuntimeError(
                f"'{success_marker}' not found in the log file(s) of the "
                f"following run(s): {', '.join(failed)}"
            )

    logger.info("model finished!")
