"""
This module contains unit tests for the `engine.run_ensemble` module functions,
specifically `append_job` and `finalize_jobs`.
It uses the `unittest` framework to ensure the correctness of these functions
under different scenarios.
"""

import subprocess
from pathlib import Path

import pytest

from engine.run_ensemble import append_job, finalize_jobs, prepare_perturbed_run_script


def test_job_successful_serial():
    job = "test 1 = 1".split()
    append_job(job=job, job_list=[], parallel=False)


def test_job_successful_parallel():
    job = "test 1 = 1".split()
    job_list = []

    append_job(job=job, job_list=job_list, parallel=True)
    finalize_jobs(job_list=job_list, dry=False, parallel=True)


def test_job_failed_serial():
    job = "test 1 = 2".split()

    with pytest.raises(subprocess.CalledProcessError):
        append_job(job=job, job_list=[], parallel=False)


def test_job_failed_parallel():
    job = "test 1 = 2".split()
    job_list = []

    append_job(job=job, job_list=job_list, parallel=True)

    with pytest.raises(subprocess.CalledProcessError):
        finalize_jobs(job_list=job_list, dry=False, parallel=True)


def test_prepare_perturbed_run_script_replaces_assignment_and_experiment(
    tmp_path: Path,
) -> None:
    runscript = tmp_path / "exp.run"
    perturbed = tmp_path / "exp.perturbed.run"

    runscript.write_text(
        "EXPERIMENT=base_exp\n" "FOO=old_value\n" "echo done\n",
        encoding="utf-8",
    )

    prepare_perturbed_run_script(
        runscript=str(runscript),
        perturbed_runscript=str(perturbed),
        experiment_name="base_exp",
        modified_experiment_name="perturbed_exp",
        lhs=["FOO"],
        rhs_new=["new_value_{seed}"],
        rhs_old=["old_value"],
        seed=42,
    )

    # verify
    content = perturbed.read_text(encoding="utf-8")

    assert "FOO=new_value_42" in content
    assert "EXPERIMENT=perturbed_exp" in content
    assert "echo done" in content


def test_prepare_perturbed_run_script_no_modifications(tmp_path: Path) -> None:
    runscript = tmp_path / "exp.run"
    perturbed = tmp_path / "exp.perturbed.run"

    runscript.write_text(
        "EXPERIMENT=base_exp\n" "FOO=old_value\n" "echo done\n",
        encoding="utf-8",
    )

    # Expect RuntimeError because nothing should be modified
    with pytest.raises(RuntimeError, match="No lines were modified"):
        prepare_perturbed_run_script(
            runscript=str(runscript),
            perturbed_runscript=str(perturbed),
            experiment_name="base_exp",
            modified_experiment_name="perturbed_exp",
            lhs=["FOOO"],
            rhs_new=["new_value_{seed}"],
            rhs_old=["old_value"],
            seed=42,
        )
