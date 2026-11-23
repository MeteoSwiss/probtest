"""
Tests for the mkexp backend of the run-ensemble command. mkexp itself is
replaced by a small stand-in script.
"""

import json
import stat
import sys
import textwrap

import pytest
from click.testing import CliRunner

from engine.run_ensemble import run_ensemble
from util.run_backends.mkexp import (
    check_success_marker,
    mkexp_command,
    perturbation_arguments,
)
from util.utils import get_seed_from_member_id

SUCCESS = "Experiment finished successfully"

# Stand-in for mkexp: records how it was called and writes
# <SCRIPT_DIR>/<EXP_ID>.run_start, which appends to <EXP_ID>.run.log.
FAKE_MKEXP = """\
    #!{python}
    import json, os, sys

    config, *settings = sys.argv[1:]
    values = dict(s.split("=", 1) for s in settings)
    exp_id = values["EXP_ID"]
    script_dir = values["SCRIPT_DIR"].replace("$EXP_ID", exp_id)
    os.makedirs(script_dir, exist_ok=True)

    with open(os.path.join(script_dir, "mkexp_call.json"), "w") as f:
        json.dump(
            {{
                "config": config,
                "settings": settings,
                "MKEXP_PATH": os.environ.get("MKEXP_PATH"),
            }},
            f,
        )

    message = "{success}" if values.get("FAIL_FOR") != exp_id else "crashed"
    with open(os.path.join(script_dir, exp_id + ".run_start"), "w") as f:
        f.write(f"echo {{message}} >> {{script_dir}}/{{exp_id}}.run.log\\n")
"""


@pytest.fixture(name="setup")
def fixture_setup(tmp_path):
    mkexp = tmp_path / "mkexp"
    mkexp.write_text(
        textwrap.dedent(FAKE_MKEXP).format(python=sys.executable, success=SUCCESS)
    )
    mkexp.chmod(mkexp.stat().st_mode | stat.S_IEXEC)

    build_dir = tmp_path / "build"
    (build_dir / "run").mkdir(parents=True)
    config = tmp_path / "exp.config"
    config.write_text("EXP_TYPE = test\n")

    return mkexp, config, build_dir


def run(setup, *extra, member_ids="1,2"):
    mkexp, config, build_dir = setup
    scripts = f"{build_dir}/experiments/{{name}}/scripts/{{name}}.run_start"
    member = "exp_member_id_{member_id}"
    return CliRunner().invoke(
        run_ensemble,
        [
            "--backend",
            "mkexp",
            "--mkexp-path",
            str(mkexp),
            "--mkexp-config",
            str(config),
            "--mkexp-option",
            f"SCRIPT_DIR={build_dir}/experiments/$EXP_ID/scripts",
            "--mkexp-option",
            "EXP_OPTIONS+=rbf_coeffs/read",
            "--run-dir",
            str(build_dir / "run"),
            "--experiment-name",
            "exp",
            "--perturbed-experiment-name",
            member,
            "--run-script-name",
            scripts.format(name="exp"),
            "--perturbed-run-script-name",
            scripts.format(name=member),
            "--member-ids",
            member_ids,
            "--lhs",
            "namelists.NAMELIST_atm.initicon_nml.pinit_seed,"
            "namelists.NAMELIST_atm.initicon_nml.pinit_amplitude",
            "--rhs-new",
            "{seed},1e-07",
            "--submit-command",
            "bash",
            "--parallel",
            "--success-marker",
            SUCCESS,
            *extra,
        ],
    )


def script_dir(setup, name):
    return setup[2] / "experiments" / name / "scripts"


def mkexp_call(setup, name):
    with open(script_dir(setup, name) / "mkexp_call.json", encoding="utf-8") as f:
        return json.load(f)


def test_reference_and_members_are_generated_and_run(setup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _, config, build_dir = setup

    result = run(setup)
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"

    common = [
        f"SCRIPT_DIR={build_dir}/experiments/$EXP_ID/scripts",
        "EXP_OPTIONS+=rbf_coeffs/read",
    ]

    call = mkexp_call(setup, "exp")
    assert call["config"] == str(config)
    assert call["MKEXP_PATH"] == str(build_dir / "run")
    # the reference is not perturbed
    assert call["settings"] == ["EXP_ID=exp", *common]

    for member_id in [1, 2]:
        name = f"exp_member_id_{member_id}"
        seed = get_seed_from_member_id(member_id)
        assert mkexp_call(setup, name)["settings"] == [
            f"EXP_ID={name}",
            *common,
            f"namelists.NAMELIST_atm.initicon_nml.pinit_seed={seed}",
            "namelists.NAMELIST_atm.initicon_nml.pinit_amplitude=1e-07",
        ]

    for name in ["exp", "exp_member_id_1", "exp_member_id_2"]:
        log = script_dir(setup, name) / f"{name}.run.log"
        assert log.read_text() == SUCCESS + "\n"


def test_reference_only(setup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = run(setup, member_ids="")
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert (script_dir(setup, "exp") / "exp.run.log").exists()
    assert not (setup[2] / "experiments" / "exp_member_id_1").exists()


def test_dry_generates_but_does_not_run(setup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = run(setup, "--dry")
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    for name in ["exp", "exp_member_id_1", "exp_member_id_2"]:
        assert (script_dir(setup, name) / f"{name}.run_start").exists()
        assert not (script_dir(setup, name) / f"{name}.run.log").exists()


def test_missing_success_marker_fails(setup, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = run(setup, "--mkexp-option", "FAIL_FOR=exp_member_id_2")
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "exp_member_id_2.run.log" in str(result.exception)
    assert "exp_member_id_1.run.log" not in str(result.exception)


@pytest.mark.parametrize("restart", [False, True])
def test_date_file_is_removed_unless_restart(setup, monkeypatch, tmp_path, restart):
    monkeypatch.chdir(tmp_path)
    date_file = script_dir(setup, "exp") / "exp.date"
    date_file.parent.mkdir(parents=True)
    date_file.write_text("20200101\n")

    result = run(setup, *(["--restart"] if restart else []), member_ids="")
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert date_file.exists() == restart


def test_run_script_not_created(setup, monkeypatch, tmp_path):
    """
    The run script name does not match what mkexp generates.
    """
    monkeypatch.chdir(tmp_path)
    result = run(setup, "--run-script-name", "somewhere/else.run_start", "--dry")
    assert isinstance(result.exception, RuntimeError)
    assert "did not create the run script" in str(result.exception)


def test_required_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(run_ensemble, ["--backend", "mkexp"])
    assert result.exit_code == 2
    assert "--mkexp-path" in result.output


def test_perturbation_arguments():
    assert perturbation_arguments(["a.b", "c"], ["{seed}", "1e-7"], 42) == [
        "a.b=42",
        "c=1e-7",
    ]
    assert not perturbation_arguments(None, None, 42)
    with pytest.raises(ValueError):
        perturbation_arguments(["a"], ["1", "2"], 42)


def test_mkexp_command_rejects_exp_id_and_malformed_options():
    assert mkexp_command("mkexp", "c", "exp", ["A=1", "B+=2"]) == [
        "mkexp",
        "c",
        "EXP_ID=exp",
        "A=1",
        "B+=2",
    ]
    with pytest.raises(ValueError):
        mkexp_command("mkexp", "c", "exp", ["EXP_ID=other"])
    with pytest.raises(ValueError):
        mkexp_command("mkexp", "c", "exp", ["no_value"])


def test_check_success_marker(tmp_path):
    log = tmp_path / "run.log"
    assert not check_success_marker(str(log), SUCCESS)
    log.write_text("start\n" + "x" * 2_000_000 + f"\n{SUCCESS}\nepilogue\n")
    assert check_success_marker(str(log), SUCCESS)
    log.write_text(f"{SUCCESS}\n" + "x" * 2_000_000)
    assert not check_success_marker(str(log), SUCCESS)
