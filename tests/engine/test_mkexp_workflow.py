"""
End-to-end tests of the commands mkexp-reference and mkexp-test with the
chilitest configurations of the ICON experiment aes_amip
(tests/data/chilitest/aes_amip).

mkexp, sbatch and the model are replaced by stand-ins: the "model" writes
output files in the layout of the experiment, perturbed according to the
settings mkexp was called with.
"""

import json
import logging
import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from probtest import cli

AES_AMIP = Path(__file__).parent.parent / "data" / "chilitest" / "aes_amip"
N_TIME = 5
N_CELLS = 20
PLEV = [50000.0, 85000.0]
SIZE = 10
SEED_KEY = "namelists.NAMELIST_atm.initicon_nml.pinit_seed"
AMPLITUDE_KEY = "namelists.NAMELIST_atm.initicon_nml.pinit_amplitude"

# Stand-in for the model: writes <work>/run_20200101T000000/<exp>_{2d,3d}.nc
FAKE_MODEL = """\
    import os, sys

    import numpy as np
    import xarray as xr

    exp_id, work_dir, seed, amplitude, offset = sys.argv[1:]
    out_dir = os.path.join(work_dir, "run_20200101T000000")
    os.makedirs(out_dir, exist_ok=True)

    start = np.datetime64("2020-01-01T00:00", "ns")
    time = start + np.arange({n_time}) * np.timedelta64(2, "m")
    base = np.random.default_rng(0)
    noise = np.random.default_rng(int(seed))

    def field(*shape):
        values = base.normal(size=shape) + 250.0
        if int(seed):
            values = values * (1.0 + float(amplitude) * noise.uniform(-1, 1, shape))
        return values + float(offset)

    xr.Dataset(
        {{
            "pr": (("time", "ncells"), field({n_time}, {n_cells})),
            "tas": (("time", "height_2m", "ncells"), field({n_time}, 1, {n_cells})),
            "ps": (("time", "ncells"), field({n_time}, {n_cells})),
            "rsut": (("time", "ncells"), field({n_time}, {n_cells})),
            "not_analysed": (("time", "ncells"), field({n_time}, {n_cells})),
        }},
        coords={{"time": time, "height_2m": [2.0]}},
    ).to_netcdf(os.path.join(out_dir, exp_id + "_2d.nc"))

    shape = ({n_time}, {n_plev}, {n_cells})
    xr.Dataset(
        {{
            "hus": (("time", "plev", "ncells"), field(*shape)),
            "ta": (("time", "plev", "ncells"), field(*shape)),
        }},
        coords={{"time": time, "plev": {plev}}},
    ).to_netcdf(os.path.join(out_dir, exp_id + "_3d.nc"))
"""

# Stand-in for mkexp: writes the run script that starts the model
FAKE_MKEXP = """\
    #!{python}
    import json, os, sys

    config, *settings = sys.argv[1:]
    values = dict(s.replace("+=", "=").split("=", 1) for s in settings)
    exp_id = values["EXP_ID"]
    script_dir = values["SCRIPT_DIR"].replace("$EXP_ID", exp_id)
    work_dir = values["WORK_DIR"].replace("$EXP_ID", exp_id)
    os.makedirs(script_dir, exist_ok=True)

    with open(os.path.join(script_dir, "mkexp_call.json"), "w") as f:
        json.dump({{"config": config, "settings": settings}}, f)

    seed = values.get("{seed_key}", "0")
    amplitude = values.get("{amplitude_key}", "0")
    offset = values.get("TEST_OFFSET", "0")
    with open(os.path.join(script_dir, exp_id + ".run_start"), "w") as f:
        f.write(
            f"{python} {model} {{exp_id}} {{work_dir}} {{seed}} {{amplitude}} "
            f"{{offset}} && echo Experiment finished successfully "
            f"> {{script_dir}}/{{exp_id}}.run.log\\n"
        )
"""

FAKE_SBATCH = """\
    #!/bin/bash
    # sbatch --wait <script>
    exec bash "${@: -1}"
"""

FAKE_ICON = """\
    #!/bin/bash
    echo "ICON version test-1.0" >&2
"""


def _executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture(name="setup")
def fixture_setup(tmp_path, monkeypatch):
    model = tmp_path / "fake_model.py"
    model.write_text(
        textwrap.dedent(FAKE_MODEL).format(
            n_time=N_TIME, n_cells=N_CELLS, n_plev=len(PLEV), plev=PLEV
        )
    )
    mkexp = _executable(
        tmp_path / "icon" / "utils" / "mkexp" / "mkexp",
        FAKE_MKEXP.format(
            python=sys.executable,
            model=model,
            seed_key=SEED_KEY,
            amplitude_key=AMPLITUDE_KEY,
        ),
    )
    _executable(tmp_path / "bin" / "sbatch", FAKE_SBATCH)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"), prepend=":")

    build_dir = tmp_path / "build"
    (build_dir / "run").mkdir(parents=True)
    _executable(build_dir / "bin" / "icon", FAKE_ICON)

    # the log file of probtest is written to the working directory
    monkeypatch.chdir(tmp_path)

    return {
        "build_dir": build_dir,
        "stats_dir": tmp_path / "stats",
        "src_dir": mkexp.parent.parent.parent,
    }


def run(command, application_config, setup, *extra, build_dir=None):
    return CliRunner().invoke(
        cli,
        [
            command,
            str(AES_AMIP / application_config),
            "--mkexp-config",
            str(AES_AMIP / "aes_amip.config"),
            "--build-dir",
            str(build_dir or setup["build_dir"]),
            "--stats-dir",
            str(setup["stats_dir"]),
            "--src-dir",
            str(setup["src_dir"]),
            "--account",
            "eg0000",
            *extra,
        ],
    )


def run_reference(setup):
    result = run("mkexp-reference", "aes_amip_update_ref_statistics.json", setup)
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"


def run_test(setup, *extra):
    """
    Test a second build against the reference statistics.
    """
    build_dir = setup["build_dir"].parent / "test_build"
    (build_dir / "run").mkdir(parents=True, exist_ok=True)
    return run(
        "mkexp-test",
        "aes_amip_tolerance_test.json",
        setup,
        *extra,
        build_dir=build_dir,
    )


def test_reference(setup):
    run_reference(setup)

    experiments = setup["build_dir"] / "experiments"
    names = ["aes_amip"] + [f"aes_amip_member_id_{i}" for i in range(1, SIZE + 1)]
    assert sorted(p.name for p in experiments.iterdir()) == sorted(names)

    # the reference is not perturbed, the members are, each with its own seed
    seeds = set()
    for name in names:
        with open(
            experiments / name / "scripts" / "mkexp_call.json", encoding="utf-8"
        ) as f:
            call = json.load(f)
        assert call["config"] == str(AES_AMIP / "aes_amip.config")
        settings = dict(s.split("=", 1) for s in call["settings"])
        assert settings["ACCOUNT"] == "eg0000"
        assert settings["EXP_OPTIONS+"] == "rbf_coeffs/read"
        if name == "aes_amip":
            assert SEED_KEY not in settings
        else:
            assert settings[AMPLITUDE_KEY] == "1e-07"
            seeds.add(settings[SEED_KEY])
    assert len(seeds) == SIZE

    stats_dir = setup["stats_dir"]
    assert sorted(p.name for p in stats_dir.iterdir()) == [
        "aes_amip_stats_2d.npz",
        "aes_amip_stats_3d.npz",
        "aes_amip_stats_meta.json",
        "aes_amip_version.txt",
    ]
    assert (stats_dir / "aes_amip_version.txt").read_text() == "ICON version test-1.0\n"
    with open(stats_dir / "aes_amip_stats_meta.json", encoding="utf-8") as f:
        assert json.load(f) == {
            "variables": {"2d": ["pr", "tas", "ps", "rsut"], "3d": ["hus", "ta"]},
            "spatial_average_type": {
                "2d": "global_mean",
                "3d": "surface_mean_per_p_level",
            },
        }

    with np.load(stats_dir / "aes_amip_stats_2d.npz") as data:
        # only the variables of the analysis, with the time axis of the model
        assert set(data.files) == {"times"} | {
            f"{v}_{s}"
            for v in ["pr", "tas", "ps", "rsut"]
            for s in ["ref", "min", "max"]
        }
        times = np.asarray(data["times"])
        assert times.dtype.kind == "M"
        assert times[1] == np.datetime64("2020-01-01T00:02")
        assert np.shape(data["tas_ref"]) == (N_TIME,)
        assert (data["tas_min"] < data["tas_ref"]).any()
        assert (data["tas_max"] > data["tas_ref"]).any()
        assert (data["tas_min"] <= data["tas_max"]).all()
        # spread of the order of the perturbation
        assert 0 < np.abs(data["tas_max"] - data["tas_min"]).max() < 250.0 * 1e-6

    with np.load(stats_dir / "aes_amip_stats_3d.npz") as data:
        np.testing.assert_array_equal(data["plev"], PLEV)
        assert np.shape(data["ta_ref"]) == (N_TIME, len(PLEV))


def test_same_model_passes(setup, caplog):
    run_reference(setup)

    with caplog.at_level(logging.INFO):
        result = run_test(setup)
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert "RESULT: check PASSED" in caplog.text
    for variable in ["pr", "tas", "ps", "rsut"]:
        assert f"passed: NetCDF:run_*/*_2d.nc {variable}" in caplog.text
    for variable in ["hus", "ta"]:
        assert f"passed: NetCDF:run_*/*_3d.nc {variable}" in caplog.text
    assert "not_analysed" not in caplog.text

    # only the unperturbed experiment is run
    experiments = setup["build_dir"].parent / "test_build" / "experiments"
    assert [p.name for p in experiments.iterdir()] == ["aes_amip"]


def test_changed_model_fails(setup, caplog):
    run_reference(setup)

    with caplog.at_level(logging.INFO):
        result = run_test(setup, "--mkexp-option", "TEST_OFFSET=0.01")
    assert result.exit_code == 1
    assert "RESULT: check FAILED" in caplog.text
    # all variables are reported, not only the first one
    assert f"pr ({N_TIME} of {N_TIME} values" in caplog.text
    total = N_TIME * len(PLEV)
    assert f"ta ({total} of {total} values" in caplog.text
    assert "Values outside the ensemble range" in caplog.text


def test_wrong_application(setup):
    result = run("mkexp-test", "aes_amip_update_ref_statistics.json", setup)
    assert result.exit_code != 0
    assert "use 'mkexp-reference'" in result.output


def test_init_writes_the_configuration(setup):
    """
    The configuration the one-shot commands use can be written to a file to
    run the commands one by one.
    """
    result = CliRunner().invoke(
        cli,
        [
            "init",
            "--chilitest-config",
            str(AES_AMIP / "aes_amip_tolerance_test.json"),
            "--mkexp-config",
            str(AES_AMIP / "aes_amip.config"),
            "--build-dir",
            str(setup["build_dir"]),
            "--stats-dir",
            str(setup["stats_dir"]),
            "--src-dir",
            str(setup["src_dir"]),
            "--config",
            "mkexp_test.json",
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    with open("mkexp_test.json", encoding="utf-8") as f:
        config = json.load(f)
    assert config["check"]["method"] == "extreme_values"
    assert config["check"]["factor"] == 1.0
    assert config["run"]["backend"] == "mkexp"
