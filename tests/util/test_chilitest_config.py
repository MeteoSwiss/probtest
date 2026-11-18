"""
Tests for the translation of chilitest application configurations into
probtest configurations. The configurations in tests/data/chilitest/aes_amip
are copies of test/tolerance-tests/aes_amip in ICON.
"""

import json
import shutil
from pathlib import Path

import pytest

from util.chilitest_config import (
    APPLICATION_REFERENCE,
    APPLICATION_TEST,
    ConfigurationError,
    RuntimeOptions,
    load_application_config,
    probtest_config,
    validate_application_config,
)

AES_AMIP = Path(__file__).parent.parent / "data" / "chilitest" / "aes_amip"
REFERENCE_CONFIG = AES_AMIP / "aes_amip_update_ref_statistics.json"
TEST_CONFIG = AES_AMIP / "aes_amip_tolerance_test.json"


@pytest.fixture(name="runtime")
def fixture_runtime(tmp_path):
    return RuntimeOptions(
        mkexp_config=str(AES_AMIP / "aes_amip.config"),
        build_dir=str(tmp_path / "build"),
        stats_dir=str(tmp_path / "stats"),
        src_dir=str(tmp_path / "icon"),
    )


def test_load_resolves_include():
    config = load_application_config(REFERENCE_CONFIG)
    assert config["ensemble_statistics"] == {
        "variables": {"2d": ["pr", "tas", "ps", "rsut"], "3d": ["hus", "ta"]},
        "spatial_average_type": {
            "2d": "global_mean",
            "3d": "surface_mean_per_p_level",
        },
        "statistics": ["min", "max"],
    }
    assert validate_application_config(config) == APPLICATION_REFERENCE
    assert (
        validate_application_config(load_application_config(TEST_CONFIG))
        == APPLICATION_TEST
    )


def test_reference_config(runtime, tmp_path):
    config, kind = probtest_config(REFERENCE_CONFIG, runtime)
    build = tmp_path / "build"

    assert kind == APPLICATION_REFERENCE

    default = config["default"]
    assert default["experiment_name"] == "aes_amip"
    assert default["perturbed_experiment_name"] == "aes_amip_member_id_{member_id}"
    assert default["member_ids"] == list(range(1, 11))
    assert default["model_output_dir"] == f"{build}/experiments/aes_amip/work"
    assert default["file_id"] == [
        ["NetCDF", "run_*/*_2d.nc"],
        ["NetCDF", "run_*/*_3d.nc"],
    ]
    assert default["tolerance_files"] == f"{tmp_path}/stats/aes_amip_stats_{{dim}}.npz"
    assert default["npz_dimensions"] == {
        "2d": "NetCDF:run_*/*_2d.nc",
        "3d": "NetCDF:run_*/*_3d.nc",
    }

    run = config["run"]
    assert run["backend"] == "mkexp"
    assert run["mkexp_path"] == f"{tmp_path}/icon/utils/mkexp/mkexp"
    assert run["mkexp_config"] == str(AES_AMIP / "aes_amip.config")
    assert run["run_dir"] == f"{build}/run"
    assert run["run_script_name"] == (
        f"{build}/experiments/aes_amip/scripts/aes_amip.run_start"
    )
    assert run["lhs"] == [
        "namelists.NAMELIST_atm.initicon_nml.pinit_seed",
        "namelists.NAMELIST_atm.initicon_nml.pinit_amplitude",
    ]
    assert run["rhs_new"] == ["{seed}", "1e-07"]
    assert run["restart"] is False
    # what chilitest passes to mkexp for every experiment
    assert run["mkexp_option"] == [
        "namelists.NAMELIST_atm.io_nml.lnetcdf_flt64_output=.TRUE.",
        f"abs_top_srcdir={tmp_path}/icon",
        f"abs_top_builddir={build}",
        f"SCRIPT_DIR={build}/experiments/$EXP_ID/scripts",
        f"LOG_DIR={build}/experiments/$EXP_ID/log",
        f"WORK_DIR={build}/experiments/$EXP_ID/work",
        f"DATA_DIR={build}/experiments/$EXP_ID/data",
        "WRITE_COMPRESSION= ",
        "EXP_OPTIONS+=rbf_coeffs/read",
    ]

    assert config["stats"]["ensemble"] is True
    assert config["stats"]["variable_filter"] == {
        "run_*/*_2d.nc": ["pr", "tas", "ps", "rsut"],
        "run_*/*_3d.nc": ["hus", "ta"],
    }
    assert config["tolerance"]["method"] == "extreme_values"
    assert config["tolerance"]["npz_times"] == {
        "2d": f"{build}/experiments/aes_amip/work/run_*/*_2d.nc",
        "3d": f"{build}/experiments/aes_amip/work/run_*/*_3d.nc",
    }

    # usable as a json config file
    json.dumps(config)


def test_test_config(runtime, tmp_path):
    config, kind = probtest_config(TEST_CONFIG, runtime)
    stats_file = f"{tmp_path}/build/experiments/aes_amip/stats_aes_amip.csv"

    assert kind == APPLICATION_TEST
    assert config["default"]["member_ids"] == []
    assert config["run"]["lhs"] == []
    assert config["run"]["rhs_new"] == []
    assert config["stats"]["ensemble"] is False
    assert config["stats"]["stats_file_name"] == stats_file
    assert config["check"] == {
        "method": "extreme_values",
        "current_files": stats_file,
        "factor": 1.0,
    }
    assert "npz_times" not in config["tolerance"]


def test_runtime_options(runtime):
    runtime.account = "eg0000"
    runtime.environment = "levante_gpu"
    runtime.output_dirname = "run_20200101"
    runtime.mkexp_options = ["EXP_OPTIONS=my/option", "FOO=bar=baz"]
    runtime.mkexp_path = "/somewhere/mkexp"

    config, _ = probtest_config(TEST_CONFIG, runtime)

    options = config["run"]["mkexp_option"]
    assert "EXP_OPTIONS+=my/option, rbf_coeffs/read" in options
    assert "EXP_OPTIONS+=rbf_coeffs/read" not in options
    assert options[-3:] == ["FOO=bar=baz", "ACCOUNT=eg0000", "ENVIRONMENT=levante_gpu"]
    assert config["run"]["mkexp_path"] == "/somewhere/mkexp"
    assert config["default"]["file_id"][0] == ["NetCDF", "run_20200101/*_2d.nc"]


def write_config(tmp_path, source, change):
    """
    Copy of the aes_amip configurations with a modified application config.
    """
    directory = tmp_path / "config"
    shutil.copytree(AES_AMIP, directory)
    path = directory / source.name
    config = json.loads(path.read_text())
    change(config)
    path.write_text(json.dumps(config))
    return path


def test_options_from_the_config_file(runtime, tmp_path):
    def change(config):
        config["experiment"] = {"options": {"ACCOUNT": "file"}, "restart": True}
        config["file_info"] = {"stats_dir": "from_file", "output_dirname": "out"}

    path = write_config(tmp_path, TEST_CONFIG, change)

    config, _ = probtest_config(path, runtime)
    assert "ACCOUNT=file" in config["run"]["mkexp_option"]
    assert config["run"]["restart"] is True
    assert config["default"]["file_id"][0] == ["NetCDF", "out/*_2d.nc"]
    # the command line takes precedence
    assert config["default"]["tolerance_files"].startswith(f"{tmp_path}/stats/")

    runtime.account = "cli"
    runtime.stats_dir = None
    config, _ = probtest_config(path, runtime)
    assert "ACCOUNT=cli" in config["run"]["mkexp_option"]
    assert "ACCOUNT=file" not in config["run"]["mkexp_option"]
    assert "/from_file/" in config["default"]["tolerance_files"]


def test_dimensions_restrict_the_analysis(runtime, tmp_path):
    def change(config):
        config["statistics"]["dimensions"] = ["3d"]

    config, _ = probtest_config(write_config(tmp_path, TEST_CONFIG, change), runtime)
    assert config["default"]["file_id"] == [["NetCDF", "run_*/*_3d.nc"]]
    assert config["stats"]["variable_filter"] == {"run_*/*_3d.nc": ["hus", "ta"]}


def _set(section, key, value):
    def change(config):
        config.setdefault(section, {})[key] = value

    return change


@pytest.mark.parametrize(
    "source, change, message",
    [
        (TEST_CONFIG, _set("tset", "a", 1), "unknown section 'tset'.*'test'"),
        (TEST_CONFIG, _set("test", "z_score", {}), "z-score test is not supported"),
        (TEST_CONFIG, _set("test", "extreme_values", {}), "missing key 'factor'"),
        (
            TEST_CONFIG,
            _set("statistics", "vrbls", {}),
            "unknown key 'vrbls'.*'variables'",
        ),
        (
            TEST_CONFIG,
            _set("statistics", "spatial_average_type", {"2d": "x", "3d": "y"}),
            "Unsupported spatial_average_type 'x'",
        ),
        (
            TEST_CONFIG,
            _set("statistics", "variables", {"2d": ["ps"]}),
            "must have the same dimensions",
        ),
        (
            TEST_CONFIG,
            _set("experiment", "options", {"WORK_DIR": "/x"}),
            r"cannot be changed: \['WORK_DIR'\]",
        ),
        (
            REFERENCE_CONFIG,
            _set("ensemble", "perturbation_method", "input_netcdf"),
            "Unsupported perturbation_method",
        ),
        (REFERENCE_CONFIG, _set("ensemble", "size", 0), "at least 1"),
        (
            REFERENCE_CONFIG,
            _set("ensemble_statistics", "statistics", ["min"]),
            "must contain 'min' and 'max'",
        ),
        (
            REFERENCE_CONFIG,
            _set("ensemble_statistics", "statistics", ["min", "max", "std"]),
            r"Unsupported ensemble statistic\(s\) \['std'\]",
        ),
        (
            REFERENCE_CONFIG,
            _set("ensemble_statistics", "include", "missing.json"),
            "Cannot read .*missing.json included by the section",
        ),
    ],
)
def test_invalid_configurations(runtime, tmp_path, source, change, message):
    path = write_config(tmp_path, source, change)
    with pytest.raises(ConfigurationError, match=message):
        probtest_config(path, runtime)


def test_all_problems_are_reported_together(runtime, tmp_path):
    def change(config):
        config["unknown"] = {}
        del config["ensemble"]["size"]
        config["ensemble"]["sizes"] = 3

    path = write_config(tmp_path, REFERENCE_CONFIG, change)
    with pytest.raises(ConfigurationError) as error:
        probtest_config(path, runtime)
    for part in ["unknown section 'unknown'", "'sizes'", "missing key 'size'"]:
        assert part in error.value.message


def test_missing_locations(runtime):
    runtime.build_dir = None
    with pytest.raises(ConfigurationError, match="--build-dir"):
        probtest_config(TEST_CONFIG, runtime)


def test_src_dir_is_found_from_the_mkexp_config(runtime):
    """
    The configurations are inside the probtest repository here.
    """
    runtime.src_dir = None
    config, _ = probtest_config(TEST_CONFIG, runtime)
    root = Path(__file__).parent.parent.parent
    assert config["run"]["mkexp_path"] == str(root / "utils" / "mkexp" / "mkexp")


def test_src_dir_cannot_be_found(runtime, tmp_path):
    mkexp_config = tmp_path / "exp.config"
    mkexp_config.write_text("")
    runtime.src_dir = None
    runtime.mkexp_config = str(mkexp_config)
    with pytest.raises(ConfigurationError, match="--src-dir"):
        probtest_config(TEST_CONFIG, runtime)
