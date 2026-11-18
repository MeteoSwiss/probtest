"""
Compatibility with chilitest (utils/chilitest in ICON): translate a chilitest
application configuration into a probtest configuration.

chilitest has two applications, each with a json configuration file:

- updating the reference statistics from an ensemble (sections ``ensemble``
  and ``ensemble_statistics``)
- testing an experiment against the reference statistics (sections
  ``statistics`` and ``test``)

Together with the mkexp configuration of the experiment and the build and
statistics directories, such a file is translated into the configuration the
probtest commands run-ensemble, stats, tolerance and check read their defaults
from. The ICON-specific settings come from a template (templates/ICON_mkexp.jinja).
"""

import difflib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import click
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from util.ensemble_stats_io import SPATIAL_AVERAGE_TYPES

APPLICATION_REFERENCE = "reference"
APPLICATION_TEST = "test"

DEFAULT_TEMPLATE = Path(__file__).parent.parent / "templates" / "ICON_mkexp.jinja"
DEFAULT_OUTPUT_DIRNAME = "run_*"
PERTURBATION_METHOD = "pinit_nml_params"

_ANALYSIS_KEYS = {"variables", "spatial_average_type", "dimensions"}
_COMMON_SCHEMA = {
    "file_info": {"mkexp_config_path", "build_dir", "stats_dir", "output_dirname"},
    "experiment": {"options", "restart"},
}
# allowed keys per section, and the required ones
_SCHEMA = {
    APPLICATION_REFERENCE: {
        **_COMMON_SCHEMA,
        "ensemble": {"perturbation_method", "size", "perturbation_amplitude"},
        "ensemble_statistics": _ANALYSIS_KEYS | {"statistics"},
    },
    APPLICATION_TEST: {
        **_COMMON_SCHEMA,
        "statistics": _ANALYSIS_KEYS,
        "test": {"extreme_values"},
        "test.extreme_values": {"factor"},
    },
}
_REQUIRED = {
    APPLICATION_REFERENCE: {
        "ensemble": {"perturbation_method", "size", "perturbation_amplitude"},
        "ensemble_statistics": {"variables", "spatial_average_type", "statistics"},
    },
    APPLICATION_TEST: {
        "statistics": {"variables", "spatial_average_type"},
        "test": {"extreme_values"},
        "test.extreme_values": {"factor"},
    },
}


class ConfigurationError(click.ClickException):
    """
    Invalid chilitest application configuration.
    """


@dataclass
class RuntimeOptions:  # pylint: disable=too-many-instance-attributes
    """
    Information that is not part of the application configuration file: where
    things are and how the experiments are submitted.
    """

    mkexp_config: Optional[str] = None
    build_dir: Optional[str] = None
    stats_dir: Optional[str] = None
    account: Optional[str] = None
    environment: Optional[str] = None
    output_dirname: Optional[str] = None
    mkexp_options: list = field(default_factory=list)
    mkexp_path: Optional[str] = None
    src_dir: Optional[str] = None
    template: Optional[str] = None


def runtime_options(function):
    """
    Click options for the RuntimeOptions, shared by the commands that take a
    chilitest application configuration.
    """
    options = [
        click.option(
            "--mkexp-config",
            help="mkexp configuration of the experiment (<experiment>.config). "
            "The experiment is named after this file.",
        ),
        click.option(
            "--build-dir",
            help="ICON build directory. The experiments are run in "
            "<build-dir>/experiments.",
        ),
        click.option(
            "--stats-dir",
            help="Directory of the reference statistics (written by "
            "mkexp-reference, read by mkexp-test).",
        ),
        click.option("--account", help="Account the jobs are charged to."),
        click.option(
            "--environment",
            help="mkexp environment, e.g. levante_gpu for a GPU build "
            "(default: chosen by mkexp for the host).",
        ),
        click.option(
            "--output-dirname",
            help="Name of the output directory inside the work directory of an "
            f"experiment (default: {DEFAULT_OUTPUT_DIRNAME}).",
        ),
        click.option(
            "--mkexp-option",
            "mkexp_options",
            multiple=True,
            metavar="KEY=VALUE",
            help="Additional setting passed to mkexp. Can be given multiple times.",
        ),
        click.option(
            "--mkexp-path",
            help="Path to the mkexp executable "
            "(default: utils/mkexp/mkexp in the ICON source directory).",
        ),
        click.option(
            "--src-dir",
            help="ICON source directory (default: the git repository that "
            "contains the mkexp configuration).",
        ),
        click.option(
            "--mkexp-template",
            "template",
            help="Template of the probtest configuration for experiments run "
            f"with mkexp (default: {DEFAULT_TEMPLATE}).",
        ),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def load_application_config(path):
    """
    Read a chilitest application configuration file.

    A top-level section may contain ``"include": "<file>"`` (relative to the
    configuration file). The content of that file provides defaults for the
    section.
    """
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigurationError(f"Cannot read {path}: {e}") from e

    if not isinstance(config, dict):
        raise ConfigurationError(f"{path} must contain a json object.")

    for name, section in config.items():
        if isinstance(section, dict) and "include" in section:
            include_path = path.parent / section["include"]
            try:
                with open(include_path, encoding="utf-8") as f:
                    included = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                raise ConfigurationError(
                    f"Cannot read {include_path} included by the section "
                    f"'{name}' of {path}: {e}"
                ) from e
            if not isinstance(included, dict):
                raise ConfigurationError(f"{include_path} must contain a json object.")
            merged = {**included, **section}
            del merged["include"]
            config[name] = merged

    return config


def _did_you_mean(name, candidates):
    close = difflib.get_close_matches(name, candidates, n=1)
    return f" (did you mean '{close[0]}'?)" if close else ""


def application_kind(config):
    if "ensemble" in config or "ensemble_statistics" in config:
        return APPLICATION_REFERENCE
    if "test" in config or "statistics" in config:
        return APPLICATION_TEST
    raise ConfigurationError(
        "Cannot tell which application the configuration is for: it has neither "
        "the sections 'ensemble' and 'ensemble_statistics' (update of the "
        "reference statistics) nor 'statistics' and 'test' (tolerance test)."
    )


def _section(config, path):
    section = config
    for part in path.split("."):
        if not isinstance(section, dict) or part not in section:
            return None
        section = section[part]
    return section


def validate_application_config(config):
    """
    Check the sections and keys of an application configuration and return the
    kind of application. All problems are reported together.
    """
    kind = application_kind(config)
    schema = _SCHEMA[kind]
    problems = []

    top_level = {path for path in schema if "." not in path}
    for name in config:
        if name not in top_level:
            problems.append(f"unknown section '{name}'{_did_you_mean(name, top_level)}")

    for path, allowed in schema.items():
        section = _section(config, path)
        required = _REQUIRED[kind].get(path)
        if section is None:
            if required and (
                "." not in path or _section(config, path.rsplit(".", 1)[0])
            ):
                problems.append(f"missing section '{path}'")
            continue
        if not isinstance(section, dict):
            problems.append(f"section '{path}' must be a json object")
            continue
        for key in section:
            if key not in allowed:
                hint = _did_you_mean(key, allowed)
                if path == "test" and key == "z_score":
                    hint = " (the z-score test is not supported)"
                problems.append(f"unknown key '{key}' in section '{path}'{hint}")
        for key in sorted((required or set()) - set(section)):
            problems.append(f"missing key '{key}' in section '{path}'")

    if problems:
        raise ConfigurationError(
            "Invalid application configuration:\n  - " + "\n  - ".join(problems)
        )
    return kind


def _analysis(config, section_name):
    """
    Variables per dimension from the analysis definition of a section.
    """
    section = config[section_name]
    variables = section["variables"]
    spatial_average_type = section["spatial_average_type"]

    if set(variables) != set(spatial_average_type):
        raise ConfigurationError(
            f"'variables' and 'spatial_average_type' of the section "
            f"'{section_name}' must have the same dimensions, got "
            f"{sorted(variables)} and {sorted(spatial_average_type)}."
        )
    for dim, average in spatial_average_type.items():
        if SPATIAL_AVERAGE_TYPES.get(dim) != average:
            supported = ", ".join(f"{d}: {a}" for d, a in SPATIAL_AVERAGE_TYPES.items())
            raise ConfigurationError(
                f"Unsupported spatial_average_type '{average}' for the dimension "
                f"'{dim}'. Supported: {supported}."
            )

    dimensions = section.get("dimensions") or list(variables)
    unknown = [dim for dim in dimensions if dim not in variables]
    if unknown:
        raise ConfigurationError(
            f"The dimension(s) {unknown} of the section '{section_name}' "
            f"have no variables."
        )
    return {dim: list(variables[dim]) for dim in dimensions}


def _find_src_dir(mkexp_config):
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                os.path.dirname(mkexp_config),
                "rev-parse",
                "--show-toplevel",
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise ConfigurationError(
            f"Cannot find the ICON source directory: {mkexp_config} is not inside "
            "a git repository. Use --src-dir."
        ) from e


def _merge_mkexp_options(defaults, user_options):
    """
    Add the user's mkexp settings (dict) to the default ones (KEY=VALUE list).

    The defaults cannot be overridden, except for EXP_OPTIONS, where the
    default is appended to the user's value.
    """

    def key_of(option):
        return option.split("=", 1)[0].rstrip("+")

    default_keys = {key_of(option) for option in defaults}
    clash = sorted((default_keys | {"EXP_ID"}) & (set(user_options) - {"EXP_OPTIONS"}))
    if clash:
        raise ConfigurationError(
            f"The following mkexp settings are set by probtest and cannot be "
            f"changed: {clash}"
        )

    merged = []
    for option in defaults:
        if key_of(option) == "EXP_OPTIONS" and "EXP_OPTIONS" in user_options:
            value = option.split("=", 1)[1]
            option = f"EXP_OPTIONS+={user_options['EXP_OPTIONS']}, {value}"
        merged.append(option)
    if "EXP_OPTIONS" in user_options and "EXP_OPTIONS" not in default_keys:
        merged.append(f"EXP_OPTIONS+={user_options['EXP_OPTIONS']}")

    merged += [
        f"{key}={value}" for key, value in user_options.items() if key != "EXP_OPTIONS"
    ]
    return merged


def _user_mkexp_options(config, runtime):
    options = dict(config.get("experiment", {}).get("options", {}))
    for entry in runtime.mkexp_options:
        key, separator, value = entry.partition("=")
        if not separator or not key:
            raise ConfigurationError(
                f"--mkexp-option expects KEY=VALUE, got '{entry}'."
            )
        options[key] = value
    if runtime.account is not None:
        options["ACCOUNT"] = runtime.account
    if runtime.environment is not None:
        options["ENVIRONMENT"] = runtime.environment
    return options


def _render_template(template, variables):
    template = Path(template)
    env = Environment(
        loader=FileSystemLoader(str(template.parent)), undefined=StrictUndefined
    )
    return json.loads(env.get_template(template.name).render(variables))


def probtest_config(application_config_file, runtime):
    """
    Translate a chilitest application configuration into a probtest
    configuration.

    Returns the configuration (dict with the sections of a probtest json
    config file) and the kind of application.
    """
    config = load_application_config(application_config_file)
    kind = validate_application_config(config)

    # the command line takes precedence over the section file_info
    file_info = config.get("file_info", {})
    mkexp_config = runtime.mkexp_config or file_info.get("mkexp_config_path")
    build_dir = runtime.build_dir or file_info.get("build_dir")
    stats_dir = runtime.stats_dir or file_info.get("stats_dir")
    missing = [
        option
        for option, value in [
            ("--mkexp-config", mkexp_config),
            ("--build-dir", build_dir),
            ("--stats-dir", stats_dir),
        ]
        if not value
    ]
    if missing:
        raise ConfigurationError(f"Missing option(s): {', '.join(missing)}")

    mkexp_config = os.path.abspath(mkexp_config)
    src_dir = os.path.abspath(runtime.src_dir or _find_src_dir(mkexp_config))

    result = _render_template(
        runtime.template or DEFAULT_TEMPLATE,
        {
            "experiment_name": Path(mkexp_config).stem,
            "mkexp_config": mkexp_config,
            "mkexp_path": runtime.mkexp_path
            or os.path.join(src_dir, "utils", "mkexp", "mkexp"),
            "src_dir": src_dir,
            "build_dir": os.path.abspath(build_dir),
            "stats_dir": os.path.abspath(stats_dir),
            "output_dirname": runtime.output_dirname
            or file_info.get("output_dirname")
            or DEFAULT_OUTPUT_DIRNAME,
        },
    )

    workflow = result["mkexp_workflow"]
    default = result["default"]
    run = result["run"]

    variables = _analysis(
        config,
        "ensemble_statistics" if kind == APPLICATION_REFERENCE else "statistics",
    )
    unknown = [dim for dim in variables if dim not in workflow["file_patterns"]]
    if unknown:
        raise ConfigurationError(
            f"No output files are defined for the dimension(s) {unknown}."
        )
    patterns = {dim: workflow["file_patterns"][dim] for dim in variables}
    file_type = workflow["file_type"]

    default["file_id"] = [[file_type, pattern] for pattern in patterns.values()]
    default["npz_dimensions"] = {
        dim: f"{file_type}:{pattern}" for dim, pattern in patterns.items()
    }
    result["stats"]["variable_filter"] = {
        patterns[dim]: names for dim, names in variables.items()
    }

    experiment = config.get("experiment", {})
    run["mkexp_option"] = _merge_mkexp_options(
        run["mkexp_option"], _user_mkexp_options(config, runtime)
    )
    run["restart"] = bool(experiment.get("restart", False))

    if kind == APPLICATION_REFERENCE:
        _translate_reference(config, result, patterns)
    else:
        _translate_test(config, result)

    return result, kind


def _translate_reference(config, result, patterns):
    ensemble = config["ensemble"]
    if ensemble["perturbation_method"] != PERTURBATION_METHOD:
        raise ConfigurationError(
            f"Unsupported perturbation_method '{ensemble['perturbation_method']}'. "
            f"Supported: {PERTURBATION_METHOD}."
        )
    size = ensemble["size"]
    if not isinstance(size, int) or size < 1:
        raise ConfigurationError(f"The ensemble size must be at least 1, got {size}.")

    statistics = config["ensemble_statistics"]["statistics"]
    if not {"min", "max"} <= set(statistics):
        raise ConfigurationError(
            f"The ensemble statistics must contain 'min' and 'max', got {statistics}."
        )
    unsupported = sorted(set(statistics) - {"min", "max"})
    if unsupported:
        raise ConfigurationError(
            f"Unsupported ensemble statistic(s) {unsupported}. Supported: min, max."
        )

    default = result["default"]
    default["member_ids"] = list(range(1, size + 1))
    result["run"]["rhs_new"] = [
        "{seed}",
        repr(float(ensemble["perturbation_amplitude"])),
    ]
    result["stats"]["ensemble"] = True
    # the time axis stored with the statistics is the one of the reference run
    result["tolerance"]["npz_times"] = {
        dim: os.path.join(default["model_output_dir"], pattern)
        for dim, pattern in patterns.items()
    }


def _translate_test(config, result):
    result["default"]["member_ids"] = []
    # nothing is perturbed
    result["run"]["lhs"] = []
    result["run"]["rhs_new"] = []
    result["stats"]["ensemble"] = False
    result["stats"]["stats_file_name"] = result["check"]["current_files"]
    result["check"]["factor"] = float(config["test"]["extreme_values"]["factor"])
