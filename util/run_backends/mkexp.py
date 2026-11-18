"""
mkexp backend: generate the run script of an experiment with mkexp, the
experiment setup tool of ICON.

mkexp is called as

    mkexp CONFIG EXP_ID=<experiment name> [KEY=VALUE ...]

where each KEY=VALUE overrides a setting of the experiment configuration, e.g.
``namelists.NAMELIST_atm.initicon_nml.pinit_seed=42``. ``KEY+=VALUE`` appends
to a setting instead of replacing it.
"""

import os
import subprocess
from typing import List, Optional

from util.log_handler import logger

_TAIL_BYTES = 1024 * 1024


def perturbation_arguments(
    lhs: Optional[List[str]], rhs_new: Optional[List[str]], seed: int
) -> List[str]:
    """
    mkexp arguments that perturb a member: one KEY=VALUE per entry of lhs and
    rhs_new, with {seed} replaced in the values.
    """
    lhs = lhs or []
    rhs_new = rhs_new or []
    if len(lhs) != len(rhs_new):
        raise ValueError(
            f"lhs ({lhs}) and rhs_new ({rhs_new}) must have the same length."
        )
    return [
        f"{left}={str(right).format(seed=seed)}" for left, right in zip(lhs, rhs_new)
    ]


def mkexp_command(
    mkexp_path: str, mkexp_config: str, experiment_name: str, options: List[str]
) -> List[str]:
    for option in options:
        if "=" not in option:
            raise ValueError(f"mkexp option '{option}' is not of the form KEY=VALUE.")
        if option.split("=", 1)[0].rstrip("+") == "EXP_ID":
            raise ValueError(
                "EXP_ID must not be set by a mkexp option, "
                "it is set to the name of the experiment."
            )
    return [mkexp_path, mkexp_config, f"EXP_ID={experiment_name}", *options]


def generate_mkexp_runscript(  # pylint: disable=too-many-positional-arguments
    mkexp_path: str,
    mkexp_config: str,
    experiment_name: str,
    options: List[str],
    mkexp_search_path: str,
    runscript: str,
    restart: bool = False,
) -> None:
    """
    Run mkexp for one experiment and make sure it created runscript.

    mkexp_search_path: directory mkexp searches for experiment types and
        options (passed as the environment variable MKEXP_PATH)
    restart: keep the state of a previous run of the experiment. By default
        the date file next to the run script is removed, so that the
        experiment starts from the beginning.
    """
    command = mkexp_command(mkexp_path, mkexp_config, experiment_name, options)
    logger.info("generating run script with '%s'", " ".join(command))
    subprocess.run(
        command, check=True, env={**os.environ, "MKEXP_PATH": mkexp_search_path}
    )

    if not os.path.isfile(runscript):
        raise RuntimeError(
            f"mkexp did not create the run script '{runscript}' "
            f"for the experiment '{experiment_name}'."
        )

    if not restart:
        date_file = os.path.join(os.path.dirname(runscript), f"{experiment_name}.date")
        if os.path.exists(date_file):
            logger.info("removing %s to start from the beginning", date_file)
            os.remove(date_file)


def run_log_file(runscript: str, run_log_name: str, experiment_name: str) -> str:
    """
    Log file of a run: run_log_name next to the run script.
    """
    return os.path.join(
        os.path.dirname(runscript),
        run_log_name.format(experiment_name=experiment_name),
    )


def check_success_marker(log_file: str, marker: str) -> bool:
    """
    Whether marker appears in the last part of log_file.
    """
    try:
        with open(log_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - _TAIL_BYTES))
            return marker.encode() in f.read()
    except OSError:
        return False
