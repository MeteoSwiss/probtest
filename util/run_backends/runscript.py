"""
Run-script backend: create the run script of a perturbed experiment by copying
an existing run script and replacing assignments and the experiment name.
"""

import re
from typing import List, Optional

from util.log_handler import logger


# replace an assignment statement (left=*right* to left=new)
# "right" is used here to identify multiple assignments of "left"
def replace_assignment(line, left, right_new, right_old, seed):
    if line.strip().startswith("#") or (left not in line):
        return line

    # make sure this is a regular assign statement (X = Y)
    split = line.split("=")
    if len(split) != 2:
        return line

    lhs, rhs = split
    lhs = lhs.strip()
    rhs = rhs.strip()

    if right_old and (right_old not in rhs):
        return line

    if lhs != left:
        return line

    right_new = right_new.format(seed=seed)

    out_line = f"{left}={right_new}\n"
    logger.info("generating new line: %s", out_line.replace("\n", ""))
    return out_line


# replace all strings matching "old" substring by "new" substring
def replace_string(line, old, new):
    out_line = re.sub(old, new, line) if old in line else line
    return out_line


def prepare_perturbed_run_script(  # pylint: disable=too-many-positional-arguments
    runscript: str,
    perturbed_runscript: str,
    experiment_name: str,
    modified_experiment_name: str,
    lhs: List[str],
    rhs_new: List[str],
    rhs_old: Optional[List],
    seed: int,
) -> None:
    """
    Prepare a `perturbed_runscript` by copying `runscript` and replacing
    namelist values and experiment names.
    Raises RuntimeError if no lines are modified.
    """

    modified = False  # Track if any line was changed

    with open(runscript, "r", encoding="utf-8") as in_file:
        with open(perturbed_runscript, "w", encoding="utf-8") as out_file:

            if rhs_old is None:
                rhs_old = [None] * len(rhs_new)
            if 1 == len(rhs_old) < len(rhs_new) and rhs_old[0] == "None":
                rhs_old = [None] * len(rhs_new)
            rhs_old = [None if r == "None" else r for r in rhs_old]

            # only modify namelist if lhs,rhs_old or rhs_new are not equal None
            if any(
                any(item is not None for item in entry)
                for entry in [lhs, rhs_old, rhs_new]
            ):
                for line in in_file:
                    out_line = line
                    # replace input directory with the ones given in config file
                    for lh, rh_old, rh_new in zip(lhs, rhs_old, rhs_new):
                        out_line = replace_assignment(line, lh, rh_new, rh_old, seed)
                        # replace first match
                        if out_line != line:
                            modified = True
                            break

                    # rename the experiment name
                    if line == out_line:
                        out_line = replace_string(
                            line, experiment_name, modified_experiment_name
                        )

                    out_file.write(out_line)
            else:
                out_file.write(in_file.read())

            if not modified:
                raise RuntimeError(
                    f"No lines were modified in the runscript '{perturbed_runscript}'. "
                    f"Check lhs({lhs}) and rhs_new({rhs_new}) values "
                    f"or experiment names({experiment_name})."
                )

            logger.info("writing model run script to: %s", perturbed_runscript)
