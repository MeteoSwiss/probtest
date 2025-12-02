"""
This module provides utility functionality for loading configuration defaults for
probtest from a JSON configuration file and defining custom Click parameter
types for handling comma-separated values.
It also contains help messages for various command-line options.
"""

import json
import os
import pathlib

import click

from util import dataframe_ops
from util.log_handler import logger


def load_defaults(sections):
    configfile = os.environ.get("PROBTEST_CONFIG", "probtest.json")
    config = {}

    if not os.path.isfile(configfile):
        logger.warning(
            "careful, configfile %s does not exist in %s. No defaults.",
            configfile,
            os.getcwd(),
        )
        logger.warning(
            "If you need e.g. ICON defaults, try: export PROBTEST_CONFIG=%s",
            pathlib.Path(__file__).parent.parent.absolute() / "templates/ICON.jinja",
        )
    else:
        logger.info("reading config file from %s", configfile)
        with open(configfile, encoding="utf-8") as f:
            try:
                config = json.load(f)
            except json.decoder.JSONDecodeError as err:
                logger.error("failed to load configfile %s", configfile)
                logger.error(err)

    tmp = {}
    if config:
        # set default variables
        if "default" in config.keys():
            tmp.update(config.get("default"))
        # give current sections precedence
        for i in range(len(sections)):
            section = "-".join(sections[: i + 1])
            if section in config.keys():
                tmp.update(config.get(section))

    return tmp


class CommaSeparatedInts(click.ParamType):
    """
    A custom Click parameter type for handling comma-separated integers.

    This class defines a Click parameter type that converts a comma-separated
    string input into a list of integers.
    It is used to parse command-line arguments where multiple integer values are
    provided as a single comma-separated string.
    """

    name = "CommaSeparatedInts"

    def convert(self, value, param, ctx):
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            self.fail(f"Input must be a string, found {value}", param, ctx)
        return [int(e) for e in filter(lambda x: x != "", value.split(","))]


class CommaSeparatedStrings(click.ParamType):
    """
    A custom Click parameter type for handling comma-separated strings.

    This class defines a Click parameter type that converts a comma-separated
    string input into a list of strings.
    It is used to parse command-line arguments where multiple string values are
    provided as a single comma-separated string.
    """

    name = "CommaSeparatedStrings"

    def convert(self, value, param, ctx):
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            self.fail(f"Input must be a int, found {value}", param, ctx)
        return list(filter(lambda x: x != "", value.split(",")))


cli_help = {
    "model_input_dir": r"The directory where the model input is stored.",
    "perturbed_model_input_dir": r"Template for the directory where the perturbed "
    + r"model input is stored. Must contain '\{member_id\}'.",
    "model_output_dir": r"The directory where the model output is stored.",
    "perturbed_model_output_dir": r"Template for the directory where the perturbed "
    + r"model output is stored. Must contain '\{member_id\}''.",
    "experiment_name": r"The name of the experiment to be run.",
    "perturbed_experiment_name": r"Template for the name of the perturbed experiments. "
    + r"Must contain '\{member_id\}'.",
    "tolerance_files_output": r"List containing the name of the output file/s "
    + r"containing the tolerances (per time step and variable), "
    + r"both for stats and fof files.",
    "tolerance_files_input": r"List containing the name of the input file/s "
    + r"containing the tolerances (per time step and variable), "
    + r"both for stats and fof files.",
    "ensemble_files": r"List containing the name of the stats file and the fof file"
    + r" representing the ensemble.",
    "stats_file_name": r"The name of the stats file to be created.",
    "member_count": r"Count of ensemble members " + r'(e.g. "10").',
    "member_id": r"ID of ensemble member " + r'(e.g. "3").',
    "member_ids": r"List of member ids" + r'(e.g. "1,3,14")',
    "fof_types": r"Type of fof files you want to analyse" + r"(e.g. AIREP,PILOT).",
    "member_type": r"Precision of experiment (e.g. 'double' or 'mixed'). "
    + r"The type is part of the created member_id, which is equal to "
    + r"(member_type + '_' + str(member_id)).",
    "perturb_amplitude": r"The amplitude of the relative perturbation.",
    "files": r"The files that need to be perturbed.",
    "variable_names": r"The variables that are perturbed.",
    "copy_all_files": r"Copy all files from the model_input_dir directory to the "
    + r"perturbed_model_input_dir.",
    "file_id": r"A unique identifier and file pattern FILE_PATTERN of the files "
    + r"containing the variables to be analysed and the file specification label "
    + r"FILE_TYPE. FILE_PATTERN may contain simple shell-style wildcards such as "
    + r'"*" and will be expanded internally by glob. Put FILE_PATTERN in quotes to '
    + r"avoid early glob expansion by the calling shell.",
    "ensemble": r"Create stats file for an ensemble and the reference file.",
    "enable_check_only": r"Check with how many stats files out "
    + r"of x (x=total_member_count) the probtest passes given a specific"
    + "tolerance file.",
    "total_member_count": r"Number of ensemble members used "
    + r"to select the x members from (min_member_count <= x <= max_member_count).",
    "max_member_count": r"Maximum number of members to select.",
    "min_factor": r"Minimum tolerance factor used to select members with.",
    "max_factor": r"Maximum tolerance factor used to select members with.",
    "selected_members_file_name": r"The name of the file in which the selected "
    + r"members and the used factor will be written.",
    "file_specification": "Specify how different file types shall be read. This "
    + r"option must be defined in the json config file. See doc string of  "
    + r"df_from_file_ids for the specification.",
    "reference_files": r"List of reference file/s to check against, "
    + r"both for stats and fof files.",
    "current_files": r"List of current files to be tested, "
    + r"both for stats and fof files",
    "factor": r"Relaxation factor for the tolerance values.",
    "timing_regex": r"Regex for the file that contains the latest log.",
    "timing_names": r"The name of the timing entries to be displayed.",
    "timing_database": r"A persistent file to keep track of performance history.",
    "append_time": r"If true: append to the performance data; If false: overwrite the "
    + r"performance data (default: false).",
    "run_dir": r"Directory from where the run is started "
    + r"(with 'submit_command model_run_script_name').",
    "perturbed_run_dir": r"Directory from where the perturbed run is started "
    + r"(with 'submit_command model_run_script_name').",
    "run_script_name": r"Name of the original experiment runscript.",
    "perturbed_run_script_name": r"Template for the perturbed experiment name. Must "
    + r"contain '\{member_id\}'.",
    "lhs": r"Replace assignments in the runscript. For multiples, use comma separated "
    + r"list. Note that the new right hand side can depend on \{member_id\} define "
    + r"left hand side.",
    "rhs_new": r"Define new right hand side.",
    "rhs_old": r"Define old right hand side (optional, put None if not needed).",
    "submit_command": r"How a model simulation is submitted.",
    "parallel": r"Run jobs in parallel.",
    "dry": r"Only generate runscripts, do not run the model.",
    "savedir": r"The directory where the plots are stored.",
    "cdo_table_file": r"File to store the cdo table into.",
    "variables": r"Select variables to print CDO diff from ensemble.",
    "times": r"Select times to print CDO diff from ensemble.",
    "histogram": r"Print out full histogram of relative differences.",
    "codebase_install": r"The directory where the code base is installed.",
    "reference": r"The directory where reference files are read from and written to.",
    "config": r"The name of the config file that is being generated.",
    "template_name": r"Path to the template for the config file.",
    "i_table": r"Which table to plot, must be an int between 0 and n_tables-1.",
    "timing_current": r"The database containing the timings for the current run.",
    "timing_reference": r"The database containing the reference timings.",
    "measurement_uncertainty": r"How much time in [s] the current time can deviate "
    + "from the reference.",
    "tolerance_factor": r"The factor by which the current time can deviate from the "
    + "reference.",
    "new_reference_threshold": r"The factor by which the current time can be faster "
    + r"than the reference before a warning gets printed.",
    "minimum_tolerance": r"Non-zero value to set variable tolerances to when the "
    + r"calculated tolerances from the ensemble are exactly zero.",
}

del dataframe_ops
