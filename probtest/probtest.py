#!/usr/bin/env python
"""
Command-line interface (CLI) for the probabilistic testing and analysis tool.

This script uses the Click library to define a command-line interface for
various subcommands related to probabilistic testing, performance analysis, and
visualization.
"""
import click
import matplotlib

from probtest.engine.cdo_table import cdo_table
from probtest.engine.check import check
from probtest.engine.init import init
from probtest.engine.performance import performance
from probtest.engine.performance_check import performance_check
from probtest.engine.perturb import perturb
from probtest.engine.run_ensemble import run_ensemble
from probtest.engine.select_members import select_members
from probtest.engine.stats import stats
from probtest.engine.tolerance import tolerance
from probtest.util.click_util import load_defaults
from probtest.util.log_handler import initialize_logger
from probtest.visualize.cdo_table_reader import cdo_table_reader
from probtest.visualize.check_plot import check_plot
from probtest.visualize.performance_meta_data import performance_meta_data
from probtest.visualize.performance_plot import performance_plot

matplotlib.use("Agg")

# pylint: disable=no-value-for-parameter


@click.group()
@click.option("--log-level", type=str, default="INFO")
@click.option("--log-file", type=str, default="probtest.log")
@click.pass_context
def cli(ctx, log_level, log_file):
    initialize_logger(log_level=log_level.upper(), log_file=log_file)
    subcommand = ctx.invoked_subcommand
    # the section is the first part of the subcommand (e.g. cdo-table -> cdo)
    sections = subcommand.split("-")
    if "init" not in sections:
        defaults = load_defaults(sections)
        ctx.default_map = {subcommand: defaults}


cli.add_command(init)
cli.add_command(perturb)
cli.add_command(stats)
cli.add_command(check)
cli.add_command(tolerance)
cli.add_command(select_members)
cli.add_command(run_ensemble)
cli.add_command(performance)
cli.add_command(check_plot)
cli.add_command(performance_plot)
cli.add_command(performance_meta_data)
cli.add_command(performance_check)
cli.add_command(cdo_table)
cli.add_command(cdo_table_reader)

if __name__ == "__main__":
    cli()
