#!/usr/bin/env python
import click
import matplotlib

from engine.cdo_table import cdo_table
from engine.check import check
from engine.init import init
from engine.performance import performance
from engine.performance_check import performance_check
from engine.perturb import perturb
from engine.run_ensemble import run_ensemble
from engine.stats import stats
from engine.tolerance import tolerance
from engine.optimal_member_sel import optimal_member_sel
from engine.test_selection import test_selection
from util.click_util import load_defaults
from util.log_handler import initialize_logger
from visualize.cdo_table_reader import cdo_table_reader
from visualize.check_plot import check_plot
from visualize.performance_meta_data import performance_meta_data
from visualize.performance_plot import performance_plot

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
    pass


cli.add_command(init)
cli.add_command(perturb)
cli.add_command(stats)
cli.add_command(check)
cli.add_command(tolerance)
cli.add_command(optimal_member_sel)
cli.add_command(test_selection)
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
