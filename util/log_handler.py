"""
This module provides logging utilities for setting up and configuring loggers.

This module provides functionality for:
- Initializing a logger with specified log levels and output destinations.
- Configuring logging to both console and file with a simple format.
"""

import logging
import sys
from typing import Optional

import pandas as pd

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

logger = logging.getLogger()


def initialize_logger(log_level="DEBUG", log_file="probtest.log"):
    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(log_level)

    logger.info("initialized logger with level %s", log_level)


def initialize_detailed_logger(
    name,
    log_level="DEBUG",
    log_file=None,
):
    detailed_logger = logging.getLogger(name)
    detailed_logger.setLevel(log_level)
    detailed_logger.propagate = False

    existing_handlers = [
        h
        for h in detailed_logger.handlers
        if getattr(h, "baseFilename", None) == log_file
    ]

    if existing_handlers:
        return detailed_logger

    formatter = logging.Formatter("%(message)s")

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        detailed_logger.addHandler(file_handler)

    detailed_logger.info("initialized named logger '%s'", name)
    return detailed_logger


def log_dataframe(
    log: logging.Logger,
    title: str,
    df: Optional[pd.DataFrame],
    verbose: bool = False,
) -> None:
    """
    Log a DataFrame only if it contains data.
    """
    if df is None or df.empty:
        return

    log.info(title)
    log.info("%s", df.to_string() if verbose else df)


def log_extreme_values_report(log: logging.Logger, result, verbose: bool = False):
    """
    Log the outcome of an extreme-values check: one line per variable and,
    for the failing ones, the values outside the allowed range.
    """
    for (file_id, variable), row in result.summary.iterrows():
        if row["failed"] == 0:
            log.info("passed: %s %s", file_id, variable)
        else:
            log.info(
                "FAILED: %s %s (%s of %s values outside the ensemble range)",
                file_id,
                variable,
                row["failed"],
                row["total"],
            )

    if result.failures.empty:
        return

    log.info("\nValues outside the ensemble range (lower <= difference <= upper)")
    with pd.option_context("display.float_format", lambda x: f"{x:.6e}"):
        log.info("%s", result.failures.to_string() if verbose else result.failures)
