"""
This module provides logging utilities for setting up and configuring loggers.

This module provides functionality for:
- Initializing a logger with specified log levels and output destinations.
- Configuring logging to both console and file with a simple format.
"""

import logging
import sys

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


def get_detailed_logger(log_file_name, logger_name="DETAILS", log_level="DEBUG"):

    return initialize_detailed_logger(
        logger_name, log_level=log_level, log_file=log_file_name
    )
