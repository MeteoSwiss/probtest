"""
This module contains unit tests for the `util/log_handler.py` module.
"""

import pandas as pd

from util.log_handler import log_dataframe


class DummyLogger:  # pylint: disable=too-few-public-methods
    """Simple logger that stores messages in memory for testing."""

    def __init__(self):
        self.messages = []

    def info(self, msg, *args):
        # mimic logging formatting behavior
        if args:
            msg = msg % args
        self.messages.append(msg)


def test_log_dataframe_none():
    logger = DummyLogger()

    log_dataframe(logger, "TITLE", None)

    assert not logger.messages


def test_log_dataframe_empty():
    logger = DummyLogger()
    df = pd.DataFrame()

    log_dataframe(logger, "TITLE", df)

    assert not logger.messages


def test_log_dataframe_non_empty():
    logger = DummyLogger()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    log_dataframe(logger, "MY TITLE", df)

    assert logger.messages[0] == "MY TITLE"

    # second message is the formatted dataframe
    assert "a" in logger.messages[1]
    assert "1" in logger.messages[1]
    assert "2" in logger.messages[1]


def test_log_dataframe_verbose():
    logger = DummyLogger()
    df = pd.DataFrame(
            {
                "a": [1, 1],
                "b": [2, 2],
                "c": [3, 3],
                "d": [4, 4],
                "e": [5, 5],
                "f": [6, 6],
                "g": [7, 7],
                "h": [8, 8],
                "i": [9, 9],
                "l": [10, 10],
                "m": [11, 11],
                "n": [12, 12],
            }
    )

    log_dataframe(logger, "TITLE", df, verbose=False)

    assert logger.messages[0] == "TITLE"

    # second message is the formatted dataframe
    assert "a" in logger.messages[1]
    assert "f" in logger.messages[1]
    assert "6" in logger.messages[1]
    assert "n" in logger.messages[1]


def test_log_dataframe_calls_to_string_format():
    logger = DummyLogger()

    class SpyDF(pd.DataFrame):
        """DataFrame subclass that overrides to_string for testing."""

        def to_string(self, *_args, **_kwargs):
            return "SPY_OUTPUT"

    df = SpyDF({"a": [1]})

    log_dataframe(logger, "TITLE", df)

    assert logger.messages[1] == "SPY_OUTPUT"
