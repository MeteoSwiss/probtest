import logging
import pandas as pd
import pytest

from util.log_handler import log_dataframe


class DummyLogger:
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

    assert logger.messages == []


def test_log_dataframe_empty():
    logger = DummyLogger()
    df = pd.DataFrame()

    log_dataframe(logger, "TITLE", df)

    assert logger.messages == []


def test_log_dataframe_non_empty():
    logger = DummyLogger()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    log_dataframe(logger, "MY TITLE", df)

    assert logger.messages[0] == "MY TITLE"

    # second message is the formatted dataframe
    assert "a" in logger.messages[1]
    assert "1" in logger.messages[1]
    assert "2" in logger.messages[1]


def test_log_dataframe_calls_to_string_format():
    logger = DummyLogger()

    class SpyDF(pd.DataFrame):
        def to_string(self, *args, **kwargs):
            return "SPY_OUTPUT"

    df = SpyDF({"a": [1]})

    log_dataframe(logger, "TITLE", df)

    assert logger.messages[1] == "SPY_OUTPUT"