"""
Tests for the fof rules of check_multiple_solutions_from_dict
"""

from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
import pytest

from util.dataframe_ops import check_multiple_solutions_from_dict


@pytest.fixture(name="plevel_dict", scope="function")
def sample_plevel_dict():
    """Reports and observations with a float column compared by a tolerance rule."""
    df_reports = pd.DataFrame({"lat": [40.0, 41.0], "r_check": [32, 32]})
    df_observation = pd.DataFrame(
        {"varno": [3, 4], "plevel": [77244.56, np.nan], "check": [32, 32]}
    )
    return {"reports": df_reports, "observation": df_observation}


def _rules_errors(dict_ref, dict_cur, rules):
    with patch("builtins.open", mock_open()):
        return check_multiple_solutions_from_dict(
            dict_ref, dict_cur, rules, log_file_name="file_name.log"
        )


def test_multiple_solutions_tolerance_rule_within(plevel_dict):
    dict_cur = {key: df.copy() for key, df in plevel_dict.items()}
    dict_cur["observation"].loc[0, "plevel"] = 77244.555
    assert _rules_errors(plevel_dict, dict_cur, {"plevel": 1e-6}) is False


def test_multiple_solutions_tolerance_rule_exceeded(plevel_dict):
    dict_cur = {key: df.copy() for key, df in plevel_dict.items()}
    dict_cur["observation"].loc[0, "plevel"] = 77252.56
    assert _rules_errors(plevel_dict, dict_cur, {"plevel": 1e-6}) is True


def test_multiple_solutions_tolerance_rule_nan(plevel_dict):
    dict_cur = {key: df.copy() for key, df in plevel_dict.items()}
    assert _rules_errors(plevel_dict, dict_cur, {"plevel": 1e-6}) is False
    dict_cur["observation"].loc[1, "plevel"] = 50000.0
    assert _rules_errors(plevel_dict, dict_cur, {"plevel": 1e-6}) is True


def test_multiple_solutions_error_in_reports_kept(plevel_dict):
    dict_cur = {key: df.copy() for key, df in plevel_dict.items()}
    dict_cur["reports"].loc[0, "r_check"] = 13
    rules = {"r_check": [32, 5], "check": [32, 5]}
    assert _rules_errors(plevel_dict, dict_cur, rules) is True


def test_multiple_solutions_tolerance_rule_boundary(plevel_dict):
    dict_cur = {key: df.copy() for key, df in plevel_dict.items()}
    dict_cur["observation"].loc[0, "plevel"] = 77244.56 + 0.5e-4 * 77245.56
    assert _rules_errors(plevel_dict, dict_cur, {"plevel": 1e-4}) is False
    assert _rules_errors(plevel_dict, dict_cur, {"plevel": 0.4e-4}) is True


def test_multiple_solutions_list_rule_nan_in_both():
    df_obs = pd.DataFrame({"check": [32.0, np.nan]})
    dict_ref = {"reports": pd.DataFrame({"lat": [1.0]}), "observation": df_obs}
    dict_cur = {key: df.copy() for key, df in dict_ref.items()}
    assert _rules_errors(dict_ref, dict_cur, {"check": [32, 5]}) is False
