import unittest

import numpy as np
import pandas as pd

from engine.check import check_variable
from util.constants import CHECK_THRESHOLD
from util.dataframe_ops import compute_rel_diff_dataframe


class TestCheck(unittest.TestCase):
    def setUp(self):
        index = pd.MultiIndex.from_arrays(
            [["var_1"] * 3 + ["var_2"], list(range(3)) + [0]],
            names=["variable", "height"],
        )
        columns = pd.MultiIndex.from_product(
            [range(4), ["max", "min", "mean"]], names=["time", "statistic"]
        )

        self.df1 = pd.DataFrame(
            np.linspace(0.9, 1.9, 4 * 12).reshape(4, 12), index=index, columns=columns
        )
        self.df2 = pd.DataFrame(
            np.linspace(1.1, 2.1, 4 * 12).T.reshape(4, 12), index=index, columns=columns
        )
        self.df1 *= -1  # some negative test data
        self.df2 *= -1
        # Relative differences (df1-df2)/((df1+df2)/2) are between 0.2 and 0.1.

        self.tol1 = pd.DataFrame(
            np.ones((2, 12)) * 0.21, index=["var_1", "var_2"], columns=columns
        )
        self.tol2 = pd.DataFrame(
            np.ones((2, 12)) * 0.15, index=["var_1", "var_2"], columns=columns
        )

    def check(self, df1, df2):
        # compute relative difference
        diff_df = compute_rel_diff_dataframe(df1, df2)
        # take maximum over height
        diff_df = diff_df.groupby(["variable"]).max()

        out1, err1, _ = check_variable(diff_df, self.tol1)
        out2, err2, _ = check_variable(diff_df, self.tol2)
        self.assertTrue(
            out1,
            "Check with large tolerances did not validate! "
            + "Here is the DataFrame:\n{}".format(err1),
        )
        self.assertFalse(
            out2,
            "Check with small tolerances did validate! "
            + "Here is the DataFrame:\n{}".format(err2),
        )

    def test_check(self):
        self.check(self.df1, self.df2)

    def test_check_one_zero(self):
        """Probtest should not pass if any of the values is 0
        and the other is much larger"""
        df1 = self.df1.copy()
        df1.loc["var_1", (2, "max")] = 0
        df2 = self.df2.copy()
        diff_df = compute_rel_diff_dataframe(df1, df2)
        diff_df = diff_df.groupby(["variable"]).max()

        out, err, _ = check_variable(diff_df, self.tol1)

        self.assertFalse(
            out,
            "Check with 0-value reference validated! "
            + "Here is the DataFrame:\n{}".format(err),
        )

        df2.loc["var_1", (2, "max")] = CHECK_THRESHOLD / 2
        # now, both data are comparable again
        self.check(df1, df2)

    def test_check_smalls(self):
        """both values are close to 0 and should be accepted even though
        their relative difference is large.

        Close to 0 means < util.constants.CHECK_THRESHOLD"""
        df1 = self.df1.copy()
        df1.loc["var_1", (2, "min")] = CHECK_THRESHOLD * 1e-5
        df2 = self.df2.copy()
        df2.loc["var_1", (2, "min")] = CHECK_THRESHOLD / -2
        self.check(df1, df2)


class TestCheckSwapped(TestCheck):
    """Test that all Checks are symmetrical"""

    def check(self, df1, df2):
        super().check(df2, df1)


if __name__ == "__main__":
    unittest.main()
