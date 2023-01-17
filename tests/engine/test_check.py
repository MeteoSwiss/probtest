import unittest

import numpy as np
import pandas as pd

from engine.check import check_variable
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

        self.df1 = pd.DataFrame(np.random.rand(4, 12) + 1, index=index, columns=columns)
        self.df2 = pd.DataFrame(np.random.rand(4, 12) + 2, index=index, columns=columns)
        self.tol1 = pd.DataFrame(
            np.ones((2, 12)) * 3, index=["var_1", "var_2"], columns=columns
        )
        self.tol2 = pd.DataFrame(
            np.ones((2, 12)), index=["var_1", "var_2"], columns=columns
        )

    def tearDown(self):
        return

    def test_check(self):
        # compute relative difference
        diff_df = compute_rel_diff_dataframe(self.df1, self.df2)
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

        return

    def test_check_zeros(self):
        self.df1.loc["var_1", (2, "max")] = 0

        self.test_check()
        return

    def test_check_smalls1(self):
        self.df1.loc["var_1", (2, "max")] = 1e-25
        self.df2.loc["var_1", (2, "max")] = 1e-14

        self.test_check()
        return

    def test_check_smalls2(self):
        self.df1.loc["var_1", (2, "max")] = 1e-14
        self.df2.loc["var_1", (2, "max")] = 1e-25

        self.test_check()
        return


if __name__ == "__main__":
    unittest.main()
