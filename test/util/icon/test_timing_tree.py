"""
This module contains unit tests to verify the functionality of the TimingTree class,
including reading log files, loading from JSON, intersection and subtraction of trees,
growing trees with new nodes, and adding trees together.
"""

import os
import shutil
import unittest

import numpy as np
import pandas as pd

from probtest.util.icon.extract_timings import read_logfile
from probtest.util.tree import TimingTree

TIMING_FILE_1 = "test/data/timing_example_1.txt"
TIMING_FILE_2 = "test/data/timing_example_2.txt"
TIMING_FILE_3 = "test/data/timing_example_3.txt"

JSON_REFERENCE = "test/data/reference"
JSON_TREE_REFERENCE = "test/data/reference_tree.json"
JSON_META_REFERENCE = "test/data/reference_meta.json"
JSON_DATA_REFERENCE = "test/data/reference_data.json"

JSON_ADD_REFERENCE = "test/data/add"

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)


class TestTimingTree(unittest.TestCase):
    """
    Unit tests for the `TimingTree` class, covering initialization, data
    manipulation, and tree operations.

    This class uses the `unittest` framework to verify the correctness of
    methods and functionality in the `TimingTree` class.
    It includes tests for reading timing data from log files, loading and
    comparing timing trees from JSON files, and performing tree operations like
    intersection, subtractions, growing, and addition.
    """

    @classmethod
    def setUpClass(cls):
        test_path = os.path.realpath("test/tmp")
        cls.test_path = test_path
        # create test directory (remake if it exists)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
        os.mkdir(test_path)

    def setUp(self):
        return

    def tearDown(self):
        return

    def assert_trees_equal(self, t1, t2):
        for i in range(t1.meta_data["n_tables"]):
            diff = t1.data[i].values - t2.data[i].values
            diff_sum = np.sum(np.abs(diff))

            self.assertDictEqual(
                t1.meta_data,
                t2.meta_data,
                msg=f"meta data does not match for table {i}",
            )
            self.assertLess(diff_sum, 1e-12, msg=f"data does not match for table {i}")
            self.assertListEqual(
                t1.root[i].to_ancestry_name_list(),
                t2.root[i].to_ancestry_name_list(),
                msg=f"tree does not match for table {i}",
            )

    def test_read_timing(self):
        for timing_file in (TIMING_FILE_1, TIMING_FILE_2, TIMING_FILE_3):
            tt = TimingTree.from_logfile(timing_file, read_logfile)

            self.assertIsNotNone(tt.data, msg="did not properly initialize data")
            self.assertIsNotNone(
                tt.meta_data, msg="did not properly initialize meta data"
            )
            self.assertIsNotNone(tt.root, msg="did not properly initialize tree")

    def test_json_load(self):
        tt_json = TimingTree.from_json(JSON_REFERENCE)
        tt = TimingTree.from_logfile(TIMING_FILE_1, read_logfile)

        self.assert_trees_equal(tt_json, tt)

    def test_intersection(self):
        tt1 = TimingTree.from_logfile(TIMING_FILE_1, read_logfile)
        tt2 = TimingTree.from_logfile(TIMING_FILE_2, read_logfile)

        intersection_nodes = tt1.root[-1].intersection(tt2.root[-1])

        names = {n.get_ancestry_name() for n in intersection_nodes}

        ref_names = {
            "root>total>integrate_nh>nh_solve>nh_solve.cellcomp",
            "root>upper_atmosphere",
            "root>total>integrate_nh>physics",
            "root>total>integrate_nh>physics>nwp_radiation>nwp_ecrad_store_fluxes",
            "root>exch_data",
            "root>nh_diagnostics",
            "root>total>integrate_nh>nh_solve>nh_solve.veltend",
            "root>total>integrate_nh>physics>nwp_radiation",
            "root>total>integrate_nh>physics>nwp_turbulence>nwp_turbtrans",
            "root>total>integrate_nh>physics>rediag_prog_vars",
            "root>total>integrate_nh>physics>cloud_cover",
            "root>total>integrate_nh>physics>nwp_turbulence>nwp_turbdiff",
            "root>total>write_restart",
            "root>total>integrate_nh>physics>sso",
            "root>total>integrate_nh>physics>nwp_radiation>nwp_radiation_upscale",
            "root>total>integrate_nh>nh_hdiff",
            "root>total>integrate_nh>transport>adv_horiz",
            "root>total>integrate_nh>transport>adv_horiz>adv_hflx",
            "root>total>integrate_nh>transport>adv_vert>adv_vflx",
            "root>total>integrate_nh>nesting>nesting.bdy_interp",
            "root>total>integrate_nh>physics>phys_acc_sync>global_sum",
            "root>total>integrate_nh>nh_solve>nh_solve.exch",
            "root>total>integrate_nh>physics>phys_acc_sync>ordglb_sum",
            "root>exch_data>exch_data.wait",
            "root>total>integrate_nh>nh_solve",
            "root>total>integrate_nh",
            "root>total>integrate_nh>physics>nwp_radiation>nwp_ecrad_utilities",
            "root>total",
            "root>total>integrate_nh>nh_solve>nh_solve.vimpl",
            "root>total>integrate_nh>physics>nwp_turbulence",
            "root>total>write_restart>write_restart_communication",
            "root>total>write_restart>write_restart_io",
            "root>total>integrate_nh>physics>diagnose_pres_temp",
            "root>total>integrate_nh>physics>nwp_microphysics",
            "root>total>integrate_nh>physics>nwp_surface",
            "root>total>integrate_nh>physics>nwp_convection",
            "root>total>integrate_nh>physics>nwp_radiation>preradiaton",
            "root>total>integrate_nh>nh_solve>nh_solve.vnupd",
            "root>total>integrate_nh>physics>nwp_radiation>nwp_radiation_downscale",
            "root>total>integrate_nh>nesting",
            "root>total>integrate_nh>transport>adv_vert",
            "root>total>integrate_nh>transport",
            "root>total>integrate_nh>physics>phys_u_v",
            "root>total>integrate_nh>physics>satad",
            "root>total>integrate_nh>physics>phys_acc_sync",
            "root>total>integrate_nh>physics>radheat",
            "root>total>integrate_nh>physics>nwp_radiation>nwp_ecrad_ecrad",
        }

        self.assertTrue(
            names == ref_names, msg="set of intersecting nodes does not match reference"
        )

    def test_sub(self):
        tt1 = TimingTree.from_logfile(TIMING_FILE_1, read_logfile)
        tt2 = TimingTree.from_logfile(TIMING_FILE_2, read_logfile)

        sub_nodes = tt1.root[-1].sub(tt2.root[-1])

        names = {n.get_ancestry_name() for n in sub_nodes}

        ref_names = {"root>total>integrate_nh>nh_solve>nh_solve.edgecomp"}

        self.assertTrue(
            names == ref_names, msg="set of differing nodes does not match reference"
        )

    def test_grow(self):
        tt1 = TimingTree.from_logfile(TIMING_FILE_1, read_logfile)
        tt2 = TimingTree.from_logfile(TIMING_FILE_2, read_logfile)

        diff_nodes = tt1.root[-1].sub(tt2.root[-1])

        tt2.grow(diff_nodes, -1)

        new_node = tt2.find("nh_solve.edgecomp", -1)

        self.assertEqual(
            new_node.name,
            "nh_solve.edgecomp",
            msg="did not add (non-present) node nh_solve.edgecomp",
        )

    def test_add(self):
        tt1 = TimingTree.from_logfile(TIMING_FILE_1, read_logfile)
        tt2 = TimingTree.from_logfile(TIMING_FILE_2, read_logfile)
        tt_added = TimingTree.from_json(JSON_ADD_REFERENCE)

        tt1.add(tt2)

        self.assert_trees_equal(tt1, tt_added)


if __name__ == "__main__":
    unittest.main()