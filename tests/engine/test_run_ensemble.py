"""
This module contains unit tests for the `engine.run_ensemble` module functions,
specifically `append_job` and `finalize_jobs`.
It uses the `unittest` framework to ensure the correctness of these functions
under different scenarios.
"""

import subprocess
import unittest

from engine.run_ensemble import append_job, finalize_jobs


class TestRunEnsemble(unittest.TestCase):
    """
    Test suite for validating job management functions in different execution
    modes.

    This class contains unit tests for functions responsible for appending and
    finalizing jobs, both in serial and parallel execution modes.
    """

    def testJobSuccessfulSerial(self):
        job = "test 1 = 1".split()
        append_job(job=job, job_list=[], parallel=False)

    def testJobSuccessfulParallel(self):
        job = "test 1 = 1".split()
        job_list = []
        append_job(job=job, job_list=job_list, parallel=True)
        finalize_jobs(job_list=job_list, dry=False, parallel=True)

    def testJobFailedSerial(self):
        job = "test 1 = 2".split()
        with self.assertRaises(subprocess.CalledProcessError):
            append_job(job=job, job_list=[], parallel=False)

    def testJobFailedParallel(self):
        job = "test 1 = 2".split()
        job_list = []
        append_job(job=job, job_list=job_list, parallel=True)
        with self.assertRaises(subprocess.CalledProcessError):
            finalize_jobs(job_list=job_list, dry=False, parallel=True)


if __name__ == "__main__":
    unittest.main()
