#!/bin/bash
set -e -x

# load python env
source /project/g110/icon/probtest/conda/miniconda/bin/activate
conda activate probtest

# set environment variables to point to reference data
hash=$(cat ${0%/*/*/*}/test_reference)
export PROBTEST_REF_DATA=/project/g110/probtest_testdata/$hash/probtest_data
export PROBTEST_CUR_DATA=./probtest_data

# execute MCH unittests
export PROBTEST_TEST_EXPERIMENT=mch_opr_r04b07
python3 -m unittest
