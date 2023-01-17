#!/bin/bash
set -x

# load python env
source /project/g110/icon/probtest/conda/miniconda/bin/activate
conda activate probtest

# install formatting packages
pre-commit install

# create temporary branch and track source branch
# this is important for publishing in the next stage
git checkout -b formatting
git branch -u origin/${gitlabSourceBranch}

# run pre-commit hooks on all files
pre-commit run --all-files &> formatted.txt

# the above command will fail if files are formatted
exit 0
