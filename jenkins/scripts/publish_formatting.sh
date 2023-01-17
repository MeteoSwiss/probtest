#!/bin/bash
set -x

# load python env
source /project/g110/icon/probtest/conda/miniconda/bin/activate
conda activate probtest

# merge source branch
git merge origin/${gitlabSourceBranch}

# commit modified files (this will implicitly run pre-commit again)
git add -u
git commit -m 'applied formatting rules'

# publish files
git push origin formatting:${gitlabSourceBranch}
