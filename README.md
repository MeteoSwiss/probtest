# Probtest

Probtest is a suite of python scripts to test various aspects of weather and climate models. Even though it is mainly developed for the ICON model, it does not contain ICON-specific definitions in the source code but allows to model specifics in a configuration script.

## Table of contents
1. [Commands](#commands)
2. [Quick start guide](#quick-start-guide)
3. [Developing in probtest](#developing-in-probtest)

## Commands

The suite consists out of the following scripts:

### probtest

This script is the entry point to the suite that manages all the other scripts and sets up command line arguments with `click`.

### init

To work with probtest, a configuration file is needed. This file contains all the options that are used for the various commands of probtest and can be generated with this command.

### perturb

Perturbs netcdf files that can be used as input by the model.

### run-ensemble

Reads and modifies model namelists to run a perturbed model ensemble.

### stats

Generates a `csv` file containing the min, max and mean values for each of the selected fields on each model level and for each time step.

### tolerance

Computes the maximum spread in each of the selected variables for each time step within a perturbed model ensemble.

### check

Compares two files generated with `stats` under the tolerance ranges generated by `tolerance`.

### select-members

Uses all given stats files generated for a model ensemble with `stats`. From those stats files, randomly selects a specified number of members to generate the tolerances with `tolerance`. Repeats this process by iteratively increasing the number of selected members and the tolerance factor until finding a selection for which all other members pass the tolerance `check`.

### check-plot

Creates a plot visualizing the results of the comparison conducted with `check`.

### cdo-table

Often people like to compare model output using `cdo diff`. This tool compares each point of each variable and time step and reports the largest relative error for each. This is different from `stats` which computes differences on statistics (min, max, mean) of each model level and time step. However, the perturbed ensemble can be used to estimate expected errors with `cdo diff` by applying the same algorithm and storing the output in a database.

### cdo-table-reader

The output of `cdo-table` is written in a [Pandas](https://pandas.pydata.org/) dataframe. To ease reading, cdo-table-reader allows to filter the database by variable name, output file and time step.

### performance

_Note: This script is ICON-specific because it parses the ICON logfile with hard-coded regex._
Read the timing output from the model. These timings are stored in a tree format in the case of nested timings. It can either create a new database or append to an existing one. It produces three databases:

- tree: the relationship between the individual timers
- meta: some metadata storing information about the model run
- data: the actual timer data

### performance-check

Compares two performance databases generated with `performance` and checks if the current one is _too slow_ compared to the reference.

### performance-meta-data

Reads some data from a series of performance databases generated with `performance` and produces a graph showing the performance over time.

### performance-plot

Visualize the performance database generated with `performance`.

## Quick start guide

Even though probtest is used exclusively with ICON at the moment, it does not contain any information about the model or its directory structure. This makes it very flexible and applicable to any circumstance (e.g. usable by Buildbot, Jenkins and human users alike). However, it also requires a lot of information about the model and the data to be processed upon invocation. Since a typical probtest usage involves multiple commands (e.g. run-ensemble -> stats -> tolerance -> check) this leads to a lot of redundancy in the invocation. Therefore, probtest can read commonly used input variables (e.g. the model output directory, the experiment name, the name of the submit script, ...) from a configuration file in json format. To further ease the process, these configuration files can be created from templates using the `init` command. A template for ICON is contained in this repository in the `templates` subdirectory.

### Setup conda

All requirements for using probtest can be easily installed with conda using the
setup scripts.

For setting up conda you use
```console
./setup_miniconda.sh -u
```
which will modify your `.bashrc`, or you can use
```console
./setup_miniconda.sh
source miniconda/bin/activate
```
which requires the source minconda.

The pinned requiremnts can be installed by
```console
./setup_env.sh
```
The unpinned requirements and updating the environment can be done by
```console
./setup_env.sh -u -e
```


### The init command

This command sets up the configuration file. For more help on the command line arguments for `init`, see

```console
python probtest.py init --help
```

The `--template-name` argument can be used to specify the template from which the configuration file is created. One of the templates provided by probtest is `templates/ICON.jinja` which is used as the default in case no other template name is provided. The init command replaces all placeholder values in the template with the values given as command line arguments. All other probtest commands can then read from the configuration file. The name of the configuration file to use is read from the `PROBTEST_CONFIG` environment variable. If this is not set explicitly, probtest will look for a file called `probtest.json` in the current directory.

Setting up the configuration file with `init` may not be fitted perfectly to where you want your probtest files to be. In that case, you can manually edit the file after creation. Alternatively, you can add arguments for your probtest commands on the command line which take precedence over the configuration file defaults. For more help on the options on a specific command, see

```console
python probtest.py {command} --help
```

### Example: Check the output of an ICON experiment with an test build compared to a reference build

Objective: Run an `exp_name` ICON experiment with an test build and check if the
output of the test is within a perturbed ensemble of the reference build.
This is in particular used to validate a GPU build against a CPU reference.

You need to have setup a proper environment, for example as described in the
section [Setup conda](#setup-conda).

#### Initialize probtest
Once set up, probtest can generate the config file according to your needs.
Initialized a `probtest.json` file in your reference build directory, `exp_name`
here should refer to your experiment script:

```console
cd icon-base-dir/reference-build
python ../externals/probtest/probtest.py init --codebase-install $PWD --experiment-name exp_name --reference $PWD --file-id NetCDF "*atm_3d_ml*.nc" --file-id NetCDF "*atm_3d_il*.nc" --file-id NetCDF "*atm_3d_hl*.nc" --file-id NetCDF "*atm_3d_pl*.nc" --file-id latlon "*atm_2d_ll*.nc" --file-id meteogram "Meteogram*.nc"
```
You might need to update the used account in the json file.
The perturbation amplitude may also need to be changed in the json file
(buildbot uses 1e-07 for mixed precision and 1e-14 for double precision).
Note that to change this you should modify the second entry of `rhs_new` in
probtest.json, which should be set to 1e-14 by default.

Note that, it is important that the `file-id` are uniquely describing the data
with the same structure.
Otherwise you might get an error like
```console
packages/pandas/core/indexes/base.py", line 4171, in _validate_can_reindex
    raise ValueError("cannot reindex on an axis with duplicate labels")
ValueError: cannot reindex on an axis with duplicate labels
```
For examples of proper `file-id`s have a look in the ICON repo at
`run/tolerance/set_probtest_file_id`.

Now you should have created a `probtest.json` file in the reference build directory.
This file contains all information needed by probtest to process the ICON experiment.

#### Generate references and tolerances for the reference build
With everything set up properly, the chain of commands can be invoked to run the
reference binary (`run-ensemble`), generate the statistics files used for
probtest comparisons (`stats`) and generate tolerances from these files
(`tolerance`).
To run the perturbed experiments and wait for the submitted jobs to finish:
```console
python ../externals/probtest/probtest.py run-ensemble
```
FYI: if the experiment does not generate all of the files listed in the
`file-id`s above, you you receive a message that certain `file-id` patterns do
not match any file.
Those files can remove them from `file-id`s.

Extract the statistics of your perturbed runs:
```console
python ../externals/probtest/probtest.py stats --ensemble
```
Note that the `--ensemble` option which is set to take precedence over the
default `False` from the configuration and make probtest process the model
output from each ensemble generated by `run-ensemble`.

Finally create the tolerance.csv file for the `exp_name` by analysing those
statistics:
```console
python ../externals/probtest/probtest.py tolerance
```

These commands will generate a number of files:

- `stats_ref.csv`: contains the post-processed output from the unperturbed reference run
- `stats_{member_num}.csv`: contain the post-processed output from the perturbed reference runs (only needed temporarily to generate the tolerance file)
- `exp_name_tolerance.csv`: contains tolerance ranges computed from the stats-files

These can then be used to compare against the output of a test binary (usually a
GPU binary).
For that, manually run the `exp_name.run` experiment with the test binary to
produce the test output.

#### Run and check with test build

To then check if your data from the test binary are validating against reference
build, first run the experiments with the test build.
Run your test simulation without probtest:
```console
cd icon-base-dir/test-build
sbatch run/exp_name.run
```

Then create the test statistics with:
```console
python ../externals/probtest/probtest.py stats --no-ensemble --model-output-dir icon-base-dir/test-build/experiments/exp_name
```
Note how `--model-output-dir` is set to take precedence over the default which
points to the reference binary output to now point to the test binary output.
This command will generate the following file:

- `stats_exp_name.csv`: contains the post-processed output of the test binary model output.

Now all files needed to perform a probtest check are available; the reference
file `stats_ref.csv`, the test file `stats_exp_name.csv` as well as the tolerance
range `exp_name_tolerance.csv`.
Providing these files to `check` will perform the check:

```console
python ../externals/probtest/probtest.py check --input-file-ref stats_ref.csv --input-file-cur stats_exp_name.csv --factor 5
```

This check can be also visualized by:
```console
python ../externals/probtest/probtest.py check-plot --input-file-ref stats_ref.csv --input-file-cur stats_exp_name.csv --tolerance-file-name exp_name_tolerance.csv --factor 5 --savedir ./plot_dir
```

Note that the reference `--input-file-ref` and test stats files
`--input-file-cur` need to be set by command line arguments.
This is because the default stored in the `ICON.jinja` template is pointing to
two files from the ensemble as a sanity check.

## Developing probtest
#### Testing with [pytest](https://docs.pytest.org/en/8.2.x/)

Our tests are executed using `pytest`, ensuring a consistent and efficient testing process. Each test dynamically generates its necessary test data, allowing for flexible and isolated testing scenarios.

Simply run
```console
pytest -s -v tests/*
```
in order to run all tests.

To run only a subset of test run
```console
pytest -s -v path/to/your/test.py
```

Reference data, crucial for validating the outcomes of our tests and detecting any deviations in `probtests` results, is maintained in the [tests/data](tests/data) directory. This approach guarantees that our tests are both comprehensive and reliable, safeguarding the integrity of our codebase.

### Formatting probtest source code

The probtest source code is formatted using multiple formatters.
Please install the pre-commit hooks (after installing all Python requirements
including the `pre-commit` package):

```console
pre-commit install
```

This hook will be executed automatically whenever you commit.
It will check your files and format them according to its rules.
If files have to be formatted, committing will fail.
Just stage and commit again to finalize the commit.
You can also run the following command, to trigger the pre-commit action without
actually committing:
```console
pre-commit run --all-files
```
