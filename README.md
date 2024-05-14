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

### The init command

This command sets up the configuration file. For more help on the command line arguments for `init`, see

```
python probtest.py init --help
```

The `--template-name` argument can be used to specify the template from which the configuration file is created. One of the templates provided by probtest is `templates/ICON.jinja` which is used as the default in case no other template name is provided. The init command replaces all placeholder values in the template with the values given as command line arguments. All other probtest commands can then read from the configuration file. The name of the configuration file to use is read from the `PROBTEST_CONFIG` environment variable. If this is not set explicitly, probtest will look for a file called `probtest.json` in the current directory.

Setting up the configuration file with `init` may not be fitted perfectly to where you want your probtest files to be. In that case, you can manually edit the file after creation. Alternatively, you can add arguments for your probtest commands on the command line which take precedence over the configuration file defaults. For more help on the options on a specific command, see

```
python probtest.py {command} --help
```

### Example: Check the output of an experiment

Objective: Run the mch_opr_r04b07 ICON experiment and check if the output of the run is ok. Probtest requires some additional python packages. On Piz Daint, there is a pre-installed python environment which can be loaded with:

```
source /project/g110/icon/probtest/conda/miniconda/bin/activate probtest
```

Alternatively, all requirements can be easily installed with conda:

```
./setup_miniconda.sh
./setup_env.sh -n probtest -d -u
```

Once set up, probtest can generate the config file according to your needs:

```
python probtest.py init --codebase-install /path/to/the/ICON/Installation/ --experiment-name mch_opr_r04b07 --reference /path/to/icon-test-references/daint_cpu_pgi/ --file-id NetCDF "*atm_3d_ml*" --file-id NetCDF "*atm_3d_hl*"
```

This will create a `probtest.json` file in the current directory. This file contains all information needed by probtest to process the ICON experiment.

With everything set up properly, the chain of commands can be invoked to run the CPU reference binary (`run-ensemble`), generate the statistics files used for probtest comparisons (`stats`) and generate tolerances from these files (`tolerance`).

```
python probtest.py run-ensemble
python probtest.py stats --ensemble
python probtest.py tolerance
```

Note the `--ensemble` option which is set to take precedence over the default `False` from the configuration and make probtest process the model output from each ensemble generated by `run-ensemble`. These commands will generate a number of files:

- `stats_ref.csv`: contains the post-processed output from the unperturbed reference run
- `stats_{member_num}.csv`: contain the post-processed output from the perturbed reference runs (only needed temporarily to generate the tolerance file)
- `mch_opr_r04b07_tolerance.csv`: contains tolerance ranges computed from the stats-files

These can then be used to compare against the output of a test binary (usually a GPU binary). For that, manually run the `exp.mch_opr_r04b07.run` experiment with the test binary to produce the test output. Then use probtest to generate the stats file for this output:

```
python probtest.py stats --model-output-dir /path/to/test-icon/experiments/mch_opr_r04b07 --stats-file-name stats_cur.csv
```

Note how `--model-output-dir` is set to take precedence over the default which points to the reference binary output to now point to the test binary output as well as the name of the generated file with `--stats-file-name` to avoid name clash with the stats file from the reference. This command will generate the following file:

- `stats_cur.csv`: contains the post-processed output of the test binary model output.

Now all files needed to perform a probtest check are available; the reference file `stats_ref.csv`, the test file `stats_cur.csv` as well as the tolerance range `mch_opr_r04b07_tolerance.csv`. Providing these files to `check` will perform the check:

```
python probtest.py check --input-file-ref stats_ref.csv --input-file-cur stats_cur.csv
```

Note that the reference `--input-file-ref` and test stats files `--input-file-cur` need to be set by command line arguments. This is because the default stored in the `ICON.jinja` template is pointing to two files from the ensemble as a sanity check.

## Developing in probtest

The tool has been developed using Visual Studio Code. To ease development there is a `.vscode/launch.json` file containing the configurations to run all of the probtest commands discussed in [Commands](#commands) based on data generated by ICON and probtest. All input data needed to run are generated by a [Jenkins plan](https://jenkins-mch.cscs.ch/job/Probtest/job/generate_probtest_testdata_github) and stored on daint:/project/g110/probtest_testdata. If you plan to make relevant changes in probtest, you need to run the Jenkins plan and update the hashes in the files `test_reference` and `.env` accordingly.

### Code formatting

Code is formatted using black and isort. Please install the pre-commit hooks (after installing all Python requirements including the `pre-commit` package):

```
pre-commit install
```

This hook will be executed automatically whenever you commit. It will check your files and format them according to its rules. If files have to be formatted, committing will fail. Just commit again to finalize the commit. You can also run the following command, to trigger the pre-commit action without actually committing:

```
pre-commit run --all-files
```

If you are using VSCode with the settings provided by this repository in `.vscode/settings.json` formatting is already enabled on save.

### Debugging individual commands

For testing, probtest has data stored in two different places.

1. Probtest input data (generated by ICON/externals/probtest) is saved on Piz Daint in /project/g110/probtest_testdata/

2. Probtest performance data. This is very little data and is saved inside this repo in tests/data/

To execute all tests, you first need to download the probtest input data, then run the initialization script to generate the associated config file from the template:

    cd /local/path/to/probtest
    REFERENCE_DATA=./reference_data # Directory to read reference data
    PROBTEST_CUR_DATA=./probtest_data # Directory to write probtest output
    test_reference=$(cat test_reference)
    scp -r daint:/project/g110/probtest_testdata/$test_reference $REFERENCE_DATA
    python probtest.py init --codebase-install $REFERENCE_DATA/icon_data --reference $PROBTEST_CUR_DATA --template-name templates/testdata.jinja --experiment-name atm_amip_les_test --config testdata.json --file-id NetCDF "*atm_3d_ml*" --file-id NetCDF "*atm_3d_hl*" --member-num 2,5

Now you can run and debug any probtest command from the _Run_ tab in VS code. (Note that the template `testdata.jinja` treats `codebase-install` differently than the default `ICON.jinja`.)

### Executing end to end tests

The reference data bundle not only includes data produced by ICON but also reference output generated by probtest. To assert that this output is not affected by local changes to probtest, one can execute the following steps:

Generate a set of probtest output with the local probtest version:

    ICON_DATA=$REFERENCE_DATA/icon_data PROBTEST_DATA=./probtest_data ./scripts/generate_icon_testdata.sh

Adjust the `.env` file to reflect the directory structure:

`PROBTEST_REF_DATA`: path to the `probtest_data` subdirectory of the reference data.
`PROBTEST_CUR_DATA`: path to the output directory you chose in the last step.
`PROBTEST_TEST_EXPERIMENT`: full sets of reference data are generated for the `mch_opr_r04b07` experiment.

Finally, you can execute the end to end tests in one of two ways:

#### VSCode Testing

In VSCode, the python interpreter checks for the `.env` file and reads the variables from it into the run environment. The settings needed to make VSCode aware of pythons `unittest`s are already in the `.vscode/settings.json` file. Therefore, the tests can simply be executed by switching to the "Testing" view and running all tests.

#### Command line on Piz Daint

The tests can be run from the command line as well. The variables in the `.env` file can be used for testing on Piz Daint. Because the `.env` file does not export any variables, we need a little work-around. Execute

     . .env && export $(cut -d= -f1 .env)

in your shell to export the variables set in the `.env` file. Next, execute the end to end tests as described above. The \<icon-commit\> and \<probtest-commit> in the command below are only the first four digits of the respective commit hashes. The hashes used for current testing, can be found in the `test_reference` file.

    ICON_DATA=/project/g110/probtest_testdata/i-<icon-commit>_p-<probtest-commit>/icon_data PROBTEST_DATA=./probtest_data ./scripts/generate_icon_testdata.sh

If the test data is generated without any error, you can execute the `unittest`s with:

    python3 -m unittest
