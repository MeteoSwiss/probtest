#!/bin/bash
#
# Create conda environment with pinned or unpinned requirements


if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "Please simply call the script instead of sourcing it!"
  return
fi

# Default env names
DEFAULT_ENV_NAME="probtest"

# Default options
ENV_NAME="${DEFAULT_ENV_NAME}"
PYVERSION=3.12
PINNED=true
EXPORT=false
CONDA=conda
HELP=false

help_msg="Usage: $(basename "${0}") [-n NAME] [-p VER] [-u] [-e] [-m] [-h]

Options:
 -n NAME    Env name [default: ${DEFAULT_ENV_NAME}
 -p VER     Python version [default: ${PYVERSION}]
 -u         Use unpinned requirements (minimal version restrictions)
 -e         Export environment files (requires -u)
 -m         Use mamba instead of conda
 -h         Print this help message and exit
"

# Eval command line options
while getopts n:p:defhimu flag; do
    case ${flag} in
        n) ENV_NAME=${OPTARG};;
        p) PYVERSION=${OPTARG};;
        e) EXPORT=true;;
        h) HELP=true;;
        m) CONDA=mamba;;
        u) PINNED=false;;
        ?) echo -e "\n${help_msg}" >&2; exit 1;;
    esac
done

if ${HELP}; then
    echo "${help_msg}"
    exit 0
fi

echo "Setting up environment for installation"
eval "$(conda shell.bash hook)" || exit  # NOT ${CONDA} (doesn't work with mamba)
conda activate || exit # NOT ${CONDA} (doesn't work with mamba)

# Create new env; pass -f to overwriting any existing one
echo "Creating ${CONDA} environment"
${CONDA} create -n ${ENV_NAME} python=${PYVERSION} --yes || exit

# Install requirements in new env
if ${PINNED}; then
    echo "Pinned installation"
    ${CONDA} env update --name ${ENV_NAME} --file requirements/environment.yml || exit
else
    echo "Unpinned installation"
    ${CONDA} env update --name ${ENV_NAME} --file requirements/requirements.yml || exit
    if ${EXPORT}; then
        echo "Export pinned prod environment"
        ${CONDA} env export --name ${ENV_NAME} --no-builds | \grep -v '^prefix:' > requirements/environment.yml || exit
    fi
fi


# Setting ECCODES_DEFINITION_PATH:
${CONDA} activate ${ENV_NAME}

CONDA_LOC=${CONDA_PREFIX}
DEFINITION_VERSION="v2.35.0.1dm1"
DEFINITION_PATH_DEFAULT=${CONDA_LOC}/share/eccodes
DEFINITION_PATH_RESOURCES=${CONDA_LOC}/share/eccodes-cosmo-resources_${DEFINITION_VERSION}

git clone -b ${DEFINITION_VERSION} https://github.com/COSMO-ORG/eccodes-cosmo-resources.git ${DEFINITION_PATH_RESOURCES} || exit

${CONDA} env config vars set ECCODES_DEFINITION_PATH=${DEFINITION_PATH_DEFAULT}/definitions:${DEFINITION_PATH_RESOURCES}/definitions
${CONDA} env config vars set ECCODES_SAMPLES_PATH=${DEFINITION_PATH_DEFAULT}/samples
${CONDA} env config vars set GRIB_DEFINITION_PATH=${DEFINITION_PATH_DEFAULT}/definitions:${DEFINITION_PATH_RESOURCES}/definitions
${CONDA} env config vars set GRIB_SAMPLES_PATH=${DEFINITION_PATH_DEFAULT}/samples

echo "Variables saved to environment: "
${CONDA} env config vars list

${CONDA} deactivate
