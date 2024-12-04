#!/usr/bin/env python
""" Setup .env with local eccodes definition """

import os
import subprocess

base_path = os.getcwd()

eccodes_definition_path = os.path.join(base_path, "resource", "eccodes")
cosmo_definition_path = os.path.join(base_path, "resource", "eccodes-cosmo-resources")

# Clone repositories
subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "-b",
        "2.39.0",
        "https://github.com/ecmwf/eccodes.git",
        eccodes_definition_path,
    ],
    check=True,
)
subprocess.run(
    [
        "git",
        "clone",
        "--depth",
        "1",
        "-b",
        "v2.36.0.3",
        "https://github.com/COSMO-ORG/eccodes-cosmo-resources.git",
        cosmo_definition_path,
    ],
    check=True,
)

# Remove deprecated directory
deprecated_path = os.path.join(eccodes_definition_path, "deprecated")
if os.path.exists(deprecated_path):
    for root, dirs, files in os.walk(deprecated_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(deprecated_path)

# Remove existing .env file
ENV_FILE = ".env"
if os.path.exists(ENV_FILE):
    os.remove(ENV_FILE)

# Write to .env file
ENV_CONTENT = (
    f"ECCODES_DEFINITION_PATH={eccodes_definition_path}/definitions:"
    f"{cosmo_definition_path}/definitions\n"
)
with open(ENV_FILE, "w", encoding="utf-8") as file:
    file.write(ENV_CONTENT)
