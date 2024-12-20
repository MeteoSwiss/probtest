#!/usr/bin/env python
""" Setup .env with local eccodes definition """

import os
import shutil
import subprocess


def main():
    base_path = os.getcwd()

    resource_path = os.path.join(base_path, "resource")

    eccodes_definition_path = os.path.join(resource_path, "eccodes")
    cosmo_definition_path = os.path.join(resource_path, "eccodes-cosmo-resources")

    # remove existing resource
    if os.path.exists(resource_path):
        shutil.rmtree(resource_path)

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
    env_file = ".env"
    if os.path.exists(env_file):
        os.remove(env_file)

    # Write to .env file
    env_content = (
        f"ECCODES_DEFINITION_PATH={eccodes_definition_path}/definitions:"
        f"{cosmo_definition_path}/definitions\n"
    )
    with open(env_file, "w", encoding="utf-8") as file:
        file.write(env_content)


if __name__ == "__main__":
    main()
