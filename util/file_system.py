from pathlib import Path

from util.log_handler import logger


def file_names_from_pattern(dir_name, file_pattern):
    err = 0
    file_names = [p.name for p in Path(dir_name).glob(file_pattern)]
    if len(file_names) < 1:
        logger.warning(
            "no files found in '{}' for pattern '{}'".format(dir_name, file_pattern)
        )
        logger.debug("the directory contains the following files:")
        for f in Path(dir_name).glob("*"):
            logger.debug(f.name)
        err = 1

    return file_names, err
