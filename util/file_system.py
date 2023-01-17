from pathlib import Path

from util.log_handler import logger


def file_names_from_regex(dir_name, file_regex):
    err = 0
    file_names = [p.name for p in Path(dir_name).glob(file_regex)]
    if len(file_names) < 1:
        logger.warning(
            "no files found in '{}' for regex '{}'".format(dir_name, file_regex)
        )
        logger.debug("the directory contains the following files:")
        for f in Path(dir_name).glob("*"):
            logger.debug(f.name)
        err = 1

    return file_names, err
