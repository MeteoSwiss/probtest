"""
This module contains unit tests for the `helpers.py` module.
"""
import os
import shutil

import pytest

from tests.helpers import setup_test_directory


def test_setup_test_directory_creates_directory():
    test_dir = "test_dir"

    # Call the function to create the directory
    created_dir = setup_test_directory(test_dir)

    try:
        # Assert the directory was created
        assert os.path.exists(created_dir), f"Directory {created_dir} was not created."
        assert os.path.isdir(created_dir), f"{created_dir} is not a directory."

        # Test cleanup behavior
        setup_test_directory(test_dir)

        assert os.path.exists(created_dir), f"Directory {created_dir} was not recreated."
        assert os.path.isdir(created_dir), f"{created_dir} is not a directory."
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_setup_test_directory_removes_existing_directory():
    test_dir = "test_dir"

    try:
        # Create an existing directory with a file in it
        os.mkdir(test_dir)
        with open(os.path.join(test_dir, "test_file.txt"), "w") as f:
            f.write("This is a test file.")

        # Assert the file exists
        assert os.path.exists(os.path.join(test_dir, "test_file.txt")), "Test file was not created."

        # Call the function to recreate the directory
        setup_test_directory(test_dir)

        # Assert the directory was recreated and the file was removed
        assert not os.path.exists(os.path.join(test_dir, "test_file.txt")), "Test file was not removed."
        assert os.path.exists(test_dir), f"Directory {test_dir} was not recreated."
        assert os.path.isdir(test_dir), f"{test_dir} is not a directory."

    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_setup_test_directory_returns_absolute_path():
    test_dir = "test_dir"

    try:
        # Call the function to create the directory
        created_dir = setup_test_directory(test_dir)

        # Assert the returned path is absolute
        assert os.path.isabs(created_dir), f"Returned path {created_dir} is not absolute."

    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)