import os
import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def new_ref() -> str:
    new_ref = os.path.join(tempfile.mkdtemp())
    print(f"\nNew reference data will be stored in {new_ref}")
    return new_ref
