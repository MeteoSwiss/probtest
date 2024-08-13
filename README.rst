===============
Getting started
===============

---------------------
Setup dev environment
---------------------

Instead of running the steps below manually, you can install `mchbuild` and then
install, test and run the application:

.. code-block:: console

    $ pipx install mchbuild
    $ cd probtest
    $ mchbuild local.build local.test
    $ mchbuild local.run

More information can be found in :file:`.mch-ci.yml`.

------------------------------------------------
Install dependencies & start the project locally
------------------------------------------------

1. Enter the project folder:

.. code-block:: console

    $ cd probtest

2. Install packages

.. code-block:: console

    $ poetry install

3. Run the probtest

.. code-block:: console

    $ poetry run probtest greeting "ms/mr developer"

-------------------------------
Run the tests and quality tools
-------------------------------

1. Run tests

.. code-block:: console

    $ poetry run pytest

2. Run pylint

.. code-block:: console

    $ poetry run pylint probtest


3. Run mypy

.. code-block:: console

    $ poetry run mypy probtest


----------------------
Generate documentation
----------------------

.. code-block:: console

    $ poetry run sphinx-build doc doc/_build

Then open the index.html file generated in *probtest/build/sphinx/html*


.. HINT::
   All **poetry run** prefixes in the commands can be avoided if running them within the poetry shell
