"""
Backends of the run-ensemble command: how the run script of an experiment is
created.

``runscript``
    Copy an existing run script and replace assignments and the experiment
    name in it.

``mkexp``
    Generate the run script with mkexp (the experiment setup tool of ICON),
    passing the changes as command line arguments.
"""

BACKEND_RUNSCRIPT = "runscript"
BACKEND_MKEXP = "mkexp"
RUN_BACKENDS = [BACKEND_RUNSCRIPT, BACKEND_MKEXP]
