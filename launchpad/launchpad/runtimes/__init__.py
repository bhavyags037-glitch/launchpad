"""Importing this package registers every runtime whose deps are installed.

Optional extras stay optional: a missing langchain/databricks install just means
that runtime is absent from RUNTIMES, and POST /agents rejects it with a 400.
"""

from launchpad.runtimes import echo  # noqa: F401

for _mod in ("langchain_react", "databricks_mlflow"):
    try:
        __import__(f"launchpad.runtimes.{_mod}")
    except ImportError:
        pass
