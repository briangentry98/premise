"""Compatibility wrapper.

Use run-gcam-to-premise.py directly for the CLI-enabled workflow.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).with_name("run-gcam-to-premise.py")
    runpy.run_path(str(target), run_name="__main__")
