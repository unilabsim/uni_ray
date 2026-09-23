"""CI-skip discipline (#299): with warp/mujoco absent, the GPU/runtime tests
must skip cleanly while the pure contract/boundary tests always run.

Runs the full suite in a subprocess whose meta path blocks ``warp`` and
``mujoco`` (mirroring a CI runner without the optional runtimes) and asserts
the run succeeds with skips instead of failures.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_suite_skips_cleanly_without_warp_and_mujoco() -> None:
    code = """
import sys


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"warp", "mujoco"}:
            # Mirrors a genuinely missing package (ModuleNotFoundError), which
            # is also what pytest.importorskip keys on.
            raise ModuleNotFoundError(f"No module named '{name.split('.')[0]}'")
        return None


sys.meta_path.insert(0, _Blocker())

import pytest

# This file must be excluded: it spawns the suite itself and would recurse.
sys.exit(pytest.main(["-q", "tests/", "--ignore=tests/test_ci_skip.py"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "skipped" in output
    assert "failed" not in output
    # The import-boundary tests are the always-on core contract coverage.
    assert "passed" in output
