"""Keep the diagnostic faults effective without assertion-rewrite noise."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

PROBE = Path(__file__).resolve().parents[1] / "fault_probe.py"


@pytest.mark.parametrize(
    ("fault", "failure"),
    [
        ("cancel_activation", "AssertionError: 0 != 400"),
        ("omit_reserve", "available 999"),
        ("clamp_negative", "available 0"),
    ],
)
def test_fault_reaches_assertion_without_rewrite_warning(tmp_path, fault, failure):
    # Override only the report destination: preserve all original fault XMLs.
    wrapper = """
import runpy, sys, pytest
original = pytest.main
xml = sys.argv[3]
def main(args, **kwargs):
    return original([*args, '--junitxml=' + xml], **kwargs)
pytest.main = main
probe, fault = sys.argv[1:3]
sys.argv = [probe, fault]
runpy.run_path(probe, run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", wrapper, str(PROBE), fault, str(tmp_path / "fault.xml")],
        env={
            **os.environ,
            "PYTHONPATH": str(Path.cwd()),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCHDYNAMO_DISABLE": "1",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert failure in output, output
    assert "PytestAssertRewriteWarning" not in output, output
