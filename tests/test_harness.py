"""Exercise the test gate through isolated pytest subprocesses."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_suite(
    tmp_path, source, *args, cuda=False, extra_root=False, collection_module=None
):
    """Use a CUDA availability double only for testing the gate itself."""
    suite = tmp_path / "tests"
    suite.mkdir()
    shutil.copy(ROOT / "tests/conftest.py", suite / "conftest.py")
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    (suite / "test_case.py").write_text(source)
    (tmp_path / "torch.py").write_text(
        f"class cuda:\n    @staticmethod\n    def is_available(): return {cuda!r}\n"
    )
    if collection_module is not None:
        (suite / "test_missing.py").write_text(collection_module)
    if extra_root:
        (tmp_path / "test_experiment.py").write_text("def test_experiment(): pass\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_no_gpu_required_fails(tmp_path):
    result = run_suite(
        tmp_path,
        "import pytest\n@pytest.mark.gpu\ndef test_gpu(): pass\n",
        "--require-gpu",
        "-m",
        "gpu",
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "CUDA" in result.stdout + result.stderr


@pytest.mark.parametrize(
    "body",
    [
        'pytest.skip("unavailable")',
        'pytest.xfail("unfinished")',
    ],
)
def test_gpu_skip_rejected(tmp_path, body):
    result = run_suite(
        tmp_path,
        f"import pytest\n@pytest.mark.gpu\ndef test_gpu(): {body}\n",
        "--require-gpu",
        "-m",
        "gpu",
        cuda=True,
    )
    assert result.returncode != 0, result.stdout + result.stderr


def test_gpu_xpass_rejected(tmp_path):
    result = run_suite(
        tmp_path,
        "import pytest\n@pytest.mark.gpu\n@pytest.mark.xfail\ndef test_gpu(): pass\n",
        "--require-gpu",
        "-m",
        "gpu",
        cuda=True,
    )
    assert result.returncode != 0, result.stdout + result.stderr


def test_require_gpu_rejects_empty_selection(tmp_path):
    result = run_suite(tmp_path, "def test_cpu(): pass\n", "--require-gpu")
    assert result.returncode != 0, result.stdout + result.stderr


def test_gpu_requires_explicit_opt_in(tmp_path):
    result = run_suite(
        tmp_path, "import pytest\n@pytest.mark.gpu\ndef test_gpu(): pass\n"
    )
    assert result.returncode != 0, result.stdout + result.stderr


def test_cpu_collection_and_execution_do_not_import_torch(tmp_path):
    result = run_suite(
        tmp_path,
        'import sys\ndef test_cpu(): assert "torch" not in sys.modules\n',
        "-m",
        "not gpu and not hf_reference and not checkpoint",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gpu_pass_is_reported(tmp_path):
    result = run_suite(
        tmp_path,
        "import pytest\n@pytest.mark.gpu\ndef test_gpu(): pass\n",
        "--require-gpu",
        "-m",
        "gpu",
        cuda=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_root_experiments_not_collected(tmp_path):
    result = run_suite(
        tmp_path, "def test_cpu(): pass\n", "--collect-only", extra_root=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests/test_case.py::test_cpu" in result.stdout
    assert "test_experiment.py" not in result.stdout


def test_quick_cpu_excludes_references_and_checkpoints(tmp_path):
    source = (
        "import pytest\ndef test_cpu(): pass\n"
        "@pytest.mark.checkpoint\ndef test_checkpoint(): pass\n"
        "@pytest.mark.hf_reference\ndef test_reference(): pass\n"
    )
    result = run_suite(
        tmp_path,
        source,
        "--collect-only",
        "-m",
        "not gpu and not hf_reference and not checkpoint",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_checkpoint" not in result.stdout
    assert "test_reference" not in result.stdout


def test_full_cpu_includes_reference_and_checkpoint(tmp_path):
    source = (
        "import pytest\ndef test_cpu(): pass\n"
        "@pytest.mark.checkpoint\ndef test_checkpoint(): pass\n"
        "@pytest.mark.hf_reference\ndef test_reference(): pass\n"
    )
    result = run_suite(tmp_path, source, "--collect-only", "-m", "not gpu")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_checkpoint" in result.stdout
    assert "test_reference" in result.stdout


def test_manifest_does_not_import_cuda_runtime(tmp_path):
    result = run_suite(
        tmp_path,
        "import sys, pytest\nfrom conftest import environment_manifest\n"
        "def test_metadata():\n"
        "    manifest = environment_manifest()\n"
        '    assert manifest["packages"]["pytest"]["version"] == pytest.__version__\n'
        '    assert "torch" not in sys.modules\n',
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "skip_source",
    [
        'pytest.skip("missing GPU resource", allow_module_level=True)',
        'pytest.importorskip("_qwen_absent_reference_fixture_74829")',
    ],
)
def test_collection_skip_rejected_even_when_another_gpu_test_passes(
    tmp_path, skip_source
):
    result = run_suite(
        tmp_path,
        "import pytest\n@pytest.mark.gpu\ndef test_gpu(): pass\n",
        "--require-gpu",
        "-m",
        "gpu",
        cuda=True,
        collection_module="import pytest\npytestmark = pytest.mark.gpu\n"
        + skip_source
        + "\n",
    )
    assert "1 passed, 1 skipped" in result.stdout
    assert result.returncode != 0, result.stdout + result.stderr
