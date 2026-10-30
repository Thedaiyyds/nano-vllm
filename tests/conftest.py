"""Explicit GPU opt-in: skipped or unexecuted GPU tests never pass the gate."""

import importlib.metadata
import json
import platform

import pytest


def environment_manifest() -> dict:
    """Read distribution metadata without importing torch or initializing CUDA."""
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "triton",
        "flash-attn",
        "transformers",
        "numpy",
        "pytest",
        "ruff",
        "iniconfig",
        "pluggy",
    ):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            continue
        source = dist.read_text("direct_url.json")
        packages[name] = {
            "version": dist.version,
            "source": json.loads(source) if source else None,
        }
    return {"python": platform.python_version(), "packages": packages}


def pytest_addoption(parser):
    parser.addoption(
        "--require-gpu",
        action="store_true",
        help="Require selected GPU tests to execute successfully on CUDA",
    )


def pytest_configure(config):
    config.pluginmanager.register(GPUGate(config), "qwen_gpu_gate")


class GPUGate:
    def __init__(self, config):
        self.config = config
        self.selected = set()
        self.completed = set()
        self.invalid = set()

    def pytest_collection_finish(self, session):
        self.selected = {
            item.nodeid
            for item in session.items
            if item.get_closest_marker("gpu") is not None
        }
        required = self.config.getoption("--require-gpu")
        if self.config.option.collectonly:
            if required:
                raise pytest.UsageError(
                    "--require-gpu cannot be satisfied by collection only"
                )
            return
        if required and not self.selected:
            raise pytest.UsageError("--require-gpu selected no GPU tests")
        if self.selected and not required:
            raise pytest.UsageError(
                'GPU tests require explicit --require-gpu; use -m "not gpu" for CPU'
            )

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        if item.nodeid not in self.selected:
            return
        try:
            import torch

            available = torch.cuda.is_available()
        except Exception as exc:
            pytest.fail(f"CUDA availability check failed: {exc}", pytrace=False)
        if not available:
            pytest.fail(
                "CUDA is unavailable; GPU validation is BLOCKED, not skipped",
                pytrace=False,
            )

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        report = outcome.get_result()
        if item.nodeid not in self.selected:
            return
        if report.skipped or hasattr(report, "wasxfail"):
            self.invalid.add(item.nodeid)
        if report.when == "call" and report.passed and not hasattr(report, "wasxfail"):
            self.completed.add(item.nodeid)

    def pytest_collectreport(self, report):
        # A skipped module has no items/markers to inspect. In required-GPU
        # mode, any collection skip makes the validation incomplete.
        if self.config.getoption("--require-gpu") and report.skipped:
            self.invalid.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.config.getoption("--require-gpu"):
            if not self.selected or self.invalid or self.completed != self.selected:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter):
        if not self.selected:
            terminalreporter.write_line("GPU=NOT RUN (no GPU tests selected)")
        elif self.invalid or self.completed != self.selected:
            terminalreporter.write_line(
                "GPU=FAILED/BLOCKED (every selected GPU test must pass without skip/xfail)"
            )
        else:
            terminalreporter.write_line(
                "GPU call phases passed; final outcome also requires clean teardown"
            )
