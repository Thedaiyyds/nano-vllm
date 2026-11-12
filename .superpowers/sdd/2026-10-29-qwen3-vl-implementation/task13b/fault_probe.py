"""Process-local faults; production files remain untouched. Run from repo root."""

import inspect
from pathlib import Path
import sys
import textwrap
from unittest.mock import patch

import pytest

fault = sys.argv[1]


class FaultPlugin:
    def pytest_sessionstart(self, session):
        # Plugin rewriting is initialized now; importing the model earlier imports
        # anyio via transformers/httpcore before pytest can mark it for rewriting.
        from nanovllm.engine.model_runner import ModelRunner
        import nanovllm.engine.model_runner as module

        if fault == "live_output":
            outputs = []
            original = ModelRunner.run_model

            def broken(*args, **kwargs):
                output = original(*args, **kwargs)
                outputs.append(output)
                return output

            target, method = "run_model", broken
        else:
            target = (
                "profile_vl_memory"
                if fault == "cancel_activation"
                else "_allocate_vl_kv_cache"
            )
            source = textwrap.dedent(inspect.getsource(getattr(ModelRunner, target)))
            if fault == "cancel_activation":
                source = source.replace(
                    "activation = peak - current", "activation = peak - peak"
                )
            elif fault == "omit_reserve":
                source = source.replace("- config.kv_cache_reserve_bytes", "- 0")
            elif fault == "clamp_negative":
                source = source.replace(
                    "select_kv_block_count(config, available)",
                    "select_kv_block_count(config, max(available, 0))",
                )
            else:
                raise ValueError(fault)
            namespace = dict(vars(module))
            exec(compile(source, "<controlled-runtime-fault>", "exec"), namespace)
            method = namespace[target]
        mutation = patch.object(ModelRunner, target, method)
        mutation.start()
        session.config.add_cleanup(mutation.stop)


nodes = (
    ["tests/test_qwen3_vl_memory.py::MemoryTests", "-m", "gpu", "--require-gpu"]
    if fault == "live_output"
    else ["tests/test_qwen3_vl_memory.py::BudgetIntegrationTests", "-m", "not gpu"]
)
# Keep the original acceptance evidence intact when rerunning diagnostics.
report_dir = Path(__file__).with_name("m1")
report_dir.mkdir(exist_ok=True)
sys.exit(
    pytest.main(
        [*nodes, "-q", "--junitxml=" + str(report_dir / ("fault-" + fault + ".xml"))],
        plugins=[FaultPlugin()],
    )
)
