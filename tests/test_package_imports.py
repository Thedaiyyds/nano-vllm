"""Lightweight public APIs must not require the GPU execution backend."""

import os
import subprocess
import sys

import pytest

BLOCK_GPU_IMPORTS = """
import importlib.abc
import sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'flash_attn', 'triton'}:
            raise ImportError('GPU extension unexpectedly imported: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
"""


def run_import(source):
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    )


def test_sampling_params_does_not_import_gpu_extensions():
    result = run_import(
        BLOCK_GPU_IMPORTS + "\nfrom nanovllm import SamplingParams\n"
        "assert SamplingParams().max_tokens == 64\n"
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.hf_reference
def test_config_does_not_import_gpu_extensions():
    result = run_import(
        BLOCK_GPU_IMPORTS + "\nfrom nanovllm.config import Config\n"
        'assert Config.__name__ == "Config"\n'
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.hf_reference
def test_llm_public_export_still_resolves():
    result = run_import(
        "from nanovllm import LLM, SamplingParams\n"
        'assert LLM.__name__ == "LLM"\n'
        'assert SamplingParams.__name__ == "SamplingParams"\n'
    )
    assert result.returncode == 0, result.stdout + result.stderr
