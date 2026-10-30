"""Legacy compatibility only: small CPU cases and two real-GPU checks."""

import json
from pathlib import Path

import pytest


@pytest.mark.hf_reference
class TestLegacyCPU:
    @pytest.fixture(autouse=True)
    def disable_compile(self, monkeypatch):
        monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")

    def test_public_exports_and_qwen3_config(self, tmp_path):
        from nanovllm import LLM, SamplingParams
        from nanovllm.config import Config

        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3",
                    "max_position_embeddings": 1024,
                    "hidden_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                }
            )
        )
        cfg = Config(str(tmp_path), max_model_len=4096, tensor_parallel_size=1)
        assert cfg.hf_config.model_type == "qwen3"
        assert cfg.max_model_len == 1024
        assert cfg.kvcache_block_size == 256
        assert LLM.__name__ == "LLM"
        assert SamplingParams(temperature=0.7).max_tokens == 64
        with pytest.raises(AssertionError):
            Config(str(tmp_path), kvcache_block_size=128)

    def test_prefix_hit_miss_and_reference_release(self):
        from nanovllm.engine.block_manager import BlockManager
        from nanovllm.engine.sequence import Sequence

        manager = BlockManager(4, 256)
        a = Sequence([1] * 300)
        manager.allocate(a, manager.can_allocate(a))
        a.num_scheduled_tokens = 300
        manager.hash_blocks(a)
        b = Sequence([1] * 256 + [2] * 44)
        assert manager.can_allocate(b) == 1
        manager.allocate(b, 1)
        shared = a.block_table[0]
        assert b.block_table[0] == shared
        assert manager.blocks[shared].ref_count == 2
        assert manager.can_allocate(Sequence([3] * 300)) == -1  # only one free block
        manager.deallocate(a)
        assert manager.blocks[shared].ref_count == 1
        manager.deallocate(b)
        assert not manager.used_block_ids
        assert len(manager.free_block_ids) == 4
        assert all(block.ref_count == 0 for block in manager.blocks)
        assert manager.can_allocate(Sequence([3] * 300)) == 0

    def test_chunked_prefill_does_not_generate_early(self):
        from nanovllm.engine.sequence import Sequence, SequenceStatus
        from nanovllm.sampling_params import SamplingParams
        from nanovllm.engine.scheduler import Scheduler
        from types import SimpleNamespace

        scheduler = Scheduler(
            SimpleNamespace(
                max_num_seqs=1,
                max_num_batched_tokens=128,
                eos=99,
                kvcache_block_size=256,
                num_kvcache_blocks=4,
            )
        )
        seq = Sequence([1] * 300, SamplingParams(max_tokens=2))
        scheduler.add(seq)
        for expected_cached in (128, 256):
            scheduled, prefill = scheduler.schedule()
            assert scheduled == [seq] and prefill
            scheduler.postprocess(scheduled, [8], prefill)
            assert seq.num_cached_tokens == expected_cached
            assert seq.num_completion_tokens == 0
            assert seq.status == SequenceStatus.WAITING
        scheduled, prefill = scheduler.schedule()
        assert seq.num_scheduled_tokens == 44
        scheduler.postprocess(scheduled, [8], prefill)
        assert seq.completion_token_ids == [8]
        assert seq.status == SequenceStatus.RUNNING
        scheduled, prefill = scheduler.schedule()
        assert not prefill
        scheduler.postprocess(scheduled, [9], prefill)
        assert seq.completion_token_ids == [8, 9]
        assert scheduler.is_finished()
        assert not scheduler.block_manager.used_block_ids

    def test_memory_pressure_preempts_and_requeues(self):
        from types import SimpleNamespace
        from nanovllm.engine.sequence import Sequence, SequenceStatus
        from nanovllm.sampling_params import SamplingParams
        from nanovllm.engine.scheduler import Scheduler

        scheduler = Scheduler(
            SimpleNamespace(
                max_num_seqs=2,
                max_num_batched_tokens=512,
                eos=99,
                kvcache_block_size=256,
                num_kvcache_blocks=2,
            )
        )
        a, b = [Sequence([v] * 256, SamplingParams(max_tokens=2)) for v in (1, 2)]
        scheduler.add(a)
        scheduler.add(b)
        scheduled, prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [8, 9], prefill)
        scheduled, prefill = scheduler.schedule()
        assert scheduled == [a] and not prefill
        assert b.status == SequenceStatus.WAITING and b.is_prefill
        assert b.block_table == [] and b.num_cached_tokens == 0
        assert list(scheduler.waiting) == [b]
        assert len(a.block_table) == 2
        scheduler.postprocess(scheduled, [10], prefill)
        assert a.is_finished
        scheduled, prefill = scheduler.schedule()
        assert scheduled == [b] and prefill
        assert (
            b.num_scheduled_tokens == 257
        )  # prompt plus the token generated before preemption
        scheduler.postprocess(scheduled, [11], prefill)
        assert scheduler.is_finished()
        assert not scheduler.block_manager.used_block_ids

    def test_positive_temperature_sampling_is_seeded_and_not_greedy(self):
        import torch
        from nanovllm.layers.sampler import Sampler
        from nanovllm import SamplingParams

        assert SamplingParams(temperature=0.7).temperature == 0.7
        sampler = Sampler()
        logits = torch.zeros(32, 3)
        temperatures = torch.full((32,), 0.7)
        torch.manual_seed(2026)
        first = sampler(logits.clone(), temperatures)
        torch.manual_seed(2026)
        second = sampler(logits.clone(), temperatures)
        assert torch.equal(first, second)
        assert (
            first.unique().numel() > 1
        )  # an argmax-only replacement would return only token 0
        decisive = sampler(
            torch.tensor([[-1000.0, 0.0, -1000.0], [0.0, -1000.0, -1000.0]]),
            torch.tensor([0.7, 1.0]),
        )
        assert decisive.tolist() == [1, 0]


def test_worker_propagates_failure_and_output(tmp_path, monkeypatch):
    import os
    from tests.support import run_worker

    (tmp_path / "legacy_test_worker.py").write_text(
        'print("worker evidence")\nraise SystemExit(7)\n'
    )
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    result = run_worker("legacy_test_worker", [])
    assert result.returncode == 7
    assert result.stdout is not None, (
        "worker stdout must be captured for validation evidence"
    )
    assert "worker evidence" in result.stdout


@pytest.mark.gpu
@pytest.mark.checkpoint
def test_qwen3_short_generation_smoke(tmp_path):
    import os
    from tests.support import run_worker, validate_legacy_record

    model = Path(os.environ.get("QWEN3_MODEL", "/home/thedai/huggingface/Qwen3-0.6B"))
    output = tmp_path / "smoke.json"
    result = run_worker("tests.support", ["--legacy-worker", str(model), str(output)])
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    validate_legacy_record(record)
    assert (
        record["vocab_size"]
        == json.loads((model / "config.json").read_text())["vocab_size"]
    )


@pytest.mark.gpu
@pytest.mark.checkpoint
def test_qwen3_fixed_input_regression(tmp_path):
    import os
    from tests.support import run_worker, assert_legacy_match

    baseline = Path(
        os.environ.get(
            "QWEN3_LEGACY_BASELINE", "tests/fixtures/legacy_qwen3_fixed.json"
        )
    )
    if not baseline.is_file():
        pytest.fail(
            "BLOCKED: original-HEAD baseline absent; explicitly capture it on real GPU first"
        )
    model = Path(os.environ.get("QWEN3_MODEL", "/home/thedai/huggingface/Qwen3-0.6B"))
    output = tmp_path / "current.json"
    result = run_worker("tests.support", ["--legacy-worker", str(model), str(output)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert_legacy_match(
        json.loads(output.read_text()), json.loads(baseline.read_text())
    )


@pytest.mark.parametrize("field", ["model_files_sha256", "sampling", "source_revision"])
def test_fixed_regression_rejects_metadata_drift(field):
    from copy import deepcopy
    from tests.support import assert_legacy_match, LEGACY_REFERENCE_REVISION

    expected = json.loads(
        (Path(__file__).parent / "fixtures/legacy_qwen3_fixed.json").read_text()
    )
    expected["model_files_sha256"] = {
        "config.json": "original-config",
        "tokenizer.json": "original-tokenizer",
        "tokenizer_config.json": "original-template",
    }
    expected["sampling"] = {"temperature": 0.7, "max_tokens": 8, "ignore_eos": False}
    expected["source_revision"] = LEGACY_REFERENCE_REVISION
    actual = deepcopy(expected)
    if field == "source_revision":
        expected[field] = "wrong-original-revision"
        # Exercise the fixture-loading boundary, not the two temporary baseline runs.
        with pytest.raises(AssertionError, match="original source revision"):
            assert_legacy_match(actual, expected, require_reference=True)
        return
    if field == "model_files_sha256":
        actual[field]["config.json"] = "changed-config"
    else:
        actual[field]["temperature"] = 0.9
    with pytest.raises(AssertionError):
        assert_legacy_match(actual, expected)
