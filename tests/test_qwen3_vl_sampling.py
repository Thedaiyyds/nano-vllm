"""Task 6: exact greedy, unchanged positive sampling, and EOS wiring."""

import atexit
import os
from types import SimpleNamespace
import unittest

import pytest


@pytest.fixture(autouse=True)
def eager_sampling(monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")


def legacy_sample(logits, temperatures):
    """The positive-only algorithm from baseline commit 89c1e40."""
    import torch

    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(
        dim=-1
    )


def scheduler_config(multimodal=True):
    return SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=512,
        max_model_len=1024,
        eos=99,
        eos_token_ids=frozenset({151645, 151643}),
        is_multimodal=multimodal,
        kvcache_block_size=256,
        num_kvcache_blocks=4,
    )


def scheduled_request(
    *, ignore_eos=False, max_tokens=3, multimodal=True, temperature=0.7
):
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams

    scheduler = Scheduler(scheduler_config(multimodal))
    seq = Sequence(
        [1, 2, 3],
        SamplingParams(
            temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos
        ),
    )
    scheduler.add(seq)
    scheduled, prefill = scheduler.schedule()
    assert scheduled == [seq] and prefill
    return scheduler, seq


def greedy_and_mixed_checks(device):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from nanovllm.layers.sampler import Sampler

    div_denominators = []

    class ObserveDivision(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.div_.Tensor:
                div_denominators.append(args[1].detach().clone())
            return func(*args, **(kwargs or {}))

    cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        logits = torch.tensor(
            [[1.0, 3.0, 2.0], [4.0, 4.0, 1.0], [-4.0, -1.0, -2.0]],
            device=device,
            dtype=torch.bfloat16,
        )
        zero = torch.zeros(3, device=device)
        cpu_before = torch.random.get_rng_state().clone()
        gpu_before = torch.cuda.get_rng_state().clone() if cuda_devices else None
        with ObserveDivision():
            actual = Sampler()(logits, zero)
        assert actual.tolist() == [1, 0, 1]
        assert actual.dtype == torch.int64 and actual.device == logits.device
        assert not div_denominators
        assert torch.equal(cpu_before, torch.random.get_rng_state())
        if cuda_devices:
            assert torch.equal(gpu_before, torch.cuda.get_rng_state())

        logits = torch.tensor(
            [[1.0, 3.0, 2.0], [0.0, 0.0, 0.0], [4.0, 4.0, 1.0], [0.0, 1.0, 2.0]],
            device=device,
        )
        temperatures = torch.tensor([0.0, 0.7, 0.0, 1.2], device=device)
        positive = torch.tensor([1, 3], device=device)
        torch.manual_seed(2026)
        expected_positive = legacy_sample(
            logits[positive].clone(), temperatures[positive]
        )
        expected_state = (
            torch.cuda.get_rng_state().clone()
            if cuda_devices
            else torch.random.get_rng_state().clone()
        )
        torch.manual_seed(2026)
        with ObserveDivision():
            actual = Sampler()(logits.clone(), temperatures)
        assert actual[[0, 2]].tolist() == [1, 0]
        assert torch.equal(actual[positive], expected_positive)
        assert div_denominators and all(
            torch.all(value > 0) for value in div_denominators
        )
        actual_state = (
            torch.cuda.get_rng_state() if cuda_devices else torch.random.get_rng_state()
        )
        assert torch.equal(actual_state, expected_state)


def positive_parity_checks(device):
    import torch
    from nanovllm.layers.sampler import Sampler

    cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.zeros(32, 257, device=device, dtype=dtype)
            temperatures = torch.linspace(0.2, 2.0, 32, device=device)
            torch.manual_seed(2026)
            expected = legacy_sample(logits.clone(), temperatures)
            expected_state = (
                torch.cuda.get_rng_state().clone()
                if cuda_devices
                else torch.random.get_rng_state().clone()
            )
            torch.manual_seed(2026)
            actual = Sampler()(logits.clone(), temperatures)
            assert torch.equal(actual, expected)
            assert actual.unique().numel() > 1
            actual_state = (
                torch.cuda.get_rng_state()
                if cuda_devices
                else torch.random.get_rng_state()
            )
            assert torch.equal(actual_state, expected_state)


@pytest.mark.hf_reference
class SamplingTests(unittest.TestCase):
    def test_zero_temperature_and_eos(self):
        import torch
        from nanovllm.layers.sampler import Sampler
        from nanovllm.sampling_params import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=1)
        self.assertEqual(
            Sampler()(
                torch.tensor([[1.0, 3.0, 2.0]]), torch.tensor([sp.temperature])
            ).item(),
            1,
        )
        scheduler, seq = scheduled_request(temperature=sp.temperature, max_tokens=1)
        scheduler.postprocess([seq], [151643], True)
        self.assertTrue(seq.is_finished)
        self.assertEqual(seq.completion_token_ids, [151643])
        self.assertFalse(scheduler.block_manager.used_block_ids)

    def test_greedy_and_mixed_rows_have_no_zero_division_or_extra_rng(self):
        greedy_and_mixed_checks("cpu")

    def test_all_positive_preserves_original_seeded_algorithm(self):
        positive_parity_checks("cpu")


@pytest.mark.parametrize(
    "temperature", [-1.0, float("nan"), float("inf"), -float("inf")]
)
def test_sampling_params_reject_invalid_temperature(temperature):
    from nanovllm.sampling_params import SamplingParams

    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True, float("nan"), float("inf")])
def test_sampling_params_require_positive_integer_budget(max_tokens):
    from nanovllm.sampling_params import SamplingParams

    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=max_tokens)


def test_finite_positive_temperature_below_previous_cutoff_is_valid():
    from nanovllm.sampling_params import SamplingParams

    assert SamplingParams(temperature=1e-12).temperature == 1e-12


@pytest.mark.hf_reference
@pytest.mark.parametrize(
    "temperature", [-1.0, float("nan"), float("inf"), -float("inf")]
)
def test_sampler_rejects_invalid_temperature_tensor(temperature):
    import torch
    from nanovllm.layers.sampler import Sampler

    with pytest.raises(ValueError, match="temperature"):
        Sampler()(torch.tensor([[1.0, 3.0, 2.0]]), torch.tensor([temperature]))


@pytest.mark.hf_reference
@pytest.mark.parametrize("token", [151645, 151643])
def test_each_vl_eos_stops_and_releases_real_blocks(token):
    scheduler, seq = scheduled_request()
    scheduler.postprocess([seq], [token], True)
    assert seq.is_finished and scheduler.is_finished()
    assert seq.completion_token_ids == [token]
    assert not scheduler.block_manager.used_block_ids
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)


@pytest.mark.hf_reference
def test_ignore_eos_ignores_whole_set_until_length_limit():
    scheduler, seq = scheduled_request(ignore_eos=True, max_tokens=3)
    for index, token in enumerate((151645, 151643, 7)):
        scheduled, prefill = ([seq], True) if index == 0 else scheduler.schedule()
        scheduler.postprocess(scheduled, [token], prefill)
        assert seq.num_completion_tokens == index + 1
        assert seq.is_finished == (index == 2)
    assert seq.completion_token_ids == [151645, 151643, 7]
    assert not scheduler.block_manager.used_block_ids


@pytest.mark.hf_reference
@pytest.mark.parametrize("token", [7, 151645, 151643])
@pytest.mark.parametrize("ignore_eos", [False, True])
def test_one_token_budget_always_stops_after_first_token(token, ignore_eos):
    scheduler, seq = scheduled_request(max_tokens=1, ignore_eos=ignore_eos)
    scheduler.postprocess([seq], [token], True)
    assert seq.is_finished and seq.completion_token_ids == [token]
    assert not scheduler.block_manager.used_block_ids


@pytest.mark.hf_reference
def test_original_qwen3_keeps_single_tokenizer_eos():
    scheduler, seq = scheduled_request(multimodal=False)
    scheduler.postprocess([seq], [151643], True)
    assert not seq.is_finished
    scheduled, prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [99], prefill)
    assert seq.is_finished and seq.completion_token_ids == [151643, 99]


@pytest.mark.hf_reference
@pytest.mark.checkpoint
@pytest.mark.parametrize("multimodal", [True, False])
def test_engine_eos_wiring_without_loading_a_gpu_model(monkeypatch, multimodal):
    import nanovllm.engine.llm_engine as engine_module
    from nanovllm.engine.sequence import Sequence
    from tests.reference_qwen3_vl import DEFAULT_MODEL

    # This isolates config/tokenizer/scheduler wiring; it is not a model smoke.
    class CPUOnlyRunner:
        def __init__(self, config, rank, events):
            config.num_kvcache_blocks = 2
            self.config = config

        def call(self, method):
            if method != "exit":
                raise AssertionError("no model inference belongs in this wiring test")

    monkeypatch.setattr(engine_module, "ModelRunner", CPUOnlyRunner)
    path = (
        os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
        if multimodal
        else os.environ.get("QWEN3_MODEL", "/home/thedai/huggingface/Qwen3-0.6B")
    )
    old_block_size = Sequence.block_size
    engine = engine_module.LLMEngine(
        path, max_num_seqs=1, enforce_eager=True, max_model_len=32
    )
    try:
        config = engine.model_runner.config
        if multimodal:
            assert config.eos == -1
            assert engine.scheduler.eos_token_ids == frozenset({151645, 151643})
        else:
            assert config.eos == engine.tokenizer.eos_token_id
            assert engine.scheduler.eos_token_ids == frozenset(
                {engine.tokenizer.eos_token_id}
            )
    finally:
        atexit.unregister(engine.exit)
        engine.exit()
        Sequence.block_size = old_block_size


@pytest.mark.gpu
@pytest.mark.hf_reference
class SamplingGPUTests(unittest.TestCase):
    def test_greedy_and_mixed_bfloat16_rows(self):
        greedy_and_mixed_checks("cuda")

    def test_positive_temperature_exactly_matches_original(self):
        positive_parity_checks("cuda")

    def test_real_vocabulary_greedy_matches_hf_argmax_rule(self):
        import torch
        from nanovllm.layers.sampler import Sampler

        logits = torch.zeros(3, 151936, device="cuda", dtype=torch.bfloat16)
        logits[0, 151645] = 3
        logits[1, 151643] = 4
        logits[2, 20:22] = 5
        expected = torch.argmax(
            logits, dim=-1
        )  # HF GenerationMixin do_sample=False rule.
        actual = Sampler()(logits, torch.zeros(3, device="cuda"))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.tolist(), [151645, 151643, 20])
