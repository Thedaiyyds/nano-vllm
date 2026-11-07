"""Runner budgets/loading and explicit real-GPU full Prefill checks."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import pytest


pytestmark = pytest.mark.hf_reference
MODEL = Path(
    os.environ.get(
        "QWEN3_VL_MODEL",
        "/home/thedai/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/"
        "snapshots/89644892e4d85e24eaac8bacfd4f463576704203",
    )
)


def budget_config(**overrides):
    import torch

    values = dict(
        text_config=SimpleNamespace(
            num_hidden_layers=28,
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
        ),
        model_dtype=torch.bfloat16,
        max_model_len=4096,
        kvcache_block_size=256,
        kv_cache_reserve_bytes=1073741824,
        num_kvcache_blocks=-1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class BudgetTests(unittest.TestCase):
    @pytest.mark.checkpoint
    def test_real_nested_config_block_bytes_and_capacity(self):
        import torch
        from nanovllm.config import Config
        from nanovllm.engine.model_runner import kv_block_bytes, select_kv_block_count

        config = Config(str(MODEL), max_num_seqs=1, enforce_eager=True)
        self.assertIsNone(config.hf_config.dtype)
        self.assertIs(config.model_dtype, torch.bfloat16)
        self.assertEqual(
            kv_block_bytes(config.text_config, 256, torch.bfloat16), 29360128
        )
        self.assertEqual(select_kv_block_count(config, 469762048), 16)
        with self.assertRaises(ValueError):
            select_kv_block_count(config, 469762047)

    def test_block_bytes_uses_kv_heads_explicit_dimension_and_dtype(self):
        import torch
        from nanovllm.engine.model_runner import kv_block_bytes

        text = SimpleNamespace(
            num_hidden_layers=3,
            hidden_size=24,
            num_attention_heads=3,
            num_key_value_heads=2,
            head_dim=16,
        )
        # 2 (K,V) * 3 layers * 128 slots * 2 KV heads * 16 * itemsize.
        self.assertEqual(kv_block_bytes(text, 128, torch.bfloat16), 49152)
        self.assertEqual(kv_block_bytes(text, 128, torch.float32), 98304)

    def test_head_dimension_fallback(self):
        import torch
        from nanovllm.engine.model_runner import kv_block_bytes

        text = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=192,
            num_attention_heads=3,
            num_key_value_heads=2,
        )
        self.assertEqual(kv_block_bytes(text, 256, torch.bfloat16), 262144)
        text.head_dim = None
        self.assertEqual(kv_block_bytes(text, 256, torch.bfloat16), 262144)

    def test_capacity_is_derived_and_rounded_up(self):
        from nanovllm.engine.model_runner import select_kv_block_count

        for length, want in [(255, 1), (256, 1), (257, 2), (1024, 4), (4096, 16)]:
            with self.subTest(length=length):
                config = budget_config(max_model_len=length)
                self.assertEqual(select_kv_block_count(config, want * 29360128), want)
                with self.assertRaises(ValueError):
                    select_kv_block_count(config, want * 29360128 - 1)
        config = budget_config(kvcache_block_size=512)
        self.assertEqual(select_kv_block_count(config, 469762048), 8)

    def test_large_budget_is_capped_and_reserve_is_not_subtracted_twice(self):
        from nanovllm.engine.model_runner import select_kv_block_count

        config = budget_config()
        self.assertEqual(select_kv_block_count(config, 469762048), 16)
        config.kv_cache_reserve_bytes *= 3
        self.assertEqual(select_kv_block_count(config, 469762048), 16)
        self.assertEqual(select_kv_block_count(config, 4697620480), 16)

    def test_preserves_defaults_and_config(self):
        import torch
        from nanovllm.engine.model_runner import select_kv_block_count

        config = budget_config(num_kvcache_blocks=123)
        before = (
            torch.get_default_dtype(),
            torch.get_default_device(),
            torch.cuda.is_initialized(),
        )
        self.assertEqual(select_kv_block_count(config, 469762048), 16)
        self.assertEqual(config.num_kvcache_blocks, 123)
        self.assertEqual(
            (
                torch.get_default_dtype(),
                torch.get_default_device(),
                torch.cuda.is_initialized(),
            ),
            before,
        )


@pytest.mark.parametrize(
    "available", [-1, 0, 469762047, 469762048.0, True, "469762048"]
)
def test_invalid_or_insufficient_net_budget(available):
    from nanovllm.engine.model_runner import select_kv_block_count

    with pytest.raises(ValueError):
        select_kv_block_count(budget_config(), available)


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_model_len", 0),
        ("max_model_len", 1.5),
        ("kvcache_block_size", 0),
        ("kvcache_block_size", -256),
    ],
)
def test_invalid_capacity_dimensions(name, value):
    from nanovllm.engine.model_runner import select_kv_block_count

    with pytest.raises(ValueError):
        select_kv_block_count(budget_config(**{name: value}), 469762048)


@pytest.mark.parametrize(
    "name,value",
    [
        ("num_hidden_layers", 0),
        ("num_key_value_heads", -1),
        ("head_dim", 0),
        ("head_dim", 1.5),
    ],
)
def test_invalid_kv_dimensions(name, value):
    import torch
    from nanovllm.engine.model_runner import kv_block_bytes

    text = budget_config().text_config
    setattr(text, name, value)
    with pytest.raises(ValueError):
        kv_block_bytes(text, 256, torch.bfloat16)


@pytest.fixture
def model_loading_rank(tmp_path_factory):
    # Reuse the existing real Gloo fixture only for model-loading tests;
    # arithmetic-only BudgetTests must not require distributed initialization.
    from tests.test_qwen3_vl_loader import single_rank

    yield from single_rank.__wrapped__(tmp_path_factory)


def tiny_model_checkpoint(directory, *, multimodal):
    """Real small HF text weights, consumed by the actual native loader."""
    import torch
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from nanovllm.config import Config

    text = dict(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        vocab_size=64,
        max_position_embeddings=64,
        dtype="bfloat16",
        tie_word_embeddings=True,
    )
    if multimodal:
        text["rope_scaling"] = dict(
            rope_type="default", mrope_interleaved=True, mrope_section=[24, 20, 20]
        )
        config = Qwen3VLConfig(text_config=text, tie_word_embeddings=True)
    else:
        config = Qwen3Config(**text)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        hf_model = (
            Qwen3VLTextModel(config.text_config)
            if multimodal
            else Qwen3ForCausalLM(config)
        )
    state = {
        ("model.language_model." + key if multimodal else key): value.detach().clone()
        for key, value in hf_model.state_dict().items()
    }
    if multimodal:
        state["model.visual.patch_embed.proj.weight"] = torch.ones(1)
    config.save_pretrained(directory)
    save_file(state, str(directory / "model.safetensors"))
    return Config(
        str(directory), max_num_seqs=1, enforce_eager=True, max_model_len=64
    ), state


@pytest.mark.usefixtures("model_loading_rank")
class ModelSelectionTests(unittest.TestCase):
    def test_vl_native_model_dtype_and_strict_loader(self):
        import tempfile
        import torch
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration

        with tempfile.TemporaryDirectory() as directory:
            config, state = tiny_model_checkpoint(Path(directory), multimodal=True)
            runner = ModelRunner.__new__(ModelRunner)
            runner.config = config
            before = torch.get_default_dtype()
            runner._load_model()
        self.assertIsInstance(runner.model, Qwen3VLForConditionalGeneration)
        self.assertIsNone(config.hf_config.dtype)
        self.assertEqual(torch.get_default_dtype(), before)
        self.assertTrue(
            all(p.dtype == torch.bfloat16 for p in runner.model.parameters())
        )
        self.assertFalse(hasattr(runner.model.model, "visual"))
        text = runner.model.model.language_model
        self.assertIs(runner.model.lm_head.weight, text.embed_tokens.weight)
        self.assertTrue(
            torch.equal(
                text.embed_tokens.weight,
                state["model.language_model.embed_tokens.weight"].bfloat16(),
            )
        )
        self.assertTrue(
            torch.equal(
                text.layers[0].self_attn.qkv_proj.weight[:256],
                state[
                    "model.language_model.layers.0.self_attn.q_proj.weight"
                ].bfloat16(),
            )
        )
        self.assertEqual(
            set(runner.weight_load_report.loaded_keys),
            set(state) - {"model.visual.patch_embed.proj.weight"},
        )
        self.assertEqual(
            runner.weight_load_report.skipped_keys,
            ["model.visual.patch_embed.proj.weight"],
        )
        self.assertEqual(runner.weight_load_report.tied_aliases, ["lm_head.weight"])

    def test_legacy_model_and_loader_still_load_real_weights(self):
        import tempfile
        import torch
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.models.qwen3 import Qwen3ForCausalLM

        with tempfile.TemporaryDirectory() as directory:
            config, state = tiny_model_checkpoint(Path(directory), multimodal=False)
            runner = ModelRunner.__new__(ModelRunner)
            runner.config = config
            before = torch.get_default_dtype()
            runner._load_model()
        self.assertIsInstance(runner.model, Qwen3ForCausalLM)
        self.assertEqual(torch.get_default_dtype(), before)
        self.assertTrue(
            all(p.dtype == torch.bfloat16 for p in runner.model.parameters())
        )
        self.assertTrue(
            torch.equal(
                runner.model.model.embed_tokens.weight,
                state["model.embed_tokens.weight"].bfloat16(),
            )
        )
        self.assertTrue(
            torch.equal(
                runner.model.model.layers[0].self_attn.qkv_proj.weight[:256],
                state["model.layers.0.self_attn.q_proj.weight"].bfloat16(),
            )
        )

    def test_vl_missing_weight_rejected_and_loading_dtype_restored(self):
        import tempfile
        import torch
        from safetensors.torch import save_file
        from nanovllm.engine.model_runner import ModelRunner

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config, state = tiny_model_checkpoint(path, multimodal=True)
            del state["model.language_model.layers.0.self_attn.q_proj.weight"]
            save_file(state, str(path / "model.safetensors"))
            runner = ModelRunner.__new__(ModelRunner)
            runner.config = config
            before = torch.get_default_dtype()
            with self.assertRaisesRegex(ValueError, "missing.*q_proj"):
                runner._load_model()
            self.assertEqual(torch.get_default_dtype(), before)


EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task8b"
)


def prefill_config():
    from nanovllm.config import Config

    return Config(str(MODEL), max_num_seqs=1, enforce_eager=True)


def tiny_cuda_runner(directory):
    import torch
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.layers.sampler import Sampler

    config, _ = tiny_model_checkpoint(directory, multimodal=True)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.block_size = config.kvcache_block_size
    runner.world_size = 1
    runner.rank = 0
    runner.sampler = Sampler()
    with torch.device("cuda"):
        runner._load_model()
    return runner


@pytest.mark.gpu
class PrefillRunnerTests(unittest.TestCase):
    def tearDown(self):
        import torch
        from nanovllm.utils.context import reset_context

        reset_context()
        torch.cuda.empty_cache()

    def test_prefill_positions_and_noncontiguous_slots(self):
        import torch
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.engine.sequence import Sequence
        from nanovllm.utils.context import get_context

        runner = ModelRunner.__new__(ModelRunner)
        runner.config = prefill_config()
        runner.block_size = 256
        seq = Sequence([9707] * 257)
        seq.num_scheduled_tokens = 257
        seq.block_table = [2, 0]
        ids, positions = runner.prepare_prefill([seq])
        self.assertEqual(tuple(positions.shape), (3, 257))
        self.assertTrue(torch.equal(positions.cpu(), torch.arange(257).expand(3, -1)))
        self.assertEqual(ids.tolist(), [9707] * 257)
        context = get_context()
        want_slots = torch.cat((torch.arange(512, 768), torch.tensor([0]))).int()
        self.assertTrue(torch.equal(context.slot_mapping.cpu(), want_slots))
        self.assertEqual(context.cu_seqlens_q.tolist(), [0, 257])
        self.assertEqual(context.cu_seqlens_k.tolist(), [0, 257])
        self.assertEqual((context.max_seqlen_q, context.max_seqlen_k), (257, 257))
        self.assertIsNone(context.block_tables)  # full Prefill uses direct Q/K/V

    def test_rejects_partial_cached_or_batched_prefill(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.engine.sequence import Sequence
        from nanovllm.utils.context import get_context

        runner = ModelRunner.__new__(ModelRunner)
        runner.config = prefill_config()
        runner.block_size = 256
        for kind in ("partial", "cached", "batch"):
            with self.subTest(kind=kind):
                seq = Sequence([9707] * 4)
                seq.num_scheduled_tokens = 4
                seqs = [seq]
                if kind == "partial":
                    seq.num_scheduled_tokens = 2
                elif kind == "cached":
                    seq.num_cached_tokens = 1
                    seq.num_scheduled_tokens = 3
                else:
                    seqs.append(seq)
                before = get_context()
                with self.assertRaisesRegex(ValueError, "VL.*[Pp]refill"):
                    runner.prepare_prefill(seqs)
                self.assertIs(get_context(), before)

    @pytest.mark.usefixtures("model_loading_rank")
    def test_language_only_binding_and_capacity_failure(self):
        import tempfile
        import torch
        from torch import nn
        from nanovllm.layers.attention import Attention

        with tempfile.TemporaryDirectory() as directory:
            runner = tiny_cuda_runner(Path(directory))
        # A real decoy Attention outside language_model must never receive KV.
        runner.model.model.visual = nn.Module()
        decoy = runner.model.model.visual.attn = Attention(2, 128, 128**-0.5, 1)
        decoy.k_cache = decoy.v_cache = torch.tensor([41.0], device="cuda")
        original_decoy = decoy.k_cache
        runner.allocate_kv_cache()
        self.assertEqual(tuple(runner.kv_cache.shape), (2, 1, 1, 256, 1, 128))
        self.assertEqual(runner.kv_cache.dtype, torch.bfloat16)
        self.assertEqual(runner.kv_cache.device.type, "cuda")
        self.assertEqual(runner.config.num_kvcache_blocks, 1)
        attention = runner.model.model.language_model.layers[0].self_attn.attn
        self.assertEqual(attention.k_cache.data_ptr(), runner.kv_cache[0, 0].data_ptr())
        self.assertEqual(attention.v_cache.data_ptr(), runner.kv_cache[1, 0].data_ptr())
        self.assertIs(decoy.k_cache, original_decoy)
        self.assertEqual(decoy.k_cache.tolist(), [41.0])
        original_pool = runner.kv_cache
        reserve = runner.config.kv_cache_reserve_bytes
        runner.config.kv_cache_reserve_bytes = torch.cuda.mem_get_info()[1]
        try:
            with self.assertRaisesRegex(ValueError, "KV budget"):
                runner.allocate_kv_cache()
            self.assertIs(runner.kv_cache, original_pool)
            self.assertEqual(runner.config.num_kvcache_blocks, 1)
        finally:
            runner.config.kv_cache_reserve_bytes = reserve
        runner.model.model.language_model.extra_attention = Attention(
            2, 128, 128**-0.5, 1
        )
        with self.assertRaisesRegex(ValueError, "language Attention"):
            runner.allocate_kv_cache()
        self.assertIs(runner.kv_cache, original_pool)

    @pytest.mark.usefixtures("model_loading_rank")
    def test_warmup_profiles_full_context_despite_smaller_batch_budget(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            runner = tiny_cuda_runner(Path(directory))
        runner.config.max_num_batched_tokens = 32
        seen = []
        original_run = runner.run

        def observe(seqs, prefill):
            seen.extend(seq.num_scheduled_tokens for seq in seqs)
            return original_run(seqs, prefill)

        runner.run = observe  # observe, then execute the actual model and kernels
        runner.warmup_model()
        self.assertEqual(seen, [64])

    @pytest.mark.checkpoint
    def test_target_checkpoint_prefill_and_physical_kv(self):
        import json
        import tempfile
        import torch
        from tests.reference_qwen3_vl import assert_aligned, tensor_metrics
        from tests.support import run_worker
        from tests.test_qwen3_vl_text import CASES, load_oracle

        EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="prefill-", dir=EVIDENCE))
        reference = run_worker(
            "tests.test_qwen3_vl_text",
            ["--full-worker", "hf", "--output", str(directory)],
        )
        (directory / "hf.log").write_text(reference.stdout + reference.stderr)
        self.assertEqual(reference.returncode, 0, reference.stdout + reference.stderr)
        native = run_worker(
            "tests.test_qwen3_vl_runner", ["--prefill-worker", str(directory)]
        )
        (directory / "native.log").write_text(native.stdout + native.stderr)
        report = json.loads((directory / "native-report.json").read_text())
        self.assertEqual(
            report["bound_attention_layers"], 28, native.stdout + native.stderr
        )
        self.assertEqual(native.returncode, 0, native.stdout + native.stderr)
        self.assertEqual(report["kv_shape"], [2, 28, 16, 256, 8, 128])
        self.assertEqual(
            report["warmup"], [{"tokens": 4096, "positions_shape": [3, 4096]}]
        )
        self.assertEqual((report["loaded_keys"], report["skipped_keys"]), (310, 315))
        identity = json.loads((directory / "hf-identity.json").read_text())
        self.assertEqual(identity["backend"], "flash_attention_2")
        self.assertEqual(identity["gpu"], report["gpu"])
        self.assertEqual(identity["dtype"], "torch.bfloat16")
        self.assertEqual(
            identity["allow_bf16_reduced_precision_reduction"],
            report["allow_bf16_reduced_precision_reduction"],
        )
        metrics, failures = {}, []
        for case in CASES:
            actual = torch.load(
                directory / f"runner-{case}.pt", map_location="cpu", weights_only=True
            )
            expected = torch.load(
                directory / f"hf-{case}.pt", map_location="cpu", weights_only=True
            )
            oracle = load_oracle(case)
            self.assertTrue(
                torch.equal(actual.pop("input_ids"), oracle["input.input_ids"][0])
            )
            metrics[case] = {}
            for name, value in actual.items():
                limits = {name: {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}}
                if value.is_floating_point():
                    metrics[case][name] = tensor_metrics(value, expected[name])
                try:
                    assert_aligned(name, value, expected[name], limits)
                except AssertionError as error:
                    failures.append(f"{case}: {error}")
            self.assertTrue(report["cases"][case]["unused_page_nan"])
        (directory / "comparison.json").write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "failures": failures,
                    "standard": "finite + exact same-device/backend HF FA2; no wide BF16 thresholds",
                },
                indent=2,
            )
        )
        self.assertFalse(
            failures, f"{directory}: {failures[:8]} ({len(failures)} total)"
        )


def prefill_worker(directory):
    """Real production Runner init, warmup, allocation and Prefill in one worker."""
    import json
    import traceback
    import torch
    import torch.distributed as dist
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence
    from nanovllm.layers.attention import Attention
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.context import get_context, reset_context
    from tests.test_qwen3_vl_text import CASES, load_oracle

    config = prefill_config()
    report = {"warmup": [], "cases": {}}

    class ObservedRunner(ModelRunner):
        def prepare_prefill(self, seqs):
            ids, positions = super().prepare_prefill(seqs)
            if not hasattr(self, "kv_cache"):
                report["warmup"].append(
                    {"tokens": len(ids), "positions_shape": list(positions.shape)}
                )
            return ids, positions

        def allocate_kv_cache(self):
            free, total = torch.cuda.mem_get_info()
            stats = torch.cuda.memory_stats()
            report["allocation_inputs"] = dict(
                free=free,
                total=total,
                peak=stats["allocated_bytes.all.peak"],
                current=stats["allocated_bytes.all.current"],
                reserve=config.kv_cache_reserve_bytes,
            )
            return super().allocate_kv_cache()

    runner = ObservedRunner.__new__(ObservedRunner)
    previous_dtype = torch.get_default_dtype()
    try:
        runner.__init__(config, 0, [])
        for case in CASES:
            oracle = load_oracle(case)
            seq = Sequence(
                oracle["input.input_ids"][0].tolist(),
                SamplingParams(temperature=0, max_tokens=1),
            )
            seq.num_scheduled_tokens = len(seq)
            seq.block_table = [2] if len(seq) <= 256 else [2, 0]
            runner.kv_cache.fill_(float("nan"))
            ids, positions = runner.prepare_prefill([seq])
            context = get_context()
            slots = context.slot_mapping.long()
            logical = torch.arange(len(seq), device="cuda")
            pages = torch.tensor([2, 0], device="cuda")
            assert torch.equal(slots, pages[logical // 256] * 256 + logical % 256)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                logits = runner.run_model(ids, positions, True)
            peak, reserved = (
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
            )
            samples = oracle["prefill.kv.sample_positions"]
            saved = {
                "input_ids": ids.cpu(),
                "positions": positions.cpu(),
                "logits": logits.cpu(),
                "kv.sample_positions": samples,
            }
            for i in range(28):
                for kind, cache in (
                    ("keys", runner.kv_cache[0, i]),
                    ("values", runner.kv_cache[1, i]),
                ):
                    saved[f"kv.{i}.{kind}"] = cache.view(-1, 8, 128)[
                        slots[samples.cuda()]
                    ].cpu()
            torch.save(saved, directory / f"runner-{case}.pt")
            free, total = torch.cuda.mem_get_info()
            report["cases"][case] = dict(
                length=len(seq),
                pages=seq.block_table,
                slot_mapping=slots.cpu().tolist(),
                peak_allocated_bytes=peak,
                peak_reserved_bytes=reserved,
                free_bytes=free,
                total_bytes=total,
                unused_page_nan=bool(torch.isnan(runner.kv_cache[:, :, 1]).all()),
            )
            reset_context()
        report["loaded_keys"] = len(runner.weight_load_report.loaded_keys)
        report["skipped_keys"] = len(runner.weight_load_report.skipped_keys)
        report["status"] = "passed"
    except Exception:
        report["exception"] = traceback.format_exc()
        raise
    finally:
        model = getattr(runner, "model", None)
        report["bound_attention_layers"] = (
            sum(
                isinstance(m, Attention)
                and m.k_cache.numel() > 0
                and m.v_cache.numel() > 0
                for m in model.modules()
            )
            if model is not None
            else 0
        )
        report["kv_shape"] = (
            list(runner.kv_cache.shape) if hasattr(runner, "kv_cache") else None
        )
        report.update(
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )
        (directory / "native-report.json").write_text(json.dumps(report, indent=2))
        # Worker teardown only; production failure recovery remains Task8c.
        reset_context()
        torch.set_default_device("cpu")
        torch.set_default_dtype(previous_dtype)
        if dist.is_initialized():
            runner.exit()


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-worker", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    prefill_worker(args.prefill_worker)
