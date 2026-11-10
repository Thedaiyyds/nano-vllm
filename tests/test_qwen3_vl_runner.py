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
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextModel,
        Qwen3VLVisionModel,
    )
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
        config = Qwen3VLConfig(
            text_config=text,
            tie_word_embeddings=True,
            vision_config=dict(
                depth=3,
                hidden_size=32,
                intermediate_size=64,
                num_heads=4,
                out_hidden_size=256,
                num_position_embeddings=16,
                deepstack_visual_indexes=[0, 1, 2],
            ),
        )
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
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            vision = Qwen3VLVisionModel(config.vision_config)
        state.update(
            {
                "model.visual." + key: value.detach().clone()
                for key, value in vision.state_dict().items()
            }
        )
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
        self.assertTrue(hasattr(runner.model.model, "visual"))
        self.assertTrue(
            torch.equal(
                runner.model.model.visual.patch_embed.proj.weight,
                state["model.visual.patch_embed.proj.weight"].bfloat16(),
            )
        )
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
            set(state),
        )
        self.assertEqual(
            runner.weight_load_report.skipped_keys,
            [],
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
        self.assertEqual((report["loaded_keys"], report["skipped_keys"]), (625, 0))
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


DECODE_EVIDENCE = EVIDENCE.parent / "task8c"


class EngineCleanupTests(unittest.TestCase):
    def test_run_failure_aborts_active_request_and_preserves_waiting(self):
        from nanovllm.engine.llm_engine import LLMEngine
        from tests.test_qwen3_vl_scheduler import sequence, vl_scheduler

        class Boundary:
            fail = True

            def call(self, method, *args):
                if self.fail:
                    raise ValueError("injected recoverable runner failure")
                return [8]

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = vl_scheduler()
        engine.model_runner = Boundary()
        first, second = sequence(), sequence(255, max_tokens=1)
        engine.scheduler.add(first)
        engine.scheduler.add(second)
        with self.assertRaisesRegex(ValueError, "injected recoverable"):
            engine.step()
        self.assertTrue(first.is_finished)
        self.assertEqual(list(engine.scheduler.waiting), [second])
        self.assertFalse(engine.scheduler.block_manager.used_block_ids)
        engine.model_runner.fail = False
        outputs, _ = engine.step()
        self.assertEqual(outputs, [(second.seq_id, [8])])
        self.assertTrue(engine.scheduler.is_finished())
        self.assertTrue(
            all(b.ref_count == 0 for b in engine.scheduler.block_manager.blocks)
        )

    def test_schedule_failure_aborts_only_queue_head(self):
        from nanovllm.engine.llm_engine import LLMEngine
        from tests.test_qwen3_vl_scheduler import sequence, vl_scheduler

        engine = LLMEngine.__new__(LLMEngine)
        engine.scheduler = vl_scheduler(num_kvcache_blocks=2)
        first, second = sequence(), sequence(255, max_tokens=1)
        engine.scheduler.add(first)
        engine.scheduler.add(second)
        manager = engine.scheduler.block_manager
        outsider = sequence(1)
        manager.allocate(outsider, 0)
        with self.assertRaisesRegex(RuntimeError, "KV.*invariant"):
            engine.step()
        self.assertTrue(first.is_finished)
        self.assertEqual(list(engine.scheduler.waiting), [second])
        manager.deallocate(outsider)
        engine.scheduler.abort(second)
        self.assertTrue(all(b.ref_count == 0 for b in manager.blocks))

    def test_engine_exit_is_idempotent_and_cancels_queues(self):
        from nanovllm.engine.llm_engine import LLMEngine
        from tests.test_qwen3_vl_scheduler import sequence, vl_scheduler

        class Boundary:
            calls = 0

            def call(self, method):
                assert method == "exit"
                self.calls += 1

        engine = LLMEngine.__new__(LLMEngine)
        engine.ps = []
        engine.scheduler = vl_scheduler()
        boundary = engine.model_runner = Boundary()
        first, second = sequence(), sequence()
        engine.scheduler.add(first)
        engine.scheduler.add(second)
        engine.scheduler.schedule()
        engine.exit()
        engine.exit()
        self.assertEqual(boundary.calls, 1)
        self.assertTrue(first.is_finished and second.is_finished)
        self.assertTrue(engine.scheduler.is_finished())
        self.assertTrue(
            all(b.ref_count == 0 for b in engine.scheduler.block_manager.blocks)
        )


@pytest.mark.gpu
class TextDecodeRunnerTests(unittest.TestCase):
    def tearDown(self):
        import torch
        from nanovllm.utils.context import reset_context

        reset_context()
        torch.cuda.empty_cache()

    def test_decode_positions_and_noncontiguous_page_transition(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.engine.sequence import Sequence
        from nanovllm.utils.context import get_context

        runner = ModelRunner.__new__(ModelRunner)
        runner.config = prefill_config()
        runner.block_size = 256
        seq = Sequence([9707] * 255)
        for length, slot in [(256, 767), (257, 0), (258, 1)]:
            seq.append_token(7)
            seq.block_table = [2] if length == 256 else [2, 0]
            ids, positions = runner.prepare_decode([seq])
            self.assertEqual(tuple(positions.shape), (3, 1))
            self.assertEqual(positions.tolist(), [[length - 1]] * 3)
            self.assertEqual(ids.tolist(), [7])
            context = get_context()
            self.assertFalse(context.is_prefill)
            self.assertEqual(context.slot_mapping.tolist(), [slot])
            self.assertEqual(context.context_lens.tolist(), [length])
            self.assertEqual(context.block_tables.tolist(), [seq.block_table])
        with self.assertRaisesRegex(ValueError, "VL.*[Dd]ecode"):
            runner.prepare_decode([seq, seq])

    @pytest.mark.usefixtures("model_loading_rank")
    def test_run_always_resets_context_and_preserves_exception(self):
        import tempfile
        import torch
        from torch import nn
        from nanovllm.engine.sequence import Sequence
        from nanovllm.utils.context import get_context, reset_context, set_context

        class FailSample(nn.Module):
            def forward(self, *args):
                raise ValueError("injected sample")

        with tempfile.TemporaryDirectory() as directory:
            runner = tiny_cuda_runner(Path(directory))
        original_prepare, original_forward, original_sampler = (
            runner.prepare_prefill,
            runner.run_model,
            runner.sampler,
        )
        for stage in ("prepare", "forward", "sample"):
            with self.subTest(stage=stage):
                runner.prepare_prefill, runner.run_model, runner.sampler = (
                    original_prepare,
                    original_forward,
                    original_sampler,
                )
                seq = Sequence([1, 2])
                seq.num_scheduled_tokens = 2
                set_context(True, slot_mapping=torch.tensor([0], device="cuda"))

                def fail(*args):
                    raise ValueError(f"injected {stage}")

                if stage == "prepare":
                    runner.prepare_prefill = fail
                elif stage == "forward":
                    runner.run_model = fail
                else:
                    runner.sampler = FailSample()
                with self.assertRaisesRegex(ValueError, f"injected {stage}"):
                    runner.run([seq], True)
                self.assertFalse(get_context().is_prefill)
                self.assertIsNone(get_context().slot_mapping)
                reset_context()

    def test_init_failures_restore_defaults_and_double_exit(self):
        import json
        import tempfile
        from tests.support import run_worker

        DECODE_EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="lifecycle-", dir=DECODE_EVIDENCE))
        result = run_worker(
            "tests.test_qwen3_vl_runner", ["--lifecycle-worker", str(directory)]
        )
        (directory / "worker.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((directory / "lifecycle.json").read_text())
        for stage, row in report["failures"].items():
            with self.subTest(stage=stage):
                self.assertIn(f"injected {stage}", row["error"])
                self.assertEqual(row["dtype_after"], row["dtype_before"])
                self.assertEqual(row["device_after"], row["device_before"])
                self.assertFalse(row["group_alive"])
                self.assertFalse(row["model_alive"])
                self.assertFalse(row["context_present"])
                self.assertFalse(row["pool_alive"])
        self.assertIsNone(report["normal"]["second_exit_error"])
        self.assertFalse(report["normal"]["group_alive"])
        self.assertFalse(report["normal"]["model_alive"])
        self.assertFalse(report["normal"]["pool_alive"])
        self.assertTrue(report["engine_init"]["runner_ready"])
        self.assertIn("injected tokenizer", report["engine_init"]["error"])
        self.assertFalse(report["engine_init"]["group_alive"])
        self.assertFalse(report["engine_init"]["model_alive"])
        self.assertTrue(report["external_group_preserved"])

    @pytest.mark.checkpoint
    def test_target_32_decode_steps_and_greedy_engine(self):
        import json
        import tempfile
        import torch
        from tests.reference_qwen3_vl import assert_aligned, tensor_metrics
        from tests.support import run_worker
        from tests.test_qwen3_vl_text import CASES

        DECODE_EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="decode-", dir=DECODE_EVIDENCE))
        for mode in ("reference", "worker"):
            result = run_worker(
                "tests.test_qwen3_vl_runner", [f"--decode-{mode}", str(directory)]
            )
            (directory / f"{mode}.log").write_text(result.stdout + result.stderr)
            if mode == "worker":
                state = json.loads((directory / "native-decode.json").read_text())
                self.assertTrue(state.get("greedy"), result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        failures, metrics = [], {}
        for case in CASES:
            actual = torch.load(
                directory / f"native-{case}.pt", map_location="cpu", weights_only=True
            )
            expected = torch.load(
                directory / f"hf-{case}.pt", map_location="cpu", weights_only=True
            )
            if state["greedy"][case] != expected["teacher_ids"].tolist():
                failures.append(f"{case}: greedy token IDs differ")
            metrics[case] = {}
            self.assertEqual(actual["decode.logits"].shape, (32, 151936))
            self.assertEqual(actual.keys(), expected.keys())
            for name, value in actual.items():
                limits = {name: {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}}
                if value.is_floating_point():
                    metrics[case][name] = tensor_metrics(value, expected[name])
                try:
                    assert_aligned(name, value, expected[name], limits)
                except AssertionError as error:
                    failures.append(f"{case}: {error}")
        self.assertTrue(state["released_after_requests"])
        self.assertEqual(state["max_running"], 1)
        self.assertIsNone(state["second_exit_error"])
        self.assertFalse(state["group_alive_after_exit"])
        self.assertFalse(
            state["parameters_alive_after_exit"],
            state["parameters_alive_after_exit"][:8],
        )
        self.assertTrue(state["pool_released_after_exit"])
        identity = json.loads((directory / "hf-identity.json").read_text())
        self.assertEqual(identity["backend"], "flash_attention_2")
        self.assertEqual(identity["gpu"], state["gpu"])
        self.assertEqual(
            identity["allow_bf16_reduced_precision_reduction"],
            state["allow_bf16_reduced_precision_reduction"],
        )
        (directory / "comparison.json").write_text(
            json.dumps({"metrics": metrics, "failures": failures}, indent=2)
        )
        self.assertFalse(
            failures, f"{directory}: {failures[:8]} ({len(failures)} total)"
        )


def lifecycle_worker(directory):
    import gc
    import json
    import weakref
    import torch
    import torch.distributed as dist
    from unittest.mock import patch
    import nanovllm.engine.llm_engine as engine_module
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.utils.context import get_context, reset_context, set_context

    tiny = directory / "tiny"
    tiny.mkdir()
    config, _ = tiny_model_checkpoint(tiny, multimodal=True)
    report = {"failures": {}}

    class Injected(ModelRunner):
        stage = ""

        def _load_model(self):
            super()._load_model()
            self.model_ref = weakref.ref(self.model)
            if self.stage == "load":
                raise ValueError("injected load")

        def warmup_model(self):
            if self.stage == "warmup":
                set_context(True, slot_mapping=torch.tensor([0], device="cuda"))
                raise ValueError("injected warmup")
            return super().warmup_model()

        def allocate_kv_cache(self):
            super().allocate_kv_cache()
            self.pool_ref = weakref.ref(self.kv_cache)
            if self.stage == "allocate":
                raise ValueError("injected allocate")

    def measured(runner):
        gc.collect()
        return dict(
            dtype_after=str(torch.get_default_dtype()),
            device_after=str(torch.get_default_device()),
            group_alive=dist.is_initialized(),
            model_alive=runner.model_ref() is not None,
            pool_alive=hasattr(runner, "pool_ref") and runner.pool_ref() is not None,
            context_present=get_context().slot_mapping is not None,
        )

    def fallback_cleanup(runner):
        # After measurement only, to let RED collect every failure safely.
        reset_context()
        for name in ("model", "kv_cache", "sampler"):
            runner.__dict__.pop(name, None)
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)
        gc.collect()
        torch.cuda.empty_cache()

    for stage in ("load", "warmup", "allocate"):
        runner = Injected.__new__(Injected)
        runner.stage = stage
        torch.set_default_dtype(torch.float64)
        before = dict(
            dtype_before=str(torch.get_default_dtype()),
            device_before=str(torch.get_default_device()),
        )
        try:
            runner.__init__(config, 0, [])
        except ValueError as error:
            report["failures"][stage] = dict(
                before, error=str(error), **measured(runner)
            )
        finally:
            fallback_cleanup(runner)
    runner = Injected(config, 0, [])
    runner.exit()
    second_exit_error = None
    try:
        runner.exit()
    except Exception as error:
        second_exit_error = repr(error)
    report["normal"] = dict(second_exit_error=second_exit_error, **measured(runner))
    fallback_cleanup(runner)
    tracked = {}

    class Tracked(ModelRunner):
        def __init__(self, *args):
            super().__init__(*args)
            tracked.update(
                runner_ready=True,
                model=weakref.ref(self.model),
                runner=weakref.ref(self),
            )

    # Inject only the tokenizer boundary after a real Runner has initialized.
    with (
        patch.object(engine_module, "ModelRunner", Tracked),
        patch.object(
            engine_module.AutoTokenizer,
            "from_pretrained",
            side_effect=ValueError("injected tokenizer"),
        ),
    ):
        try:
            engine_module.LLMEngine(
                str(tiny), max_num_seqs=1, enforce_eager=True, max_model_len=64
            )
        except Exception as error:
            gc.collect()
            report["engine_init"] = dict(
                runner_ready=tracked.get("runner_ready", False),
                error=repr(error),
                group_alive=dist.is_initialized(),
                model_alive=tracked["model"]() is not None,
            )
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        "gloo",
        init_method=(directory / "external-rank").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        try:
            ModelRunner(config, 0, [])
        except ValueError:
            report["external_group_preserved"] = (
                dist.is_initialized() and dist.get_backend() == "gloo"
            )
    finally:
        dist.destroy_process_group()
    (directory / "lifecycle.json").write_text(json.dumps(report, indent=2))


def cuda_memory_record():
    import torch

    free, total = torch.cuda.mem_get_info()
    return dict(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        allocated_bytes=torch.cuda.memory_allocated(),
        free_bytes=free,
        total_bytes=total,
    )


def decode_reference(directory):
    import json
    import torch
    from transformers import Qwen3VLForConditionalGeneration as HFModel
    from tests.reference_qwen3_vl import (
        provenance,
        last_step_cache,
        stack_step_cache,
        sample_cache,
    )
    from tests.test_qwen3_vl_text import CASES, load_oracle

    identity = provenance(MODEL)
    model = (
        HFModel.from_pretrained(
            str(MODEL),
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    identity.update(
        backend=model.model.language_model.config._attn_implementation,
        gpu=torch.cuda.get_device_name(),
        dtype=str(next(model.parameters()).dtype),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cuda=torch.version.cuda,
        cases={},
    )
    for case in CASES:
        oracle = load_oracle(case)
        ids = oracle["input.input_ids"].cuda()
        length = ids.shape[1]
        positions = torch.arange(length, device="cuda").expand(3, 1, -1)
        saved, teacher, logits = {}, [], []
        model.model.rope_deltas = None
        with torch.inference_mode():
            result = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                position_ids=positions,
                use_cache=True,
                logits_to_keep=1,
            )
            saved["prefill.logits"] = result.logits[:, -1].cpu()
            cache = result.past_key_values
            next_token = int(result.logits[0, -1].argmax())
            for step in range(32):
                teacher.append(next_token)
                pos = torch.full(
                    (3, 1, 1), length + step, device="cuda", dtype=torch.long
                )
                result = model(
                    input_ids=torch.tensor([[next_token]], device="cuda"),
                    attention_mask=torch.ones(
                        (1, length + step + 1), device="cuda", dtype=torch.long
                    ),
                    position_ids=pos,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = result.past_key_values
                logits.append(result.logits[0, -1].cpu())
                last_step_cache(cache, saved, "decode.step_kv")
                next_token = int(result.logits[0, -1].argmax())
            sample_cache(cache, saved, "decode.final_kv")
        stack_step_cache(saved)
        saved["teacher_ids"] = torch.tensor(teacher)
        saved["decode.logits"] = torch.stack(logits)
        torch.save(saved, directory / f"hf-{case}.pt")
        identity["cases"][case] = cuda_memory_record()
    (directory / "hf-identity.json").write_text(json.dumps(identity, indent=2))


def decode_worker(directory):
    import gc
    import json
    import traceback
    import weakref
    import torch
    import torch.distributed as dist
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    from nanovllm.utils.context import get_context, reset_context
    from tests.test_qwen3_vl_text import CASES, load_oracle

    engine = None
    report = {"greedy": {}, "contexts": {}, "cases": {}}

    def save_last(runner, seq, saved):
        slot = seq.block_table[(len(seq) - 1) // 256] * 256 + (len(seq) - 1) % 256
        for i in range(28):
            for kind, cache in (
                ("keys", runner.kv_cache[0, i]),
                ("values", runner.kv_cache[1, i]),
            ):
                saved.setdefault(f"decode.step_kv.{i}.{kind}", []).append(
                    cache.view(-1, 8, 128)[slot].cpu()
                )

    try:
        engine = LLMEngine(str(MODEL), max_num_seqs=1, enforce_eager=True)
        runner = engine.model_runner
        prompts = []
        for case in CASES:
            ids = load_oracle(case)["input.input_ids"][0].tolist()
            prompts.append(ids)
            reference = torch.load(
                directory / f"hf-{case}.pt", map_location="cpu", weights_only=True
            )
            seq = Sequence(
                ids, SamplingParams(temperature=0, max_tokens=33, ignore_eos=True)
            )
            seq.block_table = [2] if len(seq) <= 256 else [2, 0]
            seq.num_scheduled_tokens = len(seq)
            runner.kv_cache.fill_(float("nan"))
            inputs, positions = runner.prepare_prefill([seq])
            with torch.inference_mode():
                saved = {
                    "prefill.logits": runner.run_model(inputs, positions, True).cpu(),
                }
            logits, contexts, consumed = [], [], []
            for token in reference["teacher_ids"].tolist():
                seq.num_cached_tokens = len(seq)
                seq.append_token(token)
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                seq.block_table = [2] if len(seq) <= 256 else [2, 0]
                inputs, positions = runner.prepare_decode([seq])
                consumed.append(int(inputs.item()))
                ctx = get_context()
                contexts.append(
                    dict(
                        positions=positions.cpu().tolist(),
                        length=ctx.context_lens.tolist(),
                        slots=ctx.slot_mapping.tolist(),
                        blocks=ctx.block_tables.tolist(),
                    )
                )
                with torch.inference_mode():
                    logits.append(runner.run_model(inputs, positions, False)[0].cpu())
                save_last(runner, seq, saved)
            saved["decode.logits"] = torch.stack(logits)
            saved["teacher_ids"] = torch.tensor(consumed)
            for key in list(saved):
                if ".step_kv." in key:
                    saved[key] = torch.stack(saved[key])
            sample = reference["decode.final_kv.sample_positions"]
            saved["decode.final_kv.sample_positions"] = sample
            logical = sample.cuda()
            pages = torch.tensor([2, 0], device="cuda")
            slots = pages[logical // 256] * 256 + logical % 256
            for i in range(28):
                for kind in ("keys", "values"):
                    cache = runner.kv_cache[0 if kind == "keys" else 1, i]
                    saved[f"decode.final_kv.{i}.{kind}"] = cache.view(-1, 8, 128)[
                        slots
                    ].cpu()
            del cache
            torch.save(saved, directory / f"native-{case}.pt")
            report["contexts"][case] = contexts
            report["cases"][case] = cuda_memory_record()
            reset_context()
        runner.kv_cache.fill_(float("nan"))
        max_running = []
        original_run = runner.run

        def observe(seqs, prefill):
            max_running.append(len(engine.scheduler.running))
            assert all(not seq.block_table for seq in engine.scheduler.waiting)
            return original_run(seqs, prefill)

        runner.run = observe
        outputs = engine.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
            use_tqdm=False,
        )
        report["greedy"] = {
            case: value["token_ids"] for case, value in zip(CASES, outputs)
        }
        report["max_running"] = max(max_running)
        manager = engine.scheduler.block_manager
        report["released_after_requests"] = (
            engine.is_finished()
            and not manager.used_block_ids
            and all(b.ref_count == 0 for b in manager.blocks)
            and not manager.hash_to_block_id
        )
        single = engine.generate(
            [prompts[0]],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        assert single[0]["token_ids"] == report["greedy"][CASES[0]][:1]
        report["before_exit_memory"] = cuda_memory_record()
        report["dtype"] = str(next(runner.model.parameters()).dtype)
        parameter_refs = [
            (name, weakref.ref(parameter))
            for name, parameter in runner.model.named_parameters()
        ]
        pool_ref = weakref.ref(runner.kv_cache)
        gc_was_enabled = gc.isenabled()
        gc.disable()  # Exit must not depend on a later automatic GC threshold.
        try:
            engine.exit()
            report["parameters_alive_after_exit"] = [
                name for name, reference in parameter_refs if reference() is not None
            ]
            report["pool_released_after_exit"] = pool_ref() is None
            report["after_exit_memory"] = cuda_memory_record()
            gc.collect()  # Diagnostic only, strictly after the asserted snapshot.
            report["after_diagnostic_gc_memory"] = cuda_memory_record()
        finally:
            if gc_was_enabled:
                gc.enable()
        second_error = None
        try:
            engine.exit()
        except Exception as error:
            second_error = repr(error)
        report.update(
            second_exit_error=second_error, group_alive_after_exit=dist.is_initialized()
        )
    except Exception:
        report["exception"] = traceback.format_exc()
        raise
    finally:
        report.update(
            gpu=torch.cuda.get_device_name(),
            allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )
        (directory / "native-decode.json").write_text(json.dumps(report, indent=2))
        reset_context()
        if dist.is_initialized() and engine is not None:
            engine.exit()


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prefill-worker", type=Path)
    mode.add_argument("--lifecycle-worker", type=Path)
    mode.add_argument("--decode-reference", type=Path)
    mode.add_argument("--decode-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.prefill_worker:
        prefill_worker(args.prefill_worker)
    elif args.lifecycle_worker:
        lifecycle_worker(args.lifecycle_worker)
    elif args.decode_reference:
        decode_reference(args.decode_reference)
    else:
        decode_worker(args.decode_worker)
