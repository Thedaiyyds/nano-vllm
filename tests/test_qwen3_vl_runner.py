"""Task 8a CPU budget and model loading checks; no forward/Runner GPU claim."""

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
