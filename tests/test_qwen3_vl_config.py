"""Config normalization and the approved first-stage VL limits (CPU only)."""

import json
import os
from pathlib import Path
import tempfile
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
VL_OPTIONS = {"max_num_seqs": 1, "enforce_eager": True}


class ConfigTests(unittest.TestCase):
    @pytest.mark.checkpoint
    def test_nested_config_and_limits(self):
        import torch
        from nanovllm.config import Config

        # Reading the real nested config must work without loading any weights.
        default_dtype = torch.get_default_dtype()
        config = Config(str(MODEL), **VL_OPTIONS)
        self.assertIsNone(config.hf_config.dtype)
        self.assertIs(config.text_config, config.hf_config.text_config)
        self.assertEqual(config.text_config.hidden_size, 2048)
        self.assertEqual(config.model_dtype, torch.bfloat16)
        self.assertEqual(config.eos_token_ids, frozenset({151645, 151643}))
        self.assertTrue(config.is_multimodal)
        self.assertEqual(config.text_config.rope_parameters["rope_theta"], 5000000)
        self.assertEqual(
            config.text_config.rope_parameters["mrope_section"], [24, 20, 20]
        )
        self.assertEqual(config.max_model_len, 4096)
        self.assertEqual(config.max_visual_tokens, 256)
        self.assertEqual(config.kv_cache_reserve_bytes, 1073741824)
        self.assertEqual(torch.get_default_dtype(), default_dtype)

    def test_original_qwen3_keeps_legacy_limits_and_scalar_eos(self):
        import torch
        from nanovllm.config import Config

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3",
                        "max_position_embeddings": 8192,
                        "dtype": "float16",
                        "eos_token_id": 7,
                    }
                )
            )
            config = Config(
                directory,
                max_model_len=9000,
                tensor_parallel_size=2,
                max_visual_tokens=1024,
                kv_cache_reserve_bytes=0,
            )
            self.assertIs(config.text_config, config.hf_config)
            self.assertFalse(config.is_multimodal)
            self.assertEqual(config.model_dtype, torch.float16)
            self.assertEqual(config.eos_token_ids, frozenset({7}))
            self.assertEqual(
                config.eos, -1
            )  # Still owned by the legacy tokenizer path.
            self.assertEqual(config.max_model_len, 8192)
            self.assertEqual(config.max_num_seqs, 512)
            self.assertFalse(config.enforce_eager)


@pytest.fixture
def vl_config_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "dtype": "bfloat16",
                    "max_position_embeddings": 2048,
                    "eos_token_id": 9,
                },
            }
        )
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [9, 10]})
    )
    return tmp_path


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("tensor_parallel_size", 0),
        ("enforce_eager", False),
        ("max_num_seqs", 2),
        ("max_num_seqs", 0),
        ("max_model_len", 0),
        ("max_model_len", -1),
        ("max_model_len", 4097),
        ("max_num_batched_tokens", 0),
        ("max_num_batched_tokens", -1),
        ("max_visual_tokens", 0),
        ("max_visual_tokens", -1),
        ("max_visual_tokens", 257),
        ("kv_cache_reserve_bytes", 0),
        ("kv_cache_reserve_bytes", -1),
    ],
)
def test_vl_rejects_unapproved_limits(vl_config_dir, field, value):
    from nanovllm.config import Config

    options = VL_OPTIONS | {field: value}
    with pytest.raises(ValueError) as error:
        Config(str(vl_config_dir), **options)
    assert field in str(error.value)
    assert str(value) in str(error.value)


def test_vl_lower_limits_and_checkpoint_cap(vl_config_dir):
    from nanovllm.config import Config

    config = Config(str(vl_config_dir), **VL_OPTIONS)
    assert config.max_model_len == 2048
    assert config.eos_token_ids == frozenset({9, 10})  # GenerationConfig wins.
    config = Config(
        str(vl_config_dir),
        **VL_OPTIONS,
        max_model_len=1024,
        max_visual_tokens=64,
        kv_cache_reserve_bytes=512 * 1024**2,
    )
    assert config.max_model_len == 1024
    assert config.max_visual_tokens == 64
    assert config.kv_cache_reserve_bytes == 536870912


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_vl_requires_bfloat16(vl_config_dir, dtype):
    from nanovllm.config import Config

    path = vl_config_dir / "config.json"
    raw = json.loads(path.read_text())
    raw["text_config"]["dtype"] = dtype
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="model_dtype") as error:
        Config(str(vl_config_dir), **VL_OPTIONS)
    assert dtype in str(error.value)


def test_unknown_model_type_is_rejected(tmp_path):
    from nanovllm.config import Config

    # llama is known to AutoConfig but unsupported by nano-vLLM.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="model_type.*llama"):
        Config(str(tmp_path))
