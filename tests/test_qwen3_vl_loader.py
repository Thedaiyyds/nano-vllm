"""Strict VL checkpoint audit and small real CPU parameter loads."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import pytest

pytestmark = pytest.mark.hf_reference
LANGUAGE = "model.language_model."
ATTENTION = LANGUAGE + "layers.0.self_attn."
MLP = LANGUAGE + "layers.0.mlp."


@pytest.fixture(scope="module", autouse=True)
def single_rank(tmp_path_factory):
    import torch.distributed as dist

    if dist.is_initialized():
        assert dist.get_rank() == 0 and dist.get_world_size() == 1
        yield
        return
    rendezvous = tmp_path_factory.mktemp("vl-loader-rank") / "init"
    previous = os.environ.get("GLOO_SOCKET_IFNAME")
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    try:
        dist.init_process_group(
            "gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1
        )
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if previous is None:
            os.environ.pop("GLOO_SOCKET_IFNAME", None)
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = previous


def tiny_model():
    import torch
    from torch import nn
    from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear

    root = nn.Module()
    root.model = nn.Module()
    text = root.model.language_model = nn.Module()
    text.embed_tokens = nn.Embedding(8, 4)
    text.norm = nn.RMSNorm(4)
    layer = nn.Module()
    text.layers = nn.ModuleList([layer])
    layer.self_attn = nn.Module()
    layer.self_attn.qkv_proj = QKVParallelLinear(4, 2, 2, 1)
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = MergedColumnParallelLinear(4, [6, 6])
    root.lm_head = nn.Linear(4, 8, bias=False)
    root.lm_head.weight = text.embed_tokens.weight
    root.model.visual = nn.Module()
    root.model.visual.attn = nn.Module()
    root.model.visual.attn.qkv = nn.Linear(4, 12)
    # A projection name in the visual namespace must remain a direct parameter.
    root.model.visual.q_proj = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        for parameter in root.parameters():
            parameter.fill_(-99)
    return root


def tiny_checkpoint():
    import torch

    return {
        ATTENTION + "q_proj.weight": torch.full((4, 4), 1.0),
        ATTENTION + "k_proj.weight": torch.full((2, 4), 2.0),
        ATTENTION + "v_proj.weight": torch.full((2, 4), 3.0),
        MLP + "gate_proj.weight": torch.full((6, 4), 4.0),
        MLP + "up_proj.weight": torch.full((6, 4), 5.0),
        LANGUAGE + "embed_tokens.weight": torch.arange(32).float().reshape(8, 4),
        LANGUAGE + "norm.weight": torch.full((4,), 6.0),
        "model.visual.attn.qkv.weight": torch.full((12, 4), 7.0),
        "model.visual.attn.qkv.bias": torch.full((12,), 8.0),
        "model.visual.q_proj.weight": torch.full((4, 4), 9.0),
    }


class LoaderTests(unittest.TestCase):
    def test_packed_shards_and_aliases(self):
        import torch
        from safetensors.torch import save_file
        from nanovllm.utils.loader import load_qwen3_vl_weights

        model = tiny_model()
        state = tiny_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            missing = state.copy()
            del missing[ATTENTION + "k_proj.weight"]
            save_file(missing, str(Path(directory) / "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "missing.*k_proj"):
                load_qwen3_vl_weights(model, directory, text_only=True)
            self.assertTrue(all(torch.all(p == -99) for p in model.parameters()))
            save_file(state, str(Path(directory) / "model.safetensors"))
            report = load_qwen3_vl_weights(model, directory)
        text = model.model.language_model
        self.assertIs(model.lm_head.weight, text.embed_tokens.weight)
        self.assertEqual(report.tied_aliases, ["lm_head.weight"])
        self.assertEqual(set(report.loaded_keys), set(state))
        self.assertEqual(len(report.loaded_keys), 10)
        self.assertEqual(report.skipped_keys, [])
        self.assertTrue(
            torch.equal(
                text.layers[0].self_attn.qkv_proj.weight,
                torch.cat(
                    [
                        state[ATTENTION + name + ".weight"]
                        for name in ("q_proj", "k_proj", "v_proj")
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                text.layers[0].mlp.gate_up_proj.weight,
                torch.cat(
                    [state[MLP + name + ".weight"] for name in ("gate_proj", "up_proj")]
                ),
            )
        )
        for name in (
            LANGUAGE + "embed_tokens.weight",
            LANGUAGE + "norm.weight",
            "model.visual.attn.qkv.weight",
            "model.visual.attn.qkv.bias",
            "model.visual.q_proj.weight",
        ):
            self.assertTrue(torch.equal(model.get_parameter(name), state[name]), name)


@pytest.mark.parametrize(
    "failure", ["duplicate", "shape", "unexpected", "missing_norm", "native_packed"]
)
def test_invalid_checkpoint_is_rejected_before_copy(tmp_path, failure):
    import torch
    from safetensors.torch import save_file
    from nanovllm.utils.loader import load_qwen3_vl_weights

    state = tiny_checkpoint()
    expected = failure
    if failure == "duplicate":
        save_file(
            {ATTENTION + "q_proj.weight": state[ATTENTION + "q_proj.weight"]},
            str(tmp_path / "extra.safetensors"),
        )
        expected = "duplicate.*shard"
    elif failure == "shape":
        state[ATTENTION + "k_proj.weight"] = torch.ones(1, 4)  # copy_ would broadcast.
    elif failure == "unexpected":
        state["model.other.weight"] = torch.ones(1)
    elif failure == "missing_norm":
        del state[LANGUAGE + "norm.weight"]
        expected = "missing.*norm"
    else:
        state[ATTENTION + "qkv_proj.weight"] = torch.ones(8, 4)
        expected = "unexpected"
    save_file(state, str(tmp_path / "model.safetensors"))
    model = tiny_model()
    with pytest.raises(ValueError, match=expected):
        load_qwen3_vl_weights(model, str(tmp_path))
    assert all(torch.all(p == -99) for p in model.parameters())


def test_text_only_skips_only_visual_namespace(tmp_path):
    import torch
    from safetensors.torch import save_file
    from nanovllm.utils.loader import load_qwen3_vl_weights

    state = tiny_checkpoint()
    save_file(state, str(tmp_path / "model.safetensors"))
    model = tiny_model()
    report = load_qwen3_vl_weights(model, str(tmp_path), text_only=True)
    assert len(report.skipped_keys) == 3
    assert all(key.startswith("model.visual.") for key in report.skipped_keys)
    assert set(report.loaded_keys) == set(state) - set(report.skipped_keys)
    assert torch.equal(
        model.model.language_model.embed_tokens.weight,
        state[LANGUAGE + "embed_tokens.weight"],
    )
    assert torch.all(model.model.visual.attn.qkv.weight == -99)
    state["model.other.weight"] = torch.ones(1)
    save_file(state, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="unexpected.*model.other"):
        load_qwen3_vl_weights(model, str(tmp_path), text_only=True)


def test_tie_requires_same_parameter_not_just_same_storage(tmp_path):
    from torch import nn
    from safetensors.torch import save_file
    from nanovllm.utils.loader import load_qwen3_vl_weights

    model = tiny_model()
    model.lm_head.weight = nn.Parameter(
        model.model.language_model.embed_tokens.weight.data
    )
    save_file(tiny_checkpoint(), str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="tied.*Parameter"):
        load_qwen3_vl_weights(model, str(tmp_path))


def test_two_model_parameters_cannot_claim_the_same_source_shard(tmp_path):
    from torch import nn
    from safetensors.torch import save_file
    from nanovllm.utils.loader import load_qwen3_vl_weights

    model = tiny_model()
    model.model.language_model.layers[0].self_attn.q_proj = nn.Linear(4, 4, bias=False)
    save_file(tiny_checkpoint(), str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="duplicate.*mapping.*q_proj"):
        load_qwen3_vl_weights(model, str(tmp_path))


def test_public_load_rejects_meta_but_header_audit_accepts_it(tmp_path):
    import torch
    from safetensors.torch import save_file
    from nanovllm.utils.loader import load_qwen3_vl_weights, _audit_qwen3_vl_weights

    with torch.device("meta"):
        model = tiny_model()
    save_file(tiny_checkpoint(), str(tmp_path / "model.safetensors"))
    sources, _, report = _audit_qwen3_vl_weights(model, str(tmp_path))
    assert len(sources) == 10
    assert report.loaded_keys == []
    with pytest.raises(ValueError, match="meta parameter"):
        load_qwen3_vl_weights(model, str(tmp_path))
    assert all(p.is_meta for p in model.parameters())


def test_streams_one_tensor_at_a_time_and_keeps_legacy_loader(tmp_path):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from nanovllm.utils.loader import load_qwen3_vl_weights, load_model

    state = tiny_checkpoint()
    save_file(state, str(tmp_path / "model.safetensors"))
    live = []

    class Reader:
        def __init__(self, *args):
            self.reader = safe_open(*args)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_slice(self, name):
            return self.reader.get_slice(name)

        def get_tensor(self, name):
            assert not any(ref() is not None for ref in live), (
                "previous source tensor retained"
            )
            tensor = self.reader.get_tensor(name)
            live.append(weakref.ref(tensor))
            return tensor

    model = tiny_model()
    with patch("nanovllm.utils.loader.safe_open", Reader):
        report = load_qwen3_vl_weights(model, str(tmp_path))
    assert len(live) == len(report.loaded_keys) == 10
    assert all(ref() is None for ref in live)
    assert torch.equal(
        model.model.language_model.embed_tokens.weight,
        state[LANGUAGE + "embed_tokens.weight"],
    )
    # Exercise the unchanged old loader with a real existing packed layer.
    legacy = torch.nn.Module()
    legacy.qkv_proj = tiny_model().model.language_model.layers[0].self_attn.qkv_proj
    legacy.packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    save_file(
        {
            name + ".weight": state[ATTENTION + name + ".weight"]
            for name in ("q_proj", "k_proj", "v_proj")
        },
        str(legacy_dir / "model.safetensors"),
    )
    load_model(legacy, str(legacy_dir))
    assert torch.equal(
        legacy.qkv_proj.weight,
        model.model.language_model.layers[0].self_attn.qkv_proj.weight,
    )


@pytest.mark.checkpoint
def test_real_625_headers_have_complete_ownership_without_tensor_reads(request):
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )
    from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear
    from nanovllm.utils.loader import _audit_qwen3_vl_weights
    from tests.reference_qwen3_vl import DEFAULT_MODEL

    path = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    with torch.device("meta"):
        # HF supplies an independent parameter-name/shape schema, no model weights.
        model = Qwen3VLForConditionalGeneration(config)
        text = config.text_config
        for layer in model.model.language_model.layers:
            attention = layer.self_attn
            attention.qkv_proj = QKVParallelLinear(
                text.hidden_size,
                text.head_dim,
                text.num_attention_heads,
                text.num_key_value_heads,
            )
            del attention.q_proj, attention.k_proj, attention.v_proj
            layer.mlp.gate_up_proj = MergedColumnParallelLinear(
                text.hidden_size, [text.intermediate_size] * 2
            )
            del layer.mlp.gate_proj, layer.mlp.up_proj
    assert all(p.is_meta for p in model.parameters())
    # Header audit must never read a tensor payload.
    from safetensors import safe_open

    class HeaderReader:
        def __init__(self, *args):
            self.reader = safe_open(*args)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_slice(self, name):
            return self.reader.get_slice(name)

        def get_tensor(self, name):
            raise AssertionError("header audit attempted a payload read")

    with patch("nanovllm.utils.loader.safe_open", HeaderReader):
        sources, targets, report = _audit_qwen3_vl_weights(model, path, text_only=False)
        text_sources, _, text_report = _audit_qwen3_vl_weights(
            model, path, text_only=True
        )
    assert len(sources) == len(targets) == 625
    assert set(sources) == set(targets)
    assert len(text_sources) + len(text_report.skipped_keys) == 625
    assert len(text_sources) == 310
    assert len(text_report.skipped_keys) == 315
    assert all(key.startswith(LANGUAGE) for key in text_sources)
    assert all(key.startswith("model.visual.") for key in text_report.skipped_keys)
    assert report.loaded_keys == []  # Audited, not actually loaded.
    assert report.tied_aliases == ["lm_head.weight"]
    request.node.user_properties.extend(
        [
            ("checkpoint_keys", len(sources)),
            ("language_keys", len(text_sources)),
            ("visual_keys", len(text_report.skipped_keys)),
            (
                "native_parameter_targets",
                len({target.name for target in targets.values()}),
            ),
            ("tied_aliases", ",".join(report.tied_aliases)),
            ("payload_reads", 0),
            ("schema_device", "meta"),
        ]
    )
