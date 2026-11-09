"""Task11a: real block-0 weights, pinned HF SDPA and literal segment tests."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pytest

# Match the existing heavy vision CPU tests and the lean CI selection contract.
pytestmark = pytest.mark.hf_reference

EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11a"
)
# Predeclared same-backend component gate, not the proposed BF16 corpus envelope.
EXACT = {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}
CASES = (
    "image_square",
    "image_rect",
    "image_limit",
    "image_same_layout_other_pixels",
    "image_candy",
    "image_cat",
    "two_images",
    "two_images_changed_second",
    "two_time_groups",
    "two_time_groups_changed_second",
)


def tiny_attention():
    import torch
    from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionAttention

    module = Qwen3VLVisionAttention(SimpleNamespace(hidden_size=4, num_heads=1))
    with torch.no_grad():
        module.qkv.weight.zero_()
        module.qkv.bias.zero_()
        module.qkv.weight[8:] = torch.eye(4)
        module.proj.weight.copy_(torch.eye(4))
        module.proj.bias.zero_()
    return module


class VisionBlockCPUTests(TestCase):
    def test_noncausal_attention_uses_future_values_within_each_segment(self):
        import torch

        module = tiny_attention()
        values = torch.tensor([[0.0] * 4, [2.0] * 4, [7.0] * 4])
        positions = (torch.ones(3, 4), torch.zeros(3, 4))
        # Q=K=0 -> uniform attention; first segment's mean is exactly1.
        result = module(values, torch.tensor([0, 2, 3], dtype=torch.int32), positions)
        torch.testing.assert_close(
            result, torch.tensor([[1.0] * 4, [1.0] * 4, [7.0] * 4]), rtol=0, atol=0
        )
        changed = values.clone()
        changed[1] = 6
        after = module(changed, torch.tensor([0, 2, 3]), positions)
        self.assertEqual(after[0].tolist(), [3.0] * 4)
        torch.testing.assert_close(result[2], after[2], rtol=0, atol=0)
        self.assertFalse(hasattr(module, "k_cache"))
        self.assertFalse(hasattr(module, "v_cache"))

    def test_biases_and_both_residual_additions(self):
        import torch
        from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionBlock

        cfg = SimpleNamespace(
            hidden_size=4,
            num_heads=1,
            intermediate_size=8,
            hidden_act="gelu_pytorch_tanh",
        )
        block = Qwen3VLVisionBlock(cfg)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
            block.attn.proj.bias.fill_(1)
            block.mlp.linear_fc2.bias.fill_(2)
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-4.0, -3.0, -2.0, -1.0]])
        result = block(
            values, torch.tensor([0, 2]), (torch.ones(2, 4), torch.zeros(2, 4))
        )
        torch.testing.assert_close(result, values + 3, rtol=0, atol=0)

    def test_gelu_tanh_mlp_matches_independent_scalar_values(self):
        import math
        import torch
        from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionMLP

        mlp = Qwen3VLVisionMLP(
            SimpleNamespace(
                hidden_size=4, intermediate_size=4, hidden_act="gelu_pytorch_tanh"
            )
        ).double()
        with torch.no_grad():
            for layer in (mlp.linear_fc1, mlp.linear_fc2):
                layer.weight.copy_(torch.eye(4))
                layer.bias.zero_()
        x = [-2.0, -0.5, 0.5, 2.0]
        expected = [
            0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3)))
            for v in x
        ]
        torch.testing.assert_close(
            mlp(torch.tensor([x], dtype=torch.float64)),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-14,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    "boundaries",
    [
        [1, 3],
        [0, 2],
        [0, 4],
        [0, 0, 3],
        [0, 2, 1, 3],
        [0],
        [],
    ],
)
def test_rejects_boundaries_that_drop_repeat_or_mix_patches(boundaries):
    import torch

    with pytest.raises(ValueError):
        tiny_attention()(
            torch.ones(3, 4),
            torch.tensor(boundaries, dtype=torch.int32),
            (torch.ones(3, 4), torch.zeros(3, 4)),
        )


@pytest.mark.gpu
@pytest.mark.hf_reference
@pytest.mark.checkpoint
class VisionBlockTests(TestCase):
    def test_block_zero_rotated_qk_and_segment_isolation(self):
        from tests.support import run_worker

        EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="gpu-", dir=EVIDENCE))
        # Full reference process exits before the native GPU process starts.
        for backend in ("hf", "native"):
            result = run_worker(
                "tests.test_qwen3_vl_vision_block",
                ["--worker", backend, "--output", str(directory)],
            )
            (directory / f"{backend}.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((directory / "native-report.json").read_text())
        self.assertEqual(set(report["cases"]), set(CASES))
        self.assertTrue(report["parameters_released"])
        self.assertTrue(report["image_isolation_exact"])
        self.assertTrue(report["time_isolation_exact"])


def worker(backend, directory):
    import gc
    import weakref
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    from transformers.vision_utils import get_vision_position_ids
    from nanovllm.models import qwen3_vl_vision as native
    from tests.reference_qwen3_vl import (
        DEFAULT_MODEL,
        provenance,
        assert_aligned,
        tensor_digest,
    )
    from tests.support import digest_file
    from tests.test_qwen3_vl_text import load_oracle
    from tests.test_qwen3_vl_vision_positions import hf_absolute

    model = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    identity = provenance(model)
    cfg = AutoConfig.from_pretrained(model, local_files_only=True).vision_config
    cfg._attn_implementation = "sdpa"
    implementation = hf if backend == "hf" else native
    block = implementation.Qwen3VLVisionBlock(cfg).to(dtype=torch.bfloat16)
    embed = implementation.Qwen3VLVisionPatchEmbed(cfg).to(dtype=torch.bfloat16)
    block_prefix = "model.visual.blocks.0."
    with safe_open(
        str(Path(model) / "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        keys = sorted(k for k in f.keys() if k.startswith(block_prefix))
        # Exact schema and strict loading: no initialized weights survive.
        assert set(k.removeprefix(block_prefix) for k in keys) == set(
            block.state_dict()
        )
        block.load_state_dict(
            {k.removeprefix(block_prefix): f.get_tensor(k) for k in keys}, strict=True
        )
        patch_keys = [
            "model.visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.bias",
        ]
        embed.load_state_dict(
            {
                k.removeprefix("model.visual.patch_embed."): f.get_tensor(k)
                for k in patch_keys
            },
            strict=True,
        )
        table = f.get_tensor("model.visual.pos_embed.weight").cuda()
    block, embed = block.cuda().eval(), embed.cuda().eval()
    rope = implementation.Qwen3VLVisionRotaryEmbedding(cfg).cuda()
    captures = {}
    hooks = []
    for name, module in (
        ("norm1", block.norm1),
        ("attention", block.attn),
        ("norm2", block.norm2),
        ("mlp", block.mlp),
        ("qkv", block.attn.qkv),
    ):
        hooks.append(
            module.register_forward_hook(
                lambda m, a, out, name=name: captures.__setitem__(
                    name, out.detach().cpu().clone()
                )
            )
        )
    rotate = implementation.apply_rotary_pos_emb_vision

    def capture_rotation(q, k, cos, sin):
        result = rotate(q, k, cos, sin)
        captures["rotated_q"] = result[0].detach().cpu().clone()
        captures["rotated_k"] = result[1].detach().cpu().clone()
        return result

    identity.update(
        backend="torch SDPA noncausal dropout=0",
        dtype="torch.bfloat16",
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        source_sha256={
            str(p): digest_file(p) for p in (Path(native.__file__), Path(__file__))
        },
    )
    report = {
        "loaded_source_keys": keys + patch_keys + ["model.visual.pos_embed.weight"],
        "cases": {},
    }
    (directory / f"{backend}-identity.json").write_text(json.dumps(identity, indent=2))
    before = {}
    torch.cuda.reset_peak_memory_stats()
    for case in CASES:
        if case.startswith("two_"):
            one, two = load_oracle("image_square"), load_oracle("image_rect")
            first, second = one["input.pixel_values"], two["input.pixel_values"].clone()
            if "time_groups" in case:
                second = one["input.pixel_values"].clone()
                grid = torch.tensor([[2, 16, 16]], dtype=torch.long)
            else:
                grid = torch.cat(
                    (one["input.image_grid_thw"], two["input.image_grid_thw"])
                )
            if "changed_second" in case:
                second = second * -0.75 + 0.5
            pixels = torch.cat((first, second))
        else:
            values = load_oracle(case)
            pixels, grid = values["input.pixel_values"], values["input.image_grid_thw"]
        input_digest = tensor_digest({"pixels": pixels, "grid": grid})
        pixels, grid = pixels.cuda(), grid.cuda()
        # One segment per spatial image/time group, independently of patch contents.
        lengths = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0])
        boundaries = torch.nn.functional.pad(
            lengths.cumsum(0, dtype=torch.int32), (1, 0)
        )
        captures.clear()
        with (
            torch.inference_mode(),
            patch.object(
                implementation, "apply_rotary_pos_emb_vision", capture_rotation
            ),
        ):
            states = embed(pixels)
            if backend == "hf":
                pos = get_vision_position_ids(grid, cfg.spatial_merge_size)
                absolute = hf_absolute(table, grid, cfg.spatial_merge_size)
                position_embeddings = rope(states, pos)
            else:
                pos = native.vision_position_ids(grid, cfg.spatial_merge_size)
                absolute = native.interpolate_vision_positions(
                    table, grid, cfg.spatial_merge_size
                )
                position_embeddings = rope(pos)
            states = states + absolute.to(states.dtype)
            output = block(states, boundaries, position_embeddings)
            captures["vision_block_0"] = output.cpu().clone()
            captures["input_states"] = states.cpu().clone()
        actual = dict(captures)
        torch.save(actual, directory / f"{backend}-{case}.pt")
        metrics = {}
        if backend == "native":
            expected = torch.load(directory / f"hf-{case}.pt", weights_only=True)
            # Block assertion first: identity scaffold must reach numerical RED here.
            for name in [
                "vision_block_0",
                *[n for n in expected if n != "vision_block_0"],
            ]:
                metrics[name] = assert_aligned(
                    name, actual[name], expected[name], {name: EXACT}
                )
        if case in ("two_images", "two_time_groups"):
            before[case] = actual["vision_block_0"]
        elif "changed_second" in case:
            base = case.replace("_changed_second", "")
            torch.testing.assert_close(
                before[base][:256], actual["vision_block_0"][:256], rtol=0, atol=0
            )
            assert not torch.equal(
                before[base][256:], actual["vision_block_0"][256:]
            ), "perturbation ineffective"
            report[
                "image_isolation_exact"
                if base == "two_images"
                else "time_isolation_exact"
            ] = True
        free, total = torch.cuda.mem_get_info()
        report["cases"][case] = dict(
            grid=grid.tolist(),
            boundaries=boundaries.tolist(),
            input_digest=input_digest,
            metrics=metrics,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            free_bytes=free,
            total_bytes=total,
        )
        (directory / f"{backend}-report.json").write_text(json.dumps(report, indent=2))
    for handle in hooks:
        handle.remove()
    refs = [
        weakref.ref(p) for component in (block, embed) for p in component.parameters()
    ]
    del block, embed, module
    gc.collect()
    torch.cuda.empty_cache()
    report["parameters_released"] = all(r() is None for r in refs)
    assert report["parameters_released"]
    (directory / f"{backend}-report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("hf", "native"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    worker(args.worker, args.output)
