"""Independent pinned HF comparisons for the Task10 vision primitives."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import pytest

pytestmark = pytest.mark.hf_reference
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task10"
)


def config():
    return SimpleNamespace(
        patch_size=2,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=8,
        num_attention_heads=2,
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
    )


def hf_absolute(weight, grids, merge):
    import torch.nn.functional as F
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights

    indices, weights = get_vision_interpolation_indices_and_weights(
        grids,
        int(weight.shape[0] ** 0.5),
        mode="bilinear",
        align_corners=True,
        spatial_merge_size=merge,
    )
    return (F.embedding(indices, weight) * weights[:, :, None]).sum(1)


class VisionPositionTests(TestCase):
    def setUp(self):
        import torch

        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        import torch

        torch.set_num_threads(self.threads)

    @pytest.mark.checkpoint
    def test_patch_order_interpolation_rope(self):
        import torch
        from PIL import Image
        from transformers import AutoProcessor
        from tests.reference_qwen3_vl import DEFAULT_MODEL
        from nanovllm.models.qwen3_vl_vision import (
            vision_position_ids,
            Qwen3VLVisionPatchEmbed,
            interpolate_vision_positions,
            Qwen3VLVisionRotaryEmbedding,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionRotaryEmbedding as HFRope,
        )

        grid = torch.tensor([[1, 4, 4]])
        expected = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
            [2, 2],
            [2, 3],
            [3, 2],
            [3, 3],
        ]
        actual = vision_position_ids(grid, 2)
        self.assertEqual(actual.tolist(), expected)
        # Actual HF patchification: 4x4 distinguishable patches, in 2x2-block order.
        processor = AutoProcessor.from_pretrained(
            os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL), local_files_only=True
        )
        image = Image.new("RGB", (64, 64))
        for y in range(64):
            for x in range(64):
                image.putpixel((x, y), (10 * (4 * (y // 16) + x // 16), 0, 0))
        pixels = processor.image_processor(
            image,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            return_tensors="pt",
        )
        self.assertEqual(pixels["image_grid_thw"].tolist(), [[1, 4, 4]])
        from transformers import AutoConfig

        vc = AutoConfig.from_pretrained(
            os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL), local_files_only=True
        ).vision_config
        embed = Qwen3VLVisionPatchEmbed(vc)
        with torch.no_grad():
            embed.proj.weight.zero_()
            embed.proj.bias.zero_()
            embed.proj.weight[0, 0, 0, 0, 0] = 1
            embed.proj.weight[1, 0, 1, 0, 0] = 1
        patch = embed(pixels["pixel_values"])
        expected_values = torch.tensor(
            [0, 10, 40, 50, 20, 30, 60, 70, 80, 90, 120, 130, 100, 110, 140, 150],
            dtype=torch.float32,
        )
        self.assertEqual(patch.shape, (16, 1024))
        self.assertTrue(torch.equal(patch[:, 0], expected_values))
        self.assertTrue(torch.equal(patch[:, 1], expected_values))
        table = torch.arange(2304 * 3, dtype=torch.float32).reshape(2304, 3)
        self.assertTrue(
            torch.equal(
                interpolate_vision_positions(table, grid, 2),
                hf_absolute(table, grid, 2),
            )
        )
        rope = Qwen3VLVisionRotaryEmbedding(vc)
        actual_cos, actual_sin = rope(actual)
        reference = HFRope(vc)(torch.empty(0), actual)
        self.assertTrue(torch.equal(actual_cos, reference[0]))
        self.assertTrue(torch.equal(actual_sin, reference[1]))
        self.assertEqual(actual_cos.shape, (16, 64))
        self.assertEqual(actual_cos.dtype, torch.float32)

    def test_rectangles_temporal_repeat_and_multiple_grids(self):
        import torch
        from transformers.vision_utils import get_vision_position_ids
        from nanovllm.models.qwen3_vl_vision import (
            vision_position_ids,
            interpolate_vision_positions,
        )

        grids = torch.tensor([[2, 4, 6], [1, 6, 4]])
        actual = vision_position_ids(grids, 2)
        self.assertTrue(torch.equal(actual, get_vision_position_ids(grids, 2)))
        self.assertTrue(torch.equal(actual[:24], actual[24:48]))
        self.assertEqual(actual.shape, (72, 2))
        table = torch.linspace(-1, 1, 2304 * 3).reshape(2304, 3).bfloat16()
        self.assertTrue(
            torch.equal(
                interpolate_vision_positions(table, grids, 2),
                hf_absolute(table, grids, 2),
            )
        )

    def test_patch_consumes_temporal_values_without_duplication(self):
        import torch
        from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionPatchEmbed

        module = Qwen3VLVisionPatchEmbed(config())
        values = torch.arange(48, dtype=torch.float32).reshape(2, 24)
        with torch.no_grad():
            module.proj.weight.zero_()
            module.proj.bias.fill_(3)
            module.proj.weight[0, 0, 0, 0, 0] = 1
            module.proj.weight[1, 0, 1, 0, 0] = 1
        actual = module(values)
        self.assertEqual(actual.shape, (2, 8))
        self.assertEqual(actual[:, :2].tolist(), [[3, 7], [27, 31]])
        self.assertTrue(torch.equal(actual[:, 2:], torch.full((2, 6), 3.0)))

    def test_absolute_original_size_border_and_rounding(self):
        import torch
        from nanovllm.models.qwen3_vl_vision import (
            interpolate_vision_positions,
            vision_position_ids,
        )

        generator = torch.Generator().manual_seed(713)
        for dtype in (torch.float32, torch.bfloat16):
            table = torch.randn(2304, 7, generator=generator).to(dtype)
            for h, w, merge in [
                (48, 48, 2),
                (16, 24, 2),
                (24, 16, 2),
                (32, 32, 2),
                (64, 2, 2),
                (1, 5, 1),
                (1, 1, 1),
            ]:
                with self.subTest(dtype=dtype, h=h, w=w):
                    grid = torch.tensor([[2, h, w]])
                    actual = interpolate_vision_positions(table, grid, merge)
                    expected = hf_absolute(table, grid, merge)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.equal(actual[0], table[0].float()))
                    last_index = (0 if h == 1 else 47) * 48 + (0 if w == 1 else 47)
                    self.assertTrue(
                        torch.equal(actual[h * w - 1], table[last_index].float())
                    )
                    if h == w == 48:
                        coords = vision_position_ids(grid, merge)
                        self.assertTrue(
                            torch.equal(
                                actual, table[coords[:, 0] * 48 + coords[:, 1]].float()
                            )
                        )

    def test_axial_layout_and_fp32_rotation(self):
        import torch
        from nanovllm.models.qwen3_vl_vision import (
            Qwen3VLVisionRotaryEmbedding,
            apply_rotary_pos_emb_vision,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision as hf_apply,
        )

        vc = config()
        vc.hidden_size = 128
        vc.num_attention_heads = 2
        pos = torch.tensor([[0, 0], [2, 9], [9, 2], [31, 47]])
        rope = Qwen3VLVisionRotaryEmbedding(vc)
        cos, sin = rope(pos)
        inv = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        angles = torch.cat((pos[:, 0, None] * inv, pos[:, 1, None] * inv), -1)
        self.assertTrue(torch.equal(cos, torch.cat((angles, angles), -1).cos()))
        self.assertTrue(torch.equal(sin, torch.cat((angles, angles), -1).sin()))
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.linspace(-2, 2, 4 * 3 * 64).reshape(4, 3, 64).to(dtype)
            k = q[:, :2].flip(-1)
            actual = apply_rotary_pos_emb_vision(q, k, cos, sin)
            expected = hf_apply(q, k, cos, sin)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
            self.assertTrue(all(a.dtype == dtype for a in actual))
        # Frequencies must survive normal mixed-precision module conversion.
        self.assertTrue(torch.equal(rope.bfloat16()(pos)[0], cos))


@pytest.mark.parametrize(
    "grid,merge",
    [
        ([], 2),
        ([[1, 3, 4]], 2),
        ([[0, 4, 4]], 2),
        ([[1, -2, 4]], 2),
        ([[1, 4, 4]], 0),
        ([[1, 4, 4]], True),
    ],
)
def test_invalid_grids(grid, merge):
    import torch
    from nanovllm.models.qwen3_vl_vision import vision_position_ids

    with pytest.raises(ValueError):
        vision_position_ids(torch.tensor(grid, dtype=torch.long).reshape(-1, 3), merge)


@pytest.mark.gpu
@pytest.mark.checkpoint
class VisionPositionGPUTests(TestCase):
    def test_real_weights_patch_absolute_and_rotary(self):
        import tempfile
        from tests.support import run_worker

        EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="gpu-", dir=EVIDENCE))
        result = run_worker(
            "tests.test_qwen3_vl_vision_positions", ["--gpu-worker", str(directory)]
        )
        (directory / "worker.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = json.loads((directory / "comparison.json").read_text())
        self.assertEqual(
            state["loaded_source_keys"],
            [
                "model.visual.patch_embed.proj.weight",
                "model.visual.patch_embed.proj.bias",
                "model.visual.pos_embed.weight",
            ],
        )
        self.assertEqual(len(state["cases"]), 7)
        self.assertFalse(state["failures"])
        self.assertTrue(state["parameters_released"])


def gpu_worker(directory):
    import gc
    import weakref
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor
    from PIL import Image
    from tests.reference_qwen3_vl import (
        DEFAULT_MODEL,
        provenance,
        assert_aligned,
        tensor_metrics,
    )
    from tests.test_qwen3_vl_text import load_oracle
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionPatchEmbed as HFPatch,
        Qwen3VLVisionRotaryEmbedding as HFRope,
        apply_rotary_pos_emb_vision as hf_apply,
    )
    from transformers.vision_utils import get_vision_position_ids
    from nanovllm.models.qwen3_vl_vision import (
        Qwen3VLVisionPatchEmbed,
        Qwen3VLVisionRotaryEmbedding,
        vision_position_ids,
        interpolate_vision_positions,
        apply_rotary_pos_emb_vision,
    )

    model = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    identity = provenance(model)
    vc = AutoConfig.from_pretrained(model, local_files_only=True).vision_config
    keys = [
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        "model.visual.pos_embed.weight",
    ]
    with safe_open(
        str(Path(model) / "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        tensors = {name: f.get_tensor(name) for name in keys}
    state = {
        name.removeprefix("model.visual.patch_embed."): tensors[name]
        for name in keys[:2]
    }
    native = Qwen3VLVisionPatchEmbed(vc).to(dtype=torch.bfloat16)
    reference = HFPatch(vc).to(dtype=torch.bfloat16)
    native.load_state_dict(state, strict=True)
    reference.load_state_dict(state, strict=True)
    native = native.cuda().eval()
    reference = reference.cuda().eval()
    table = tensors[keys[2]].cuda()
    rope = Qwen3VLVisionRotaryEmbedding(vc).cuda()
    hf_rope = HFRope(vc).cuda()
    identity.update(
        gpu=torch.cuda.get_device_name(),
        dtype=str(native.proj.weight.dtype),
        cuda=torch.version.cuda,
    )
    report = {"loaded_source_keys": keys, "cases": {}, "failures": []}
    torch.cuda.reset_peak_memory_stats()
    processor = None
    for case in [
        "image_square",
        "image_rect",
        "image_limit",
        "image_same_layout_other_pixels",
        "image_candy",
        "image_cat",
        "image_rect_transposed",
    ]:
        if case == "image_rect_transposed":
            processor = AutoProcessor.from_pretrained(model, local_files_only=True)
            values = processor.image_processor(
                Image.new("RGB", (256, 384), (180, 40, 20)), return_tensors="pt"
            )
            pixels, grid = values["pixel_values"], values["image_grid_thw"]
        else:
            values = load_oracle(case)
            pixels, grid = values["input.pixel_values"], values["input.image_grid_thw"]
        pixels = pixels.cuda()
        grid = grid.cuda()
        with torch.inference_mode():
            positions = vision_position_ids(grid, 2)
            hf_pos = get_vision_position_ids(grid, 2)
            patch = native(pixels)
            expected_patch = reference(pixels)
            absolute = interpolate_vision_positions(table, grid, 2)
            expected_absolute = hf_absolute(table, grid, 2)
            cos, sin = rope(positions)
            hf_cos, hf_sin = hf_rope(patch, hf_pos)
            q = patch.reshape(-1, 16, 64)
            k = q.flip(-1)
            rotated = apply_rotary_pos_emb_vision(q, k, cos, sin)
            expected_rotated = hf_apply(q, k, hf_cos, hf_sin)
            actual = {
                "positions": positions,
                "patch": patch,
                "absolute": absolute,
                "patch_plus_absolute": patch + absolute.to(patch.dtype),
                "cos": cos,
                "sin": sin,
                "rotated_q": rotated[0],
                "rotated_k": rotated[1],
            }
            expected = {
                "positions": hf_pos,
                "patch": expected_patch,
                "absolute": expected_absolute,
                "patch_plus_absolute": expected_patch
                + expected_absolute.to(expected_patch.dtype),
                "cos": hf_cos,
                "sin": hf_sin,
                "rotated_q": expected_rotated[0],
                "rotated_k": expected_rotated[1],
            }
            metrics = {}
            for name, value in actual.items():
                if value.is_floating_point():
                    metrics[name] = tensor_metrics(value, expected[name])
                try:
                    assert_aligned(
                        name,
                        value,
                        expected[name],
                        {name: {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}},
                    )
                except AssertionError as error:
                    report["failures"].append(f"{case}:{error}")
            torch.save(
                {
                    "actual": {k: v.cpu() for k, v in actual.items()},
                    "expected": {k: v.cpu() for k, v in expected.items()},
                },
                directory / f"{case}.pt",
            )
            free, total = torch.cuda.mem_get_info()
            report["cases"][case] = {
                "grid": grid.tolist(),
                "metrics": metrics,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "free_bytes": free,
                "total_bytes": total,
            }
    refs = [
        weakref.ref(p) for module in (native, reference) for p in module.parameters()
    ]
    del (
        native,
        reference,
        table,
        pixels,
        grid,
        actual,
        expected,
        patch,
        expected_patch,
        absolute,
        expected_absolute,
        q,
        k,
        cos,
        sin,
        hf_cos,
        hf_sin,
        rotated,
        expected_rotated,
        positions,
        hf_pos,
        value,
    )
    gc.collect()
    torch.cuda.empty_cache()
    report["parameters_released"] = all(r() is None for r in refs)
    (directory / "identity.json").write_text(json.dumps(identity, indent=2))
    (directory / "comparison.json").write_text(json.dumps(report, indent=2))
    assert not report["failures"], report["failures"]


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-worker", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    gpu_worker(args.gpu_worker)
