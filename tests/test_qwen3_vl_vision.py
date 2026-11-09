"""Task11c full vision stack: real weights, pinned HF and assembled isolation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase

import pytest

pytestmark = pytest.mark.hf_reference
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11c"
)
EXACT = {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}
CASES = (
    "image_square",
    "image_rect",
    "image_limit",
    "image_same_layout_other_pixels",
    "image_candy",
    "image_cat",
    "image_rect_transposed",
    "two_images",
    "two_images_changed_second",
    "two_images_reversed",
    "two_time_groups",
    "two_time_groups_changed_second",
)


def tiny_vision():
    import torch
    from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionModel

    cfg = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        num_attention_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        depth=4,
        deepstack_visual_indexes=[0, 1, 2],
        out_hidden_size=16,
        rope_parameters={"rope_type": "axial", "rope_theta": 10000.0},
    )
    module = Qwen3VLVisionModel(cfg).eval()
    with torch.no_grad():
        for p in module.parameters():
            p.zero_()
        module.patch_embed.proj.weight[0, 0, 0, 0, 0] = 1
        module.pos_embed.weight.fill_(0.5)
        for i, block in enumerate(module.blocks):
            block.attn.proj.bias.fill_(i + 1)
    return module, cfg


class VisionCPUTests(TestCase):
    def test_deepstack_consumes_complete_selected_blocks_and_main_uses_last(self):
        import torch

        vision, _ = tiny_vision()
        inputs = {}
        handles = []
        for name, merger in [
            ("main", vision.merger),
            *[(str(i), m) for i, m in enumerate(vision.deepstack_merger_list)],
        ]:
            handles.append(
                merger.register_forward_pre_hook(
                    lambda m, a, name=name: inputs.__setitem__(
                        name, a[0].detach().clone()
                    )
                )
            )
        pixels = torch.zeros(16, 24)
        pixels[:, 0] = torch.arange(16)
        main, deepstack = vision(pixels, torch.tensor([[1, 4, 4]]))
        self.assertEqual(main.shape, (4, 16))
        self.assertEqual(len(deepstack), 3)
        self.assertTrue(all(x.shape == (4, 16) for x in deepstack))
        # All block branches are zero except explicit biases1/2/3/4; absolute=0.5.
        for name, offset in [("0", 1.5), ("1", 3.5), ("2", 6.5), ("main", 10.5)]:
            expected = torch.full((16, 8), offset)
            expected[:, 0] += torch.arange(16)
            torch.testing.assert_close(inputs[name], expected, rtol=0, atol=0)
        for h in handles:
            h.remove()

    def test_grid_builds_image_time_boundaries_and_position_embeddings(self):
        import torch
        from transformers.vision_utils import get_vision_position_ids
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionRotaryEmbedding,
        )

        vision, cfg = tiny_vision()
        saved = {}

        def save(m, args):
            saved["cu"] = args[1].clone()
            saved["rope"] = args[2]

        hook = vision.blocks[0].register_forward_pre_hook(save)
        grids = torch.tensor([[1, 4, 4], [2, 4, 6]])
        vision(torch.zeros(64, 24), grids)
        self.assertEqual(saved["cu"].dtype, torch.int32)
        self.assertEqual(saved["cu"].tolist(), [0, 16, 40, 64])
        expected = Qwen3VLVisionRotaryEmbedding(cfg)(
            torch.empty(0), get_vision_position_ids(grids, 2)
        )
        for a, b in zip(saved["rope"], expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        hook.remove()


@pytest.mark.parametrize("grid", [[[1, 4, 6]], [[0, 4, 4]], [[1, 3, 4]], []])
def test_grid_mismatch_rejected_before_patch_projection(grid):
    import torch

    vision, _ = tiny_vision()
    called = []
    handle = vision.patch_embed.register_forward_hook(lambda *a: called.append(True))
    with pytest.raises(ValueError):
        vision(torch.zeros(16, 24), torch.tensor(grid, dtype=torch.long).reshape(-1, 3))
    assert not called
    handle.remove()


@pytest.mark.gpu
@pytest.mark.checkpoint
class VisionTests(TestCase):
    def test_blocks_mergers_and_segment_isolation(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="gpu-", dir=EVIDENCE))
        for backend in ("hf", "native"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.test_qwen3_vl_vision",
                    "--worker",
                    backend,
                    "--output",
                    str(directory),
                ],
                timeout=240,
                capture_output=True,
                text=True,
            )
            (directory / f"{backend}.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((directory / "native-report.json").read_text())
        self.assertEqual(len(report["loaded_source_keys"]), 315)
        self.assertEqual(len(report["skipped_language_keys"]), 310)
        self.assertEqual(set(report["cases"]), set(CASES))
        self.assertTrue(report["parameters_released"])
        self.assertTrue(report["image_isolation_exact"])
        self.assertTrue(report["time_isolation_exact"])


def load_visual(vision, model):
    """Standalone component audit, then one-tensor streaming copies; no full state_dict."""
    import torch
    from safetensors import safe_open

    params = dict(vision.named_parameters())
    with safe_open(
        str(Path(model) / "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        keys = sorted(k for k in f.keys() if k.startswith("model.visual."))
        skipped = sorted(k for k in f.keys() if k.startswith("model.language_model."))
        assert set(keys + skipped) == set(f.keys())
        assert {k.removeprefix("model.visual.") for k in keys} == set(params)
        for k in keys:
            assert tuple(f.get_slice(k).get_shape()) == tuple(
                params[k.removeprefix("model.visual.")].shape
            )
        with torch.no_grad():
            for k in keys:
                value = f.get_tensor(k)
                p = params[k.removeprefix("model.visual.")]
                assert p.dtype == value.dtype == torch.bfloat16 and not p.is_meta
                p.copy_(value)
                del value
    return keys, skipped


def case_inputs(case, model):
    import torch
    from tests.test_qwen3_vl_text import load_oracle

    if case == "image_rect_transposed":
        from transformers import AutoProcessor
        from PIL import Image

        p = AutoProcessor.from_pretrained(model, local_files_only=True)
        values = p.image_processor(
            Image.new("RGB", (256, 384), (180, 40, 20)), return_tensors="pt"
        )
        return values["pixel_values"], values["image_grid_thw"]
    if case.startswith("two_"):
        first = load_oracle("image_square")
        second = load_oracle("image_rect" if "images" in case else "image_square")
        a, b = first["input.pixel_values"], second["input.pixel_values"].clone()
        grid = torch.cat(
            (first["input.image_grid_thw"], second["input.image_grid_thw"])
        )
        if "time_groups" in case:
            grid = torch.tensor([[2, 16, 16]])
        if "changed_second" in case:
            b = b * -0.75 + 0.5
        if case == "two_images_reversed":
            return torch.cat((b, a)), grid.flip(0)
        return torch.cat((a, b)), grid
    values = load_oracle(case)
    return values["input.pixel_values"], values["input.image_grid_thw"]


def worker(backend, directory):
    import gc
    import weakref
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel as HFVision,
    )
    from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionModel
    from tests.reference_qwen3_vl import (
        DEFAULT_MODEL,
        provenance,
        assert_aligned,
        tensor_digest,
    )
    from tests.support import digest_file

    model = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    identity = provenance(model)
    cfg = AutoConfig.from_pretrained(model, local_files_only=True).vision_config
    cfg._attn_implementation = "sdpa"
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        vision = (HFVision if backend == "hf" else Qwen3VLVisionModel)(cfg)
    finally:
        torch.set_default_dtype(previous)
    # HF rotary constructors explicitly use FP32; preserve that dtype on device move.
    if backend == "hf":
        assert vision.rotary_pos_emb.inv_freq.dtype == torch.float32
    torch.cuda.reset_peak_memory_stats()
    vision = vision.cuda().eval()
    keys, skipped = load_visual(vision, model)
    identity.update(
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        dtype="torch.bfloat16",
        backend="SDPA noncausal; compile/Graph off",
        source_sha256={
            str(p): digest_file(p)
            for p in (
                Path(__file__),
                Path(__file__).parents[1] / "nanovllm/models/qwen3_vl_vision.py",
            )
        },
    )
    (directory / f"{backend}-identity.json").write_text(json.dumps(identity, indent=2))
    captures = {}
    hooks = []
    for i, block in enumerate(vision.blocks):
        hooks.append(
            block.register_forward_hook(
                lambda m, a, out, i=i: captures.__setitem__(
                    f"vision_layer_{i}", out.detach().cpu().clone()
                )
            )
        )
    hooks.append(
        vision.patch_embed.register_forward_hook(
            lambda m, a, out: captures.__setitem__("patch", out.detach().cpu().clone())
        )
    )
    report = {
        "loaded_source_keys": keys,
        "skipped_language_keys": skipped,
        "cases": {},
        "failures": [],
    }
    before = {}
    for case in CASES:
        pixels, grid = case_inputs(case, model)
        input_digest = tensor_digest({"pixels": pixels, "grid": grid})
        captures.clear()
        with torch.inference_mode():
            result = vision(pixels.cuda(), grid.cuda())
            main, deep = (
                (result.pooler_output, result.deepstack_features)
                if backend == "hf"
                else result
            )
            assert len(deep) == 3
            if case == "image_rect":
                assert tuple(main.shape) == (96, 2048)
                assert all(tuple(x.shape) == (96, 2048) for x in deep)
            captures["main"] = main.cpu().clone()
            for i, x in enumerate(deep):
                captures[f"deepstack_{i}"] = x.cpu().clone()
        actual = dict(captures)
        torch.save(actual, directory / f"{backend}-{case}.pt")
        metrics = {}
        if backend == "native":
            expected = torch.load(directory / f"hf-{case}.pt", weights_only=True)
            # Exact plan RED assertion first, then every layer and all four outputs.
            order = [
                "vision_layer_17",
                *[k for k in expected if k != "vision_layer_17"],
            ]
            for k in order:
                try:
                    metrics[k] = assert_aligned(k, actual[k], expected[k], {k: EXACT})
                except AssertionError as error:
                    report["failures"].append(f"{case}: {error}")
                    (directory / f"{backend}-report.json").write_text(
                        json.dumps(report, indent=2)
                    )
                    raise
        if case in ("two_images", "two_time_groups"):
            before[case] = actual
        elif "changed_second" in case:
            base = case.replace("_changed_second", "")
            for k, v in actual.items():
                limit = 64 if k == "main" or k.startswith("deepstack_") else 256
                torch.testing.assert_close(
                    before[base][k][:limit], v[:limit], rtol=0, atol=0
                )
            assert not torch.equal(before[base]["main"][64:], actual["main"][64:])
            report[
                "image_isolation_exact"
                if base == "two_images"
                else "time_isolation_exact"
            ] = True
        free, total = torch.cuda.mem_get_info()
        report["cases"][case] = dict(
            grid=grid.tolist(),
            input_digest=input_digest,
            metrics=metrics,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            free_bytes=free,
            total_bytes=total,
        )
        (directory / f"{backend}-report.json").write_text(json.dumps(report, indent=2))
    for h in hooks:
        h.remove()
    refs = [weakref.ref(p) for p in vision.parameters()]
    del vision, block
    gc.collect()
    torch.cuda.empty_cache()
    report["parameters_released"] = all(r() is None for r in refs)
    assert report["parameters_released"]
    (directory / f"{backend}-report.json").write_text(json.dumps(report, indent=2))
    assert not report["failures"], report["failures"]


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("hf", "native"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    worker(args.worker, args.output)
