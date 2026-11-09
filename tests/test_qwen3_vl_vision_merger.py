"""Task11b: fixed HF hidden states, real merger weights and independent CPU math."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase

import pytest

pytestmark = pytest.mark.hf_reference
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task11b"
)
EXACT = {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}
CASES = (
    "image_square",
    "image_rect",
    "image_limit",
    "image_same_layout_other_pixels",
    "image_candy",
    "image_cat",
)
# (checkpoint namespace, post-shuffle norm, independently captured input layer)
MERGERS = (
    ("merger", False, 23),
    ("deepstack_merger_list.0", True, 5),
    ("deepstack_merger_list.1", True, 11),
    ("deepstack_merger_list.2", True, 17),
)


def tiny_merger(post):
    import torch
    from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionPatchMerger

    cfg = SimpleNamespace(hidden_size=2, spatial_merge_size=2, out_hidden_size=8)
    module = Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=post).double()
    with torch.no_grad():
        module.norm.weight.fill_(1)
        module.norm.bias.zero_()
        for layer in (module.linear_fc1, module.linear_fc2):
            layer.weight.copy_(torch.eye(8))
            layer.bias.zero_()
    return module


def scalar_expected(rows, post):
    groups = [sum(rows[i : i + 4], []) for i in range(0, len(rows), 4)]
    norm_groups = groups if post else rows
    normalized = []
    for group in norm_groups:
        mean = sum(group) / len(group)
        variance = sum((v - mean) ** 2 for v in group) / len(group)
        normalized.extend((v - mean) / math.sqrt(variance + 1e-6) for v in group)
    activated = [0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in normalized]
    return [activated[i : i + 8] for i in range(0, len(activated), 8)]


class VisionMergerCPUTests(TestCase):
    def test_same_hidden_distinguishes_pre_and_postshuffle_normalization(self):
        import torch

        rows = [
            [0.0, 2.0],
            [10.0, 16.0],
            [-9.0, -1.0],
            [3.0, 4.0],
            [5.0, 1.0],
            [-4.0, 8.0],
            [2.0, 2.0],
            [20.0, 30.0],
        ]
        x = torch.tensor(rows, dtype=torch.float64)
        outputs = []
        for post in (False, True):
            actual = tiny_merger(post)(x)
            expected = torch.tensor(scalar_expected(rows, post), dtype=torch.float64)
            torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
            outputs.append(actual)
        self.assertFalse(torch.allclose(*outputs))

    def test_merge_keeps_consecutive_block_order_and_uses_both_linear_biases(self):
        import torch

        module = tiny_merger(False)
        with torch.no_grad():
            module.linear_fc1.bias.copy_(torch.arange(8).double() / 8)
            module.linear_fc2.bias.copy_(torch.arange(8).double())
        rows = [
            [0.0, 2.0],
            [2.0, 0.0],
            [3.0, 3.0],
            [-2.0, 2.0],
            [9.0, 3.0],
            [0.0, 0.0],
            [-2.0, -1.0],
            [4.0, 0.0],
        ]
        normalized = []
        for group in rows:
            mean = sum(group) / 2
            variance = sum((v - mean) ** 2 for v in group) / 2
            normalized.extend((v - mean) / math.sqrt(variance + 1e-6) for v in group)
        expected = []
        for start in (0, 8):
            result = []
            for j, v in enumerate(normalized[start : start + 8]):
                z = v + j / 8
                result.append(0.5 * z * (1 + math.erf(z / math.sqrt(2))) + j)
            expected.append(result)
        torch.testing.assert_close(
            module(torch.tensor(rows, dtype=torch.float64)),
            torch.tensor(expected, dtype=torch.float64),
            rtol=1e-13,
            atol=1e-13,
        )


@pytest.mark.gpu
@pytest.mark.checkpoint
class VisionMergerTests(TestCase):
    def test_real_main_and_three_deepstack_mergers(self):
        from tests.support import run_worker

        EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="gpu-", dir=EVIDENCE))
        for backend in ("hf", "native"):
            result = run_worker(
                "tests.test_qwen3_vl_vision_merger",
                ["--worker", backend, "--output", str(directory)],
            )
            (directory / f"{backend}.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((directory / "native-report.json").read_text())
        self.assertEqual(set(report["mergers"]), {m[0] for m in MERGERS})
        for name, post, _ in MERGERS:
            record = report["mergers"][name]
            self.assertEqual(record["norm_weight_shape"], [4096 if post else 1024])
            self.assertEqual(set(record["cases"]), set(CASES))
            self.assertTrue(record["parameters_released"])


def hidden_digest(value):
    import torch

    h = hashlib.sha256()
    h.update(str((tuple(value.shape), value.dtype)).encode())
    h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def worker(backend, directory):
    import gc
    import weakref
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionPatchMerger as HFMerger,
    )
    from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionPatchMerger
    from tests.reference_qwen3_vl import DEFAULT_MODEL, provenance, assert_aligned
    from tests.support import digest_file
    from tests.test_qwen3_vl_text import load_oracle

    model = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    identity = provenance(model)
    cfg = AutoConfig.from_pretrained(model, local_files_only=True).vision_config
    # Retain only the four required CPU hidden states, not the complete artifacts.
    inputs = {}
    for case in CASES:
        values = load_oracle(case)
        inputs[case] = {
            layer: values[f"vision.layer.{layer}"] for _, _, layer in MERGERS
        }
        inputs[case]["grid"] = values["input.image_grid_thw"].tolist()
    del values
    identity.update(
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        dtype="torch.bfloat16",
        backend="PyTorch LayerNorm/Linear/GELU(approximate=none)",
        source_sha256={
            str(p): digest_file(p)
            for p in (
                Path(__file__),
                Path(__file__).parents[1] / "nanovllm/models/qwen3_vl_vision.py",
            )
        },
    )
    (directory / f"{backend}-identity.json").write_text(json.dumps(identity, indent=2))
    report = {"mergers": {}, "failures": []}
    torch.cuda.reset_peak_memory_stats()
    for name, post, layer in MERGERS:
        merger_type = HFMerger if backend == "hf" else Qwen3VLVisionPatchMerger
        merger = merger_type(cfg, use_postshuffle_norm=post).to(dtype=torch.bfloat16)
        prefix = "model.visual." + name + "."
        with safe_open(
            str(Path(model) / "model.safetensors"), framework="pt", device="cpu"
        ) as f:
            keys = sorted(k for k in f.keys() if k.startswith(prefix))
            assert set(k.removeprefix(prefix) for k in keys) == set(merger.state_dict())
            merger.load_state_dict(
                {k.removeprefix(prefix): f.get_tensor(k) for k in keys}, strict=True
            )
        merger = merger.cuda().eval()
        # Explicit parent-plan dimension assertions before numerical checks.
        assert tuple(merger.norm.weight.shape) == (4096 if post else 1024,)
        captures = {}
        hooks = []
        for label, component in (
            ("norm", merger.norm),
            ("fc1", merger.linear_fc1),
            ("gelu", merger.act_fn),
        ):
            hooks.append(
                component.register_forward_hook(
                    lambda m, a, out, label=label: captures.__setitem__(
                        label, out.detach().cpu().clone()
                    )
                )
            )
        record = {
            "source_keys": keys,
            "input_layer": layer,
            "norm_weight_shape": list(merger.norm.weight.shape),
            "cases": {},
        }
        report["mergers"][name] = record
        for case in CASES:
            cpu_hidden = inputs[case][layer]
            hidden = cpu_hidden.cuda()
            captures.clear()
            with torch.inference_mode():
                output = merger(hidden)
                captures["output"] = output.cpu().clone()
                captures["input"] = cpu_hidden
            actual = dict(captures)
            assert output.shape == (len(hidden) // 4, 2048)
            filename = name.replace(".", "_") + "-" + case + ".pt"
            torch.save(actual, directory / f"{backend}-{filename}")
            metrics = {}
            if backend == "native":
                expected = torch.load(directory / f"hf-{filename}", weights_only=True)
                # Numerical output first: the importable scaffold must fail here.
                for key in ("output", "norm", "fc1", "gelu", "input"):
                    label = "deep_merger" if post else "main_merger"
                    try:
                        metrics[key] = assert_aligned(
                            label, actual[key], expected[key], {label: EXACT}
                        )
                    except AssertionError as error:
                        report["failures"].append(f"{name}:{case}:{key}: {error}")
                        break
            free, total = torch.cuda.mem_get_info()
            record["cases"][case] = dict(
                grid=inputs[case]["grid"],
                hidden_digest=hidden_digest(cpu_hidden),
                output_shape=list(output.shape),
                metrics=metrics,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                free_bytes=free,
                total_bytes=total,
            )
            (directory / f"{backend}-report.json").write_text(
                json.dumps(report, indent=2)
            )
        for handle in hooks:
            handle.remove()
        refs = [weakref.ref(p) for p in merger.parameters()]
        del merger
        gc.collect()
        torch.cuda.empty_cache()
        record["parameters_released"] = all(r() is None for r in refs)
        assert record["parameters_released"]
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
