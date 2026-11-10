"""Task12 fusion: literal wiring checks and two independent real-checkpoint modes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import TestCase

import pytest

from tests.test_qwen3_vl_loader import single_rank as single_rank

pytestmark = pytest.mark.hf_reference
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task12"
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


def wiring_model():
    """Cheap decoder arithmetic isolates wrapper ordering; real decoders run on GPU."""
    import torch
    from torch import nn
    from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from tests.test_qwen3_vl_text import small_config

    cfg = small_config()
    cfg.text_config.num_hidden_layers = 4
    model = Qwen3VLForConditionalGeneration(cfg)
    text = model.model.language_model
    with torch.no_grad():
        text.embed_tokens.weight.fill_(1)

    class CompleteResidual(nn.Module):
        def forward(self, positions, hidden):
            return hidden * 2 + 1

    text.layers = nn.ModuleList([CompleteResidual() for _ in range(4)])
    text.norm = nn.Identity()
    return model


class FusionCPUTests(TestCase):
    def test_replacement_and_complete_residual_injection_only_first_three(self):
        # Missing replacement, pre-residual injection or a fourth injection changes literals.
        import torch

        model = wiring_model()
        ids = torch.tensor([1, 2, 3, 4])
        mask = torch.tensor([False, True, False, True])
        main = torch.stack([torch.full((256,), 10.0), torch.full((256,), 20.0)])
        deep = [torch.full((2, 256), x) for x in (3.0, 5.0, 7.0)]
        observed = []
        handles = [
            layer.register_forward_pre_hook(lambda m, a: observed.append(a[1].clone()))
            for layer in model.model.language_model.layers
        ]
        with torch.inference_mode():
            out = model(
                ids,
                torch.zeros(3, 4, dtype=torch.long),
                visual_features=(main, deep),
                visual_mask=mask,
            )
        for handle in handles:
            handle.remove()
        # Inputs to layer1/2/3 are complete previous residual plus its DeepStack.
        for actual, values in zip(
            observed,
            ([1, 10, 1, 20], [3, 24, 3, 44], [7, 54, 7, 94], [15, 116, 15, 196]),
        ):
            torch.testing.assert_close(
                actual,
                torch.tensor(values).float()[:, None].expand(-1, 256),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            out,
            torch.tensor([31, 233, 31, 393]).float()[:, None].expand(-1, 256),
            rtol=0,
            atol=0,
        )
        # Caller-owned feature buffers are immutable across repeated forwards.
        torch.testing.assert_close(
            main[:, 0], torch.tensor([10.0, 20.0]), rtol=0, atol=0
        )
        torch.testing.assert_close(deep[0], torch.full((2, 256), 3.0), rtol=0, atol=0)

    def test_default_text_path_still_has_no_visual_state(self):
        import torch

        model = wiring_model()
        result = model(torch.tensor([1, 2]), torch.zeros(3, 2, dtype=torch.long))
        torch.testing.assert_close(result, torch.full((2, 256), 31.0), rtol=0, atol=0)
        assert not hasattr(model.model, "visual")


@pytest.mark.parametrize(
    "invalid",
    [
        "features_only",
        "mask_only",
        "mask_rank",
        "mask_dtype",
        "mask_length",
        "count",
        "main_rank",
        "main_hidden",
        "deep_length",
        "deep_count",
        "deep_hidden",
        "deep_rank",
    ],
)
def test_invalid_fusion_rejected_before_decoder_or_cache_side_effect(invalid):
    import torch

    model = wiring_model()
    ids = torch.tensor([1, 2, 3, 4])
    mask = torch.tensor([False, True, False, True])
    main = torch.ones(2, 256)
    deep = [torch.ones(2, 256) for _ in range(3)]
    if invalid == "features_only":
        mask = None
    elif invalid == "mask_only":
        main = None
    elif invalid == "mask_rank":
        mask = mask[None]
    elif invalid == "mask_dtype":
        mask = mask.long()
    elif invalid == "mask_length":
        mask = mask[:-1]
    elif invalid == "count":
        mask = torch.tensor([True, True, True, False])
    elif invalid == "main_rank":
        main = main[None]
    elif invalid == "main_hidden":
        main = main[:, :-1]
    elif invalid == "deep_length":
        deep = deep[:2]
    elif invalid == "deep_count":
        deep[2] = torch.ones(1, 256)
    elif invalid == "deep_hidden":
        deep[2] = torch.ones(2, 255)
    elif invalid == "deep_rank":
        deep[2] = deep[2][None]
    # Observe the earliest decoder boundary: real attention/KV cannot have run.
    touched = []
    handle = model.model.language_model.layers[0].register_forward_pre_hook(
        lambda *a: touched.append(True)
    )
    with pytest.raises(ValueError):
        model(
            ids,
            torch.zeros(3, 4, dtype=torch.long),
            visual_features=None if main is None else (main, deep),
            visual_mask=mask,
        )
    assert touched == []
    handle.remove()


@pytest.mark.checkpoint
def test_full_wrapper_registers_native_vision_and_complete_checkpoint_schema():
    import torch
    from transformers import AutoConfig
    from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionModel
    from nanovllm.utils.loader import _audit_qwen3_vl_weights
    from tests.reference_qwen3_vl import DEFAULT_MODEL

    path = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(
            AutoConfig.from_pretrained(path, local_files_only=True), text_only=False
        )
    assert isinstance(model.model.visual, Qwen3VLVisionModel)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    sources, targets, report = _audit_qwen3_vl_weights(model, path)
    assert len(sources) == len(targets) == 625
    assert report.skipped_keys == []


def capture_fusion(text, saved):
    """Next-layer inputs include HF's injection, unlike decoder output hooks."""
    handles = []

    def save(key, tensor):
        saved[key] = tensor.detach().reshape(-1, tensor.shape[-1]).cpu().clone()

    for i in range(4):
        key = "fused" if i == 0 else f"deepstack_after_layer_{i - 1}"
        # Native layers receive (positions, hidden); HF receives (hidden, ...).
        handles.append(
            text.layers[i].register_forward_pre_hook(
                lambda m, a, key=key: save(
                    key, a[0] if a[0].is_floating_point() else a[1]
                )
            )
        )
    handles.append(text.norm.register_forward_hook(lambda m, a, out: save("norm", out)))
    return handles


@pytest.mark.gpu
@pytest.mark.checkpoint
class FusionTests(TestCase):
    def run_mode(self, mode):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"{mode}-", dir=EVIDENCE))
        for backend in ("hf", mode):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tests.test_qwen3_vl_fusion",
                    "--worker",
                    backend,
                    "--output",
                    str(directory),
                ],
                timeout=360,
                capture_output=True,
                text=True,
            )
            (directory / f"{backend}.log").write_text(result.stdout + result.stderr)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((directory / f"{mode}-report.json").read_text())
        self.assertEqual(len(report["loaded_keys"]), 625)
        self.assertEqual(report["skipped_keys"], [])
        self.assertTrue(report["parameters_released"])
        self.assertEqual(set(report["cases"]), set(CASES))

    def test_replacement_and_deepstack(self):
        """HF_FIXED_FEATURES isolates fusion from the native encoder."""
        self.run_mode("hf_fixed_features")

    def test_native_encoded_visual_features(self):
        """NATIVE_ENCODED_FEATURES exercises the full native image/text pipeline."""
        self.run_mode("native_encoded_features")


def worker(backend, directory, fault=None):
    import gc
    import weakref
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration as HFModel
    from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from nanovllm.layers.mrope import build_mrope_positions
    from nanovllm.utils.context import reset_context
    from nanovllm.utils.loader import load_qwen3_vl_weights
    from tests.test_qwen3_vl_text import bf16_cuda, prefill_context, load_oracle
    from tests.reference_qwen3_vl import (
        DEFAULT_MODEL,
        provenance,
        hf_positions,
        assert_aligned,
        tensor_metrics,
        tensor_digest,
    )
    from tests.support import digest_file

    path = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    identity = provenance(path)
    identity.update(
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        torch=torch.__version__,
        dtype="torch.bfloat16",
        language_backend="flash_attention_2",
        vision_backend="sdpa",
        mode=backend,
        TORCHDYNAMO_DISABLE=os.environ.get("TORCHDYNAMO_DISABLE"),
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        source_sha256={
            str(p): digest_file(p)
            for p in (
                Path(__file__),
                Path("nanovllm/models/qwen3_vl.py"),
                Path("nanovllm/models/qwen3_vl_vision.py"),
            )
        },
    )
    (directory / f"{backend}-identity.json").write_text(json.dumps(identity, indent=2))
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    rendezvous = tempfile.TemporaryDirectory()
    if backend == "hf":
        model, info = HFModel.from_pretrained(
            path,
            dtype=torch.bfloat16,
            attn_implementation={
                "text_config": "flash_attention_2",
                "vision_config": "sdpa",
            },
            local_files_only=True,
            output_loading_info=True,
        )
        assert (
            not info["missing_keys"]
            and not info["unexpected_keys"]
            and not info["mismatched_keys"]
        )
        model = model.cuda().eval()
        assert (
            model.model.language_model.config._attn_implementation
            == "flash_attention_2"
        )
        assert model.model.visual.config._attn_implementation == "sdpa"
        assert model.model.visual.rotary_pos_emb.inv_freq.dtype == torch.float32
        report = {"cases": {}}
    else:
        dist.init_process_group(
            "gloo",
            init_method=Path(rendezvous.name, "init").as_uri(),
            rank=0,
            world_size=1,
        )
        with bf16_cuda():
            model = Qwen3VLForConditionalGeneration(
                AutoConfig.from_pretrained(path, local_files_only=True), text_only=False
            ).eval()
        loaded = load_qwen3_vl_weights(model, path)
        assert len(loaded.loaded_keys) == 625 and loaded.skipped_keys == []
        report = {
            "loaded_keys": loaded.loaded_keys,
            "skipped_keys": loaded.skipped_keys,
            "tied_aliases": loaded.tied_aliases,
            "cases": {},
            "failures": [],
        }
        for layer in model.model.language_model.layers:
            layer.self_attn.attn.k_cache = torch.empty(
                2, 256, 8, 128, device="cuda", dtype=torch.bfloat16
            )
            layer.self_attn.attn.v_cache = torch.empty_like(
                layer.self_attn.attn.k_cache
            )
        del layer
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    text = model.model.language_model
    parameter_ref = weakref.ref(next(model.parameters()))
    torch.cuda.reset_peak_memory_stats()
    try:
        for case in CASES[:1] if fault else CASES:
            oracle = load_oracle(case)
            inputs = {
                k.removeprefix("input."): v.cuda()
                for k, v in oracle.items()
                if k.startswith("input.")
            }
            ids = inputs["input_ids"][0]
            mask = ids == model.config.image_token_id
            positions, delta = (
                hf_positions(model, inputs)
                if backend == "hf"
                else build_mrope_positions(
                    ids.tolist(),
                    inputs["mm_token_type_ids"][0].cpu(),
                    inputs["image_grid_thw"].cpu(),
                    2,
                )
            )
            positions = positions.reshape(3, -1).cuda()
            assert torch.equal(positions.cpu(), oracle["positions"].reshape(3, -1))
            assert int(torch.as_tensor(delta).item()) == int(
                oracle["rope_delta"].item()
            )
            saved = {}
            handles = capture_fusion(text, saved)
            with torch.inference_mode():
                token_embeds = text.embed_tokens(ids).detach().cpu()
                if backend == "hf":

                    def save_visual(m, a, out, saved=saved):
                        saved["main"] = out.pooler_output.detach().cpu().clone()
                        for i, value in enumerate(out.deepstack_features):
                            saved[f"deep_{i}"] = value.detach().cpu().clone()

                    handles.append(
                        model.model.visual.register_forward_hook(save_visual)
                    )
                    result = model(
                        **inputs,
                        position_ids=positions[:, None],
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    saved["logits"] = result.logits[:, -1].cpu().clone()
                    del result
                else:
                    expected = torch.load(
                        directory / f"hf-{case}.pt", weights_only=True
                    )
                    if backend == "hf_fixed_features":
                        main = expected["main"].cuda()
                        deep = [expected[f"deep_{i}"].cuda() for i in range(3)]
                    else:
                        main, deep = model.encode_images(
                            inputs["pixel_values"].to(torch.bfloat16),
                            inputs["image_grid_thw"],
                        )
                    saved["main"] = main.cpu().clone()
                    saved.update(
                        {
                            f"deep_{i}": value.cpu().clone()
                            for i, value in enumerate(deep)
                        }
                    )
                    if fault == "mask":
                        mask = torch.roll(mask, 1)
                    elif fault == "order":
                        deep = deep[::-1]
                    elif fault == "placement":
                        # Controlled wrong placement in this process only: inject before
                        # decoder0 instead of after its complete attention+MLP residual.
                        original = text.layers[0].forward

                        def misplaced(
                            p, h, original=original, mask=mask, first=deep[0]
                        ):
                            h = h.clone()
                            h[mask] = h[mask] + first
                            return original(p, h)

                        text.layers[0].forward = misplaced
                        deep = [torch.zeros_like(deep[0]), *deep[1:]]
                    prefill_context(
                        len(ids),
                        torch.arange(len(ids), device="cuda", dtype=torch.int32),
                    )
                    hidden = model(
                        ids, positions, visual_features=(main, deep), visual_mask=mask
                    )
                    if fault == "placement":
                        text.layers[0].forward = original
                        del original, misplaced
                    saved["logits"] = model.compute_logits(hidden).cpu().clone()
                    del hidden, main, deep
                for handle in handles:
                    handle.remove()
                handles.clear()
                torch.save(saved, directory / f"{backend}-{case}.pt")
                # Explicit replacement/order and preservation checks precede global comparisons.
                torch.testing.assert_close(
                    saved["fused"][mask.cpu()], saved["main"], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    saved["fused"][~mask.cpu()],
                    token_embeds[~mask.cpu()],
                    rtol=0,
                    atol=0,
                )
            record = {
                "length": len(ids),
                "visual_tokens": int(mask.sum()),
                "input_digest": tensor_digest({k: v.cpu() for k, v in inputs.items()}),
                "metrics": {},
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "free_total_bytes": torch.cuda.mem_get_info(),
            }
            if backend != "hf":
                assert saved.keys() == expected.keys()
                for key in saved:
                    record["metrics"][key] = tensor_metrics(saved[key], expected[key])
                    try:
                        assert_aligned(key, saved[key], expected[key], {key: EXACT})
                    except AssertionError as error:
                        report["failures"].append(f"{case}: {error}")
                a, b = saved["logits"][0].float(), expected["logits"][0].float()
                values, indices = torch.topk(b, 5)
                record.update(
                    top5_ids=indices.tolist(),
                    hf_top5=values.tolist(),
                    native_top5=a[indices].tolist(),
                    top1_margin=float(values[0] - values[1]),
                    argmax_equal=bool(a.argmax() == b.argmax()),
                )
                del expected
            report["cases"][case] = record
            (directory / f"{backend}-report.json").write_text(
                json.dumps(report, indent=2)
            )
            del oracle, inputs, ids, mask, positions, saved, token_embeds
            reset_context()
        del text, model
        gc.collect()
        torch.cuda.empty_cache()
        report.update(
            parameters_released=parameter_ref() is None,
            allocated_after_release=torch.cuda.memory_allocated(),
            reserved_after_release=torch.cuda.memory_reserved(),
        )
        assert report["parameters_released"]
        (directory / f"{backend}-report.json").write_text(json.dumps(report, indent=2))
        assert not report.get("failures"), "\n".join(report.get("failures", []))
    finally:
        reset_context()
        if dist.is_initialized():
            dist.destroy_process_group()
        rendezvous.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        required=True,
        choices=("hf", "hf_fixed_features", "native_encoded_features"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fault", choices=("mask", "order", "placement"))
    args = parser.parse_args()
    worker(args.worker, args.output, args.fault)
