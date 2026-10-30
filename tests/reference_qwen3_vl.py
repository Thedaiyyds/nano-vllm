"""Pinned HF oracle. All heavy imports and GPU work are explicit, never collection-time."""

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

from tests.support import digest_file

MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
HF_REVISION = "2711ad22c82105e240a2877b92d082fe2857936f"
DEFAULT_MODEL = f"/home/thedai/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/{MODEL_REVISION}"
CASES = Path(__file__).parent / "fixtures/qwen3_vl/cases.json"


def tensor_metrics(actual, expected):
    import torch

    assert actual.shape == expected.shape, "shape mismatch"
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), "NaN/Inf"
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    d = (a - b).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_l2": (d.norm() / max(b.norm().item(), 1e-12)).item(),
    }


def assert_aligned(name, actual, expected, tolerances):
    import torch

    assert actual.shape == expected.shape, f"{name}: shape mismatch"
    assert actual.dtype == expected.dtype, f"{name}: dtype mismatch"
    if not expected.is_floating_point():
        assert torch.equal(actual, expected), (
            f"{name}: integers/masks must match exactly"
        )
        return {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}
    assert name in tolerances, f"{name}: missing frozen threshold"
    metrics = tensor_metrics(actual, expected)
    for key, value in metrics.items():
        limit = tolerances[name][key]
        assert isinstance(limit, (int, float)) and 0 <= limit < float("inf")
        assert value <= limit, f"{name}: {key}={value} exceeds {limit}"
    return metrics


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for key, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_case(name: str, model_path: str) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    spec = json.loads(CASES.read_text())[name]
    if "length" in spec:
        ids = torch.full((1, spec["length"]), 9707, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": torch.zeros_like(ids),
        }
    if "file" in spec:
        with Image.open(Path(__file__).resolve().parents[1] / spec["file"]) as original:
            image = original.convert("RGB").resize(tuple(spec["size"]))
    else:
        image = Image.new("RGB", tuple(spec["size"]), tuple(spec["color"]))
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    result = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look carefully: "},
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the picture briefly."},
                ],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"return_mm_token_type_ids": True},
    )
    inputs = {k: v.cpu() for k, v in result.items() if isinstance(v, torch.Tensor)}
    grid = inputs["image_grid_thw"]
    assert (grid[:, 0] == 1).all() and (grid[:, 1:] % 2 == 0).all()
    count = int((inputs["input_ids"] == 151655).sum())
    assert count == int((grid.prod(-1) // 4).sum()) <= 256
    assert torch.equal(inputs["mm_token_type_ids"] == 1, inputs["input_ids"] == 151655)
    return inputs


def provenance(model_path):
    from urllib.parse import unquote, urlparse

    dist = importlib.metadata.distribution("transformers")
    source = json.loads(dist.read_text("direct_url.json"))
    hf_root = Path(unquote(urlparse(source["url"]).path))
    head = subprocess.check_output(
        ["git", "-C", str(hf_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert head == HF_REVISION, f"HF revision drift: {head}"

    # Local research comments are allowed, executable AST changes are not.
    class StripDocs(ast.NodeTransformer):
        def visit(self, node):
            node = super().visit(node)
            if (
                isinstance(
                    node,
                    (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and node.body
            ):
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    node.body.pop(0)
            return node

    for rel in [
        "src/transformers/models/qwen3_vl/modeling_qwen3_vl.py",
        "src/transformers/models/qwen3_vl/processing_qwen3_vl.py",
        "src/transformers/vision_utils.py",
    ]:
        original = subprocess.check_output(
            ["git", "-C", str(hf_root), "show", f"{head}:{rel}"], text=True
        )
        assert ast.dump(StripDocs().visit(ast.parse(original))) == ast.dump(
            StripDocs().visit(ast.parse((hf_root / rel).read_text()))
        ), rel
    model = Path(model_path)
    files = {p.name: digest_file(p) for p in sorted(model.iterdir()) if p.is_file()}
    validate_checkpoint_fingerprints(files)
    diff = subprocess.check_output(
        ["git", "-C", str(hf_root), "diff", "HEAD", "--", "src/transformers"]
    )
    return {
        "model_revision": MODEL_REVISION,
        "transformers_revision": head,
        "model_files_sha256": files,
        "hf_dirty_sha256": hashlib.sha256(diff).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "flash-attn", "triton", "numpy")
        },
    }


def register_capture_hooks(model, saved):
    """Observe real HF operations; wrappers delegate to the original computation."""
    hooks = []

    def save(name, output):
        if isinstance(output, (tuple, list)):
            for i, value in enumerate(output):
                save(f"{name}.{i}", value)
        else:
            saved[name] = output.detach().cpu().clone()

    def output_hook(name):
        return lambda module, args, result: save(name, result)

    visual = model.model.visual
    hooks.append(visual.patch_embed.register_forward_hook(output_hook("vision.patch")))
    hooks.append(
        visual.rotary_pos_emb.register_forward_hook(output_hook("vision.rope"))
    )
    hooks.append(
        visual.blocks[0].register_forward_pre_hook(
            lambda m, a: save("vision.patch_plus_absolute", a[0])
        )
    )
    for index in (0, 5, 11, 17, 23):
        hooks.append(
            visual.blocks[index].register_forward_hook(
                output_hook(f"vision.layer.{index}")
            )
        )
    hooks.append(visual.merger.register_forward_hook(output_hook("vision.main")))
    for index, merger in enumerate(visual.deepstack_merger_list):
        hooks.append(
            merger.register_forward_hook(output_hook(f"vision.deepstack.{index}"))
        )
    text = model.model.language_model
    hooks.append(
        text.register_forward_pre_hook(
            lambda m, a, kw: save("text.fused_embedding", kw["inputs_embeds"]),
            with_kwargs=True,
        )
    )
    hooks.append(text.rotary_emb.register_forward_hook(output_hook("text.rope")))
    hooks.append(text.norm.register_forward_hook(output_hook("text.final_norm")))
    for index in range(3):
        hooks.append(
            text.layers[index].register_forward_hook(
                output_hook(f"text.before_deepstack.{index}")
            )
        )

        def after_pre(module, args, index=index):
            save(f"text.after_deepstack.{index}", args[0])

        hooks.append(text.layers[index + 1].register_forward_pre_hook(after_pre))
        hooks.append(
            text.layers[index].self_attn.q_norm.register_forward_hook(
                output_hook(f"text.q_norm.{index}")
            )
        )
        hooks.append(
            text.layers[index].self_attn.k_norm.register_forward_hook(
                output_hook(f"text.k_norm.{index}")
            )
        )
    from transformers.models.qwen3_vl import modeling_qwen3_vl as impl

    original = impl.apply_rotary_pos_emb
    counter = [0]

    def rotary(*args, **kwargs):
        result = original(*args, **kwargs)
        index = counter[0]
        counter[0] += 1
        if index < 3:
            save(f"text.rotated_qk.{index}", result)
        return result

    original_interp = impl.get_vision_interpolation_indices_and_weights
    original_vision_rotary = impl.apply_rotary_pos_emb_vision
    interpolation = {}
    vision_counter = [0]

    def interp(*args, **kwargs):
        indices, weights = original_interp(*args, **kwargs)
        interpolation["weights"] = weights
        save("vision.interpolation_indices", indices)
        save("vision.interpolation_weights", weights)
        return indices, weights

    def absolute(module, args, output):
        # Delegate embedding to HF, then evaluate exactly the same HF four-tap sum.
        positions = (output * interpolation["weights"][:, :, None]).sum(1)
        save("vision.absolute_fp32", positions)
        save("vision.absolute_bf16", positions.to(output.dtype))

    def vision_rotary(*args, **kwargs):
        result = original_vision_rotary(*args, **kwargs)
        if vision_counter[0] == 0:
            save("vision.rotated_qk", result)
        vision_counter[0] += 1
        return result

    hooks.append(visual.pos_embed.register_forward_hook(absolute))

    @contextmanager
    def traces():
        with (
            patch.object(impl, "apply_rotary_pos_emb", rotary),
            patch.object(impl, "get_vision_interpolation_indices_and_weights", interp),
            patch.object(impl, "apply_rotary_pos_emb_vision", vision_rotary),
        ):
            yield

    return hooks, traces()


def sample_cache(cache, saved, prefix):
    import torch

    length = cache.get_seq_length()
    positions = sorted({0, length // 2, length - 1})
    saved[f"{prefix}.sample_positions"] = torch.tensor(positions)
    for i, layer in enumerate(cache.layers):
        for name in ("keys", "values"):
            tensor = getattr(layer, name)
            saved[f"{prefix}.{i}.{name}"] = (
                tensor[0, :, positions, :].transpose(0, 1).contiguous().cpu()
            )


def last_step_cache(cache, saved, prefix):
    for layer_id, layer in enumerate(cache.layers):
        for name in ("keys", "values"):
            key = f"{prefix}.{layer_id}.{name}"
            saved.setdefault(key, []).append(
                getattr(layer, name)[0, :, -1, :].detach().cpu()
            )


def stack_step_cache(saved):
    import torch

    for key, value in list(saved.items()):
        if ".step_kv." in key and isinstance(value, list):
            saved[key] = torch.stack(value)


def hf_positions(model, inputs):
    return model.model.get_rope_index(
        inputs["input_ids"],
        inputs["mm_token_type_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        attention_mask=inputs["attention_mask"],
    )


def cached_run(model, inputs, positions, delta, teacher=None, capture=None):
    import torch

    length = inputs["input_ids"].shape[1]
    model.model.rope_deltas = None
    result = model(**inputs, position_ids=positions, use_cache=True, logits_to_keep=1)
    logits = [result.logits[0, -1].detach().cpu()]
    cache = result.past_key_values
    if capture is not None:
        sample_cache(cache, capture, "prefill.kv")
        last_step_cache(cache, capture, "decode.step_kv")
    tokens = (
        [int(result.logits[0, -1].argmax())] if teacher is None else teacher.tolist()
    )
    for step in range(1, 32):
        token = torch.tensor([[tokens[step - 1]]], device="cuda")
        pos = (
            torch.full((3, 1, 1), length + step - 1, dtype=torch.long, device="cuda")
            + delta
        )
        result = model(
            input_ids=token,
            attention_mask=torch.ones(
                (1, length + step), dtype=torch.long, device="cuda"
            ),
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = result.past_key_values
        assert cache.get_seq_length() == length + step
        logits.append(result.logits[0, -1].detach().cpu())
        if capture is not None:
            last_step_cache(cache, capture, "decode.step_kv")
        if teacher is None:
            tokens.append(int(result.logits[0, -1].argmax()))
    if capture is not None:
        sample_cache(cache, capture, "decode.kv")
    return torch.stack(logits), torch.tensor(tokens, dtype=torch.long)


def capture_reference(case: str, backend: str, output_dir: Path) -> Path:
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    assert torch.cuda.is_available(), "BLOCKED: CUDA unavailable"
    model_path = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    identity = provenance(model_path)
    inputs_cpu = build_case(case, model_path)
    identity["input_digest"] = tensor_digest(inputs_cpu)
    identity["oracle_sha256"] = digest_file(Path(__file__))
    identity["cases_sha256"] = digest_file(CASES)
    out = output_dir / case
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=backend,
        local_files_only=True,
        output_loading_info=True,
    )
    assert (
        not info["missing_keys"]
        and not info["unexpected_keys"]
        and not info["mismatched_keys"]
    ), info
    model = model.to("cuda").eval()
    inputs = {k: v.cuda() for k, v in inputs_cpu.items()}
    saved = {f"input.{k}": v for k, v in inputs_cpu.items()}
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        positions, delta = hf_positions(model, inputs)
        saved["positions"], saved["rope_delta"] = positions.cpu(), delta.cpu()
        # Always derive the continuation using the same SDPA HF implementation.
        model.set_attn_implementation("sdpa")
        _, teacher = cached_run(model, inputs, positions, delta)
        saved["teacher_ids"] = teacher
        model.set_attn_implementation(backend)
        # Capture only the first Prefill; hooks are removed before Decode.
        hooks, traced_rotary = register_capture_hooks(model, saved)
        with traced_rotary:
            model.model.rope_deltas = None
            result = model(
                **inputs, position_ids=positions, use_cache=True, logits_to_keep=1
            )
            saved["prefill.logits"] = result.logits.detach().cpu()
        for hook in hooks:
            hook.remove()
        del result
        cached_logits, _ = cached_run(model, inputs, positions, delta, teacher, saved)
        saved["decode.cached_logits"] = cached_logits
        assert torch.equal(saved["prefill.logits"][0, -1], cached_logits[0]), (
            "hooks altered Prefill"
        )
        saved["step_input_positions"] = torch.arange(
            inputs["input_ids"].shape[1] - 1, inputs["input_ids"].shape[1] + 31
        )
        # Reuse HF vision output only; the entire language sequence is recomputed each step.
        vision_kwargs = {}
        if "pixel_values" in inputs:
            vision_kwargs["mm_encoder_outputs"] = {
                "image": model.model.get_image_features(
                    inputs["pixel_values"], inputs["image_grid_thw"]
                )
            }
        recomputed = []
        for step in range(32):
            full = {k: v for k, v in inputs.items() if k != "pixel_values"}
            full["input_ids"] = torch.cat(
                [inputs["input_ids"], teacher[:step].cuda().view(1, -1)], dim=1
            )
            full["attention_mask"] = torch.ones_like(full["input_ids"])
            full["mm_token_type_ids"] = torch.cat(
                [
                    inputs["mm_token_type_ids"],
                    torch.zeros((1, step), dtype=torch.long, device="cuda"),
                ],
                dim=1,
            )
            full_pos, _ = hf_positions(model, full)
            expected_pos = torch.cat(
                [
                    positions,
                    torch.arange(
                        inputs["input_ids"].shape[1],
                        inputs["input_ids"].shape[1] + step,
                        device="cuda",
                    )
                    .view(1, 1, -1)
                    .expand(3, 1, -1)
                    + delta,
                ],
                dim=-1,
            )
            assert torch.equal(full_pos, expected_pos)
            result = model(
                **full,
                **vision_kwargs,
                position_ids=full_pos,
                use_cache=True,
                logits_to_keep=1,
            )
            assert result.past_key_values.get_seq_length() == full["input_ids"].shape[1]
            recomputed.append(result.logits[0, -1].cpu())
            last_step_cache(result.past_key_values, saved, "recomputed.step_kv")
            if step == 31:
                sample_cache(result.past_key_values, saved, "recomputed.kv")
            del result
        saved["decode.recomputed_logits"] = torch.stack(recomputed)
        stack_step_cache(saved)
    assert "text.rotated_qk.0.0" in saved and "text.after_deepstack.2" in saved
    if "pixel_values" in inputs:
        assert all(f"vision.deepstack.{i}" in saved for i in range(3))
        assert "vision.absolute_fp32" in saved and "vision.rotated_qk.0" in saved
    relation = tensor_metrics(
        saved["decode.cached_logits"], saved["decode.recomputed_logits"]
    )
    a, b = saved["decode.cached_logits"], saved["decode.recomputed_logits"]
    relation["top1_agreement"] = int((a.argmax(-1) == b.argmax(-1)).sum())
    relation["steps"] = 32
    relation["top5_overlap_mean"] = (
        sum(
            len(set(x.tolist()) & set(y.tolist())) / 5
            for x, y in zip(a.topk(5).indices, b.topk(5).indices)
        )
        / 32
    )
    relation["cached_top1_margin_min"] = (
        (a.topk(2).values[:, 0] - a.topk(2).values[:, 1]).min().item()
    )
    identity.update(
        case=case,
        backend=backend,
        device=torch.cuda.get_device_name(),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        teacher_digest=tensor_digest({"teacher_ids": teacher}),
        cached_vs_recomputed=relation,
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
        free_bytes=torch.cuda.mem_get_info()[0],
        tensor_metadata={
            k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in saved.items()
        },
    )
    out.mkdir(parents=True)
    torch.save(saved, out / "tensors.pt")
    identity["tensor_file_sha256"] = digest_file(out / "tensors.pt")
    (out / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    return out / "manifest.json"


def validate_artifact(manifest, root):
    import torch

    assert manifest["model_revision"] == MODEL_REVISION, "model revision"
    assert manifest["transformers_revision"] == HF_REVISION, "HF revision"
    assert digest_file(root / "tensors.pt") == manifest["tensor_file_sha256"], (
        "artifact hash"
    )
    values = torch.load(root / "tensors.pt", map_location="cpu", weights_only=True)
    assert set(values) == set(manifest["tensor_metadata"]), "tensor inventory"
    for name, value in values.items():
        assert list(value.shape) == manifest["tensor_metadata"][name]["shape"], name
        assert str(value.dtype) == manifest["tensor_metadata"][name]["dtype"], name
        assert torch.isfinite(value).all(), name
    validate_checkpoint_fingerprints(manifest["model_files_sha256"])
    assert required_tensor_names(manifest["case"]).issubset(values), (
        "missing required tensors"
    )
    length = values["input.input_ids"].shape[1]
    assert values["positions"].shape == (3, 1, length)
    assert values["decode.cached_logits"].shape == (32, 151936)
    assert values["decode.recomputed_logits"].shape == (32, 151936)
    assert torch.equal(
        values["step_input_positions"], torch.arange(length - 1, length + 31)
    )
    for layer in range(28):
        for kind in ("keys", "values"):
            for prefix in ("decode.step_kv", "recomputed.step_kv"):
                assert values[f"{prefix}.{layer}.{kind}"].shape == (32, 8, 128)
            assert torch.equal(
                values[f"prefill.kv.{layer}.{kind}"][0],
                values[f"decode.kv.{layer}.{kind}"][0],
            ), "cached prefix changed"
    return values


def calibrate(root: Path):
    import torch

    cases = list(json.loads(CASES.read_text()))
    output = {
        "status": "PROPOSED_FROZEN_PENDING_USER_ACCEPTANCE",
        "model_revision": MODEL_REVISION,
        "transformers_revision": HF_REVISION,
        "method": "2 * maximum observed HF variation + one outward ULP of SDPA reference; not a formal bound for arbitrary kernels",
        "zero_variation_policy": "One-ULP rounding budget is an independently motivated proposal requiring written user acceptance, not a claim of bitwise cross-kernel equivalence",
        "cases": {},
    }
    for case in cases:
        records, manifests = [], []
        for run in ("hf-sdpa", "hf-eager", "hf-repeat"):
            folder = root / run / case
            manifest = json.loads((folder / "manifest.json").read_text())
            validate_run_identity(manifest, case, run)
            values = validate_artifact(manifest, folder)
            if manifests:
                for key in (
                    "case",
                    "model_revision",
                    "transformers_revision",
                    "model_files_sha256",
                    "hf_dirty_sha256",
                    "packages",
                    "input_digest",
                    "teacher_digest",
                    "device",
                    "driver",
                    "oracle_sha256",
                    "cases_sha256",
                ):
                    assert manifest[key] == manifests[0][key], f"{case}: {key} drift"
                assert set(values) == set(records[0]), f"{case}: incomplete components"
            manifests.append(manifest)
            records.append(values)
        cache_pairs = [paired_cache_metrics(r) for r in records]
        reference, eager, repeat = records
        entries = {}
        for name, value in reference.items():
            if not value.is_floating_point():
                assert torch.equal(value, eager[name]) and torch.equal(
                    value, repeat[name]
                ), name
                continue
            observations = [
                tensor_metrics(eager[name], value),
                tensor_metrics(repeat[name], value),
            ]
            if name == "decode.cached_logits":
                observations.extend(
                    tensor_metrics(r["decode.recomputed_logits"], r[name])
                    for r in records
                )
            if name in cache_pairs[0]:
                observations.extend(pair[name] for pair in cache_pairs)
            ulp = (
                torch.nextafter(
                    value.abs(), torch.full_like(value, float("inf"))
                ).double()
                - value.abs().double()
            ).abs()
            floor = {
                "max_abs": ulp.max().item(),
                "mean_abs": ulp.mean().item(),
                "rel_l2": ulp.norm().item() / max(value.double().norm().item(), 1e-12),
            }
            limits = {k: 2 * max(m[k] for m in observations) + floor[k] for k in floor}
            assert all(0 <= v < float("inf") for v in limits.values()), name
            entries[name] = {
                **limits,
                "observations": observations,
                "one_ulp_budget": floor,
            }
        output["cases"][case] = {
            "limits": entries,
            "cached_vs_recomputed_kv": cache_pairs,
            "cached_vs_recomputed": [m["cached_vs_recomputed"] for m in manifests],
            "source_manifest_sha256": [
                digest_file(root / run / case / "manifest.json")
                for run in ("hf-sdpa", "hf-eager", "hf-repeat")
            ],
        }
    path = root / "tolerances.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen thresholds: {path}")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)


def validate_run_identity(manifest, case, run):
    assert manifest["case"] == case, "case/experiment mismatch"
    expected = {"hf-sdpa": "sdpa", "hf-eager": "eager", "hf-repeat": "sdpa"}
    assert manifest["backend"] == expected[run], "backend/experiment mismatch"


def validate_checkpoint_fingerprints(files):
    trusted = json.loads((CASES.parent / "checkpoint.json").read_text())
    assert trusted["model_revision"] == MODEL_REVISION
    for name, checksum in trusted["files_sha256"].items():
        assert files.get(name) == checksum, (
            f"pinned checkpoint content mismatch: {name}"
        )


def paired_cache_metrics(values):
    import torch

    assert torch.equal(
        values["decode.kv.sample_positions"], values["recomputed.kv.sample_positions"]
    ), "cache sample positions"
    result = {}
    for key, value in values.items():
        if (
            key.startswith(("decode.kv.", "decode.step_kv."))
            and value.is_floating_point()
        ):
            other = key.replace("decode.", "recomputed.", 1)
            assert other in values, f"missing cache counterpart {other}"
            result[key] = tensor_metrics(value, values[other])
    return result


def required_tensor_names(case):
    names = {
        "input.input_ids",
        "input.attention_mask",
        "input.mm_token_type_ids",
        "positions",
        "rope_delta",
        "teacher_ids",
        "step_input_positions",
        "prefill.logits",
        "text.fused_embedding",
        "text.final_norm",
        "text.rope.0",
        "text.rope.1",
        "decode.cached_logits",
        "decode.recomputed_logits",
    }
    for prefix in ("prefill.kv", "decode.kv", "recomputed.kv"):
        names.add(f"{prefix}.sample_positions")
        for layer in range(28):
            names.update(f"{prefix}.{layer}.{kind}" for kind in ("keys", "values"))
    for prefix in ("decode.step_kv", "recomputed.step_kv"):
        for layer in range(28):
            names.update(f"{prefix}.{layer}.{kind}" for kind in ("keys", "values"))
    for layer in range(3):
        names.update(
            f"text.{kind}.{layer}"
            for kind in ("before_deepstack", "after_deepstack", "q_norm", "k_norm")
        )
        names.update(f"text.rotated_qk.{layer}.{index}" for index in (0, 1))
    if case.startswith("image_"):
        names.update(
            {
                "input.pixel_values",
                "input.image_grid_thw",
                "vision.patch",
                "vision.patch_plus_absolute",
                "vision.rope.0",
                "vision.rope.1",
                "vision.main",
                "vision.absolute_fp32",
                "vision.absolute_bf16",
                "vision.interpolation_indices",
                "vision.interpolation_weights",
                "vision.rotated_qk.0",
                "vision.rotated_qk.1",
            }
        )
        names.update(f"vision.layer.{i}" for i in (0, 5, 11, 17, 23))
        names.update(f"vision.deepstack.{i}" for i in range(3))
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(json.loads(CASES.read_text())))
    parser.add_argument("--backend", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibrate", type=Path)
    args = parser.parse_args()
    if args.calibrate:
        calibrate(args.calibrate)
    else:
        if not args.case or not args.output:
            parser.error("--case and --output are required")
        print(capture_reference(args.case, args.backend, args.output))


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    main()
