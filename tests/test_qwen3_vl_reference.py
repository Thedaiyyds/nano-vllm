"""Independent checks for reference tooling and actual GPU attention kernels."""

import pytest

pytestmark = pytest.mark.hf_reference


def test_comparison_reports_hand_calculated_error():
    import torch
    from tests.reference_qwen3_vl import assert_aligned

    limits = {"x": {"max_abs": 1.0, "mean_abs": 0.5, "rel_l2": 0.5}}
    result = assert_aligned(
        "x", torch.tensor([1.0, 3.0]), torch.tensor([1.0, 2.0]), limits
    )
    assert result["max_abs"] == 1.0
    assert result["mean_abs"] == 0.5
    assert result["rel_l2"] == pytest.approx(1 / (5**0.5))


@pytest.mark.parametrize(
    "failure", ["shape", "dtype", "nan", "inf", "missing", "excess", "relative"]
)
def test_comparison_rejects_invalid_or_excess_error(failure):
    import torch
    from tests.reference_qwen3_vl import assert_aligned

    expected = torch.tensor([1.0, 2.0])
    actual = expected.clone()
    limits = {"x": {"max_abs": 0.01, "mean_abs": 0.01, "rel_l2": 0.01}}
    if failure == "shape":
        actual = torch.ones(3)
    elif failure == "dtype":
        actual = actual.double()
    elif failure in ("nan", "inf"):
        actual[0] = float(failure)
    elif failure == "missing":
        limits = {}
    else:
        actual[0] += 1
        if failure == "relative":
            limits["x"].update(max_abs=10.0, mean_abs=10.0)
    with pytest.raises(AssertionError):
        assert_aligned("x", actual, expected, limits)


def test_integer_positions_require_exact_equality():
    import torch
    from tests.reference_qwen3_vl import assert_aligned

    with pytest.raises(AssertionError):
        assert_aligned("positions", torch.tensor([0, 2]), torch.tensor([0, 1]), {})


@pytest.mark.gpu
@pytest.mark.parametrize("length", [255, 256, 257])
def test_real_paged_attention_kernels(length):
    import json
    import os
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    from nanovllm.layers.attention import store_kvcache
    from tests.reference_qwen3_vl import tensor_metrics

    torch.manual_seed(2026 + length)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    # Qwen3-VL GQA head geometry; inputs are already post-RoPE Q/K.
    q = torch.randn(length, 16, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(length, 8, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn_like(k) * 0.1
    kc = torch.full((4, 256, 8, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
    vc = torch.full_like(kc, float("nan"))
    table = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    physical = torch.tensor(
        [int(table[0, i // 256]) * 256 + i % 256 for i in range(length)],
        device="cuda",
        dtype=torch.int32,
    )
    store_kvcache(k, v, kc, vc, physical)
    # A -1 padding slot must not overwrite any actual data.
    store_kvcache(
        k[:1], v[:1], kc, vc, torch.tensor([-1], device="cuda", dtype=torch.int32)
    )
    gathered_k = kc.view(-1, 8, 128)[physical.long()]
    gathered_v = vc.view(-1, 8, 128)[physical.long()]
    assert torch.equal(k, gathered_k) and torch.equal(v, gathered_v)
    assert torch.isnan(kc[1]).all() and torch.isnan(vc[3]).all()
    cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    prefill = flash_attn_varlen_func(q, k, v, cu, cu, length, length, causal=True)
    paged = flash_attn_varlen_func(
        q, kc, vc, cu, cu, length, length, causal=True, block_table=table
    )
    decode = flash_attn_with_kvcache(
        q[-1:].unsqueeze(1),
        kc,
        vc,
        cache_seqlens=torch.tensor([length], device="cuda", dtype=torch.int32),
        block_table=table,
        causal=True,
    )[:, 0]
    k_heads = k.float().transpose(0, 1).repeat_interleave(2, dim=0).unsqueeze(0)
    v_heads = v.float().transpose(0, 1).repeat_interleave(2, dim=0).unsqueeze(0)
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(
            q.float().transpose(0, 1).unsqueeze(0), k_heads, v_heads, is_causal=True
        )[0].transpose(0, 1)
    # Include both probability conversion (P -> BF16 for PV) and output rounding.
    # This controlled-input engineering budget is not a universal error theorem.
    scores = (
        q.float().transpose(0, 1).unsqueeze(0) @ k_heads.transpose(-2, -1) / 128**0.5
    )
    causal = torch.ones(length, length, dtype=torch.bool, device="cuda").tril()
    probabilities = scores.masked_fill(~causal, float("-inf")).softmax(-1)
    weighted_abs_v = (probabilities @ v_heads.abs())[0].transpose(0, 1)
    bound = (
        torch.finfo(torch.bfloat16).eps / 2 * (ref.abs() + weighted_abs_v)
        + 8 * length * torch.finfo(torch.float32).eps * v.float().abs().max()
    )
    records = {}
    for name, actual, expected, limits in [
        ("prefill", prefill, ref, bound),
        ("paged_prefill", paged, ref, bound),
        ("decode", decode, ref[-1:], bound[-1:]),
    ]:
        assert ((actual.float() - expected).abs() <= limits).all(), name
        records[name] = tensor_metrics(actual.float(), expected)
    root = Path(
        os.environ.get("QWEN3_VL_ARTIFACTS", "/tmp/nano-vllm-qwen3-vl-validation")
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / f"kernel-{length}.json").write_text(json.dumps(records, indent=2))


@pytest.mark.gpu
def test_real_bf16_matmul_precision_budget():
    import torch
    from tests.reference_qwen3_vl import tensor_metrics

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    result = a @ b
    ref = a.float() @ b.float()
    gamma = (
        128
        * torch.finfo(torch.float32).eps
        / (1 - 128 * torch.finfo(torch.float32).eps)
    )
    bound = (
        gamma * (a.float().abs() @ b.float().abs())
        + torch.finfo(torch.bfloat16).eps / 2 * ref.abs()
    )
    assert ((result.float() - ref).abs() <= bound).all()
    assert tensor_metrics(result.float(), ref)["rel_l2"] < 0.004


@pytest.mark.parametrize(
    "field", ["model_revision", "transformers_revision", "tensor_file_sha256"]
)
def test_artifact_identity_and_integrity_are_required(tmp_path, field):
    import torch
    from tests.support import digest_file
    from tests.reference_qwen3_vl import MODEL_REVISION, HF_REVISION, validate_artifact

    torch.save({"x": torch.ones(2)}, tmp_path / "tensors.pt")
    manifest = {
        "model_revision": MODEL_REVISION,
        "transformers_revision": HF_REVISION,
        "tensor_file_sha256": digest_file(tmp_path / "tensors.pt"),
        "tensor_metadata": {"x": {"shape": [2], "dtype": "torch.float32"}},
    }
    manifest[field] = "wrong-value"
    with pytest.raises(AssertionError):
        validate_artifact(manifest, tmp_path)


@pytest.mark.checkpoint
@pytest.mark.parametrize(
    "case,expected",
    [
        ("text_short", 24),
        ("text_255", 255),
        ("text_256", 256),
        ("text_257", 257),
        ("image_square", [1, 16, 16]),
        ("image_rect", [1, 16, 24]),
        ("image_same_layout_other_pixels", [1, 16, 16]),
        ("image_limit", [1, 32, 32]),
        ("image_candy", [1, 16, 16]),
        ("image_cat", [1, 16, 24]),
    ],
)
def test_cpu_cases_have_exact_processor_contract(case, expected):
    import os
    from tests.reference_qwen3_vl import build_case, DEFAULT_MODEL

    inputs = build_case(case, os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL))
    assert all(t.device.type == "cpu" for t in inputs.values())
    if isinstance(expected, int):
        assert inputs["input_ids"].shape == (1, expected)
        assert inputs["mm_token_type_ids"].count_nonzero().item() == 0
    else:
        assert inputs["image_grid_thw"].tolist() == [expected]
        assert inputs["pixel_values"].shape == (expected[1] * expected[2], 1536)
        assert (inputs["input_ids"] == 151655).sum().item() == expected[1] * expected[
            2
        ] // 4


@pytest.mark.parametrize("field", ["backend", "case"])
def test_calibration_rejects_wrong_experiment_identity(field):
    from tests.reference_qwen3_vl import validate_run_identity

    record = {"case": "text_short", "backend": "eager"}
    record[field] = "sdpa" if field == "backend" else "text_255"
    with pytest.raises(AssertionError):
        validate_run_identity(record, "text_short", "hf-eager")


def test_checkpoint_content_cannot_reuse_revision_label():
    import json
    from tests.reference_qwen3_vl import CASES, validate_checkpoint_fingerprints

    files = json.loads((CASES.parent / "checkpoint.json").read_text())["files_sha256"]
    files["model.safetensors"] = "different-weights-same-architecture"
    with pytest.raises(AssertionError):
        validate_checkpoint_fingerprints(files)


def test_cache_pair_positions_must_match():
    import torch
    from tests.reference_qwen3_vl import paired_cache_metrics

    values = {
        "decode.kv.sample_positions": torch.tensor([0, 1, 2]),
        "recomputed.kv.sample_positions": torch.tensor([0, 1, 3]),
    }
    with pytest.raises(AssertionError):
        paired_cache_metrics(values)


def test_cache_pair_error_is_not_hidden_as_two_independent_components():
    import torch
    from tests.reference_qwen3_vl import paired_cache_metrics

    values = {
        "decode.kv.sample_positions": torch.tensor([0, 1, 2]),
        "recomputed.kv.sample_positions": torch.tensor([0, 1, 2]),
        "decode.step_kv.0.keys": torch.tensor([1.0]),
        "recomputed.step_kv.0.keys": torch.tensor([2.0]),
    }
    result = paired_cache_metrics(values)
    assert result.get("decode.step_kv.0.keys", {}).get("max_abs", 0) == 1.0


def test_all_runs_missing_required_components_cannot_form_baseline(tmp_path):
    import json
    import torch
    from tests.support import digest_file
    from tests.reference_qwen3_vl import (
        CASES,
        MODEL_REVISION,
        HF_REVISION,
        validate_artifact,
    )

    torch.save({"x": torch.ones(2)}, tmp_path / "tensors.pt")
    manifest = {
        "model_revision": MODEL_REVISION,
        "transformers_revision": HF_REVISION,
        "case": "text_short",
        "model_files_sha256": json.loads(
            (CASES.parent / "checkpoint.json").read_text()
        )["files_sha256"],
        "tensor_file_sha256": digest_file(tmp_path / "tensors.pt"),
        "tensor_metadata": {"x": {"shape": [2], "dtype": "torch.float32"}},
    }
    with pytest.raises(AssertionError, match="required tensors"):
        validate_artifact(manifest, tmp_path)
