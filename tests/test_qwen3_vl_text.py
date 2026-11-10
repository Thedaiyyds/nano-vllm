"""Task 5 text decoder checks; heavy work is explicit and GPU opt-in only."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest

import pytest

from tests.test_qwen3_vl_loader import single_rank as single_rank

pytestmark = pytest.mark.hf_reference
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task5"
)


@pytest.fixture(autouse=True)
def record_math_settings(request):
    """Record settings without changing the production arithmetic policy."""
    import torch

    yield
    request.node.user_properties.append(
        (
            "allow_bf16_reduced_precision_reduction",
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )
    )


@contextmanager
def bf16_cuda():
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            yield
    finally:
        torch.set_default_dtype(previous)


def small_config():
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        tie_word_embeddings=True,
        text_config={
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "vocab_size": 64,
            "max_position_embeddings": 4096,
            "dtype": "bfloat16",
            "rope_theta": 5000000,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
            },
            "tie_word_embeddings": True,
        },
    )


def prefill_context(length, slots=None):
    import torch
    from nanovllm.utils.context import set_context

    cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    set_context(True, cu, cu, length, length, slot_mapping=slots)


def capture_layers(text, saved):
    hooks = []

    def save(name, value):
        saved[name] = value.detach().reshape(-1, value.shape[-1]).cpu().clone()

    for index, layer in enumerate(text.layers[:3]):
        hooks.append(
            layer.register_forward_hook(
                lambda m, a, out, i=index: save(f"layer.{i}", out)
            )
        )
        hooks.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, a, i=index: save(f"residual.{i}", a[0])
            )
        )
    hooks.append(text.norm.register_forward_hook(lambda m, a, out: save("norm", out)))
    return hooks


@pytest.mark.gpu
class TextTests(unittest.TestCase):
    def tearDown(self):
        import torch
        from nanovllm.utils.context import reset_context

        reset_context()
        torch.cuda.empty_cache()

    def test_residual_order_and_tie(self):
        import torch
        from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
        from nanovllm.utils.loader import load_qwen3_vl_weights
        from tests.reference_qwen3_vl import assert_aligned
        from tests.support import run_worker

        config = small_config()
        with bf16_cuda():
            model = Qwen3VLForConditionalGeneration(config)
        self.assertIs(
            model.lm_head.weight, model.model.language_model.embed_tokens.weight
        )
        # HF and nano execute in separate processes. The HF worker exits before nano forward.
        del model
        torch.cuda.empty_cache()
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="small-", dir=EVIDENCE))
        result = run_worker(
            "tests.test_qwen3_vl_text", ["--small-reference", str(directory)]
        )
        (directory / "hf.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        reference = torch.load(
            directory / "reference.pt", map_location="cpu", weights_only=True
        )
        with bf16_cuda():
            model = Qwen3VLForConditionalGeneration(config).eval()
        report = load_qwen3_vl_weights(model, str(directory), text_only=True)
        self.assertEqual(report.skipped_keys, [])
        saved = {}
        hooks = capture_layers(model.model.language_model, saved)
        ids = reference["ids"].cuda()
        positions = reference["positions"].cuda()
        prefill_context(len(ids))
        with torch.inference_mode():
            hidden = model(ids, positions)
            logits = model.compute_logits(hidden)
        for hook in hooks:
            hook.remove()
        self.assertEqual(tuple(logits.shape), (1, 64))
        saved["logits"] = logits.cpu()
        metrics, failures = {}, []
        for name, value in saved.items():
            from tests.reference_qwen3_vl import tensor_metrics

            metrics[name] = tensor_metrics(value, reference[name])
            try:
                assert_aligned(
                    name,
                    value,
                    reference[name],
                    {name: {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}},
                )
            except AssertionError as error:
                failures.append(str(error))
        (directory / "metrics.json").write_text(
            json.dumps({"metrics": metrics, "failures": failures}, indent=2)
        )
        self.assertFalse(failures, f"{directory}: {failures}")

    def test_bfloat16_residual_rounding_witness(self):
        import torch
        from torch import nn
        from nanovllm.models.qwen3_vl import Qwen3VLDecoderLayer

        with bf16_cuda():
            layer = Qwen3VLDecoderLayer(small_config().text_config).eval()
        residual = torch.tensor(
            [1.0, 1.0, 1.0, 2.0], device="cuda", dtype=torch.bfloat16
        ).repeat(64)[None]
        attention = torch.tensor(
            [0.00390625, -0.00390625, 0.01171875, -0.0078125],
            device="cuda",
            dtype=torch.bfloat16,
        ).repeat(64)[None]

        class ConstantAttention(nn.Module):
            def forward(self, positions, hidden):
                return attention

        class Scale(nn.Module):
            def forward(self, x):
                return x * 8

        layer.self_attn = ConstantAttention()
        layer.mlp = Scale()
        rounded = residual + attention

        def norm(x):
            return (
                x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            ).to(torch.bfloat16)

        expected = rounded + norm(rounded) * 8
        delayed = rounded + norm(residual.float() + attention.float()) * 8
        self.assertFalse(torch.equal(expected, delayed))
        with torch.inference_mode():
            actual = layer(torch.zeros(3, 1, dtype=torch.long, device="cuda"), residual)
        self.assertTrue(torch.equal(actual, expected))

    def test_visual_features_require_paired_mask_before_decoder(self):
        import torch
        from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration

        with bf16_cuda():
            model = Qwen3VLForConditionalGeneration(small_config())
        with self.assertRaisesRegex(ValueError, "together"):
            model(
                torch.ones(1, dtype=torch.long, device="cuda"),
                torch.zeros(3, 1, dtype=torch.long, device="cuda"),
                visual_features=(torch.ones(1, 256, device="cuda"), []),
            )
        self.assertFalse(hasattr(model.model, "visual"))


def small_reference(directory):
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(2026)
    config = small_config().text_config
    config._attn_implementation = "flash_attention_2"
    with bf16_cuda():
        model = Qwen3VLTextModel(config).eval()
    state = {
        "model.language_model." + name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(state, str(directory / "model.safetensors"))
    saved = {}
    hooks = capture_layers(model, saved)
    ids = torch.tensor([1, 7, 3, 8, 11, 20, 6, 5, 13, 4, 2], device="cuda")
    positions = torch.arange(len(ids), device="cuda").expand(3, -1).clone()
    with torch.inference_mode():
        output = model(
            input_ids=ids[None], position_ids=positions[:, None], use_cache=False
        )
        saved["logits"] = torch.nn.functional.linear(
            output.last_hidden_state[:, -1], model.embed_tokens.weight
        ).cpu()
    for hook in hooks:
        hook.remove()
    saved["ids"], saved["positions"] = ids.cpu(), positions.cpu()
    torch.save(saved, directory / "reference.pt")


@pytest.mark.checkpoint
def test_native_text_schema_matches_real_checkpoint():
    import torch
    from transformers import AutoConfig
    from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from nanovllm.utils.loader import _audit_qwen3_vl_weights
    from tests.reference_qwen3_vl import DEFAULT_MODEL

    path = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(
            AutoConfig.from_pretrained(path, local_files_only=True)
        )
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    sources, _, report = _audit_qwen3_vl_weights(model, path, text_only=True)
    assert len(sources) == 310 and len(report.skipped_keys) == 315
    assert report.loaded_keys == []
    assert len(list(model.parameters())) == 226


CASES = ("text_short", "text_255", "text_256", "text_257")


def load_oracle(case):
    from tests.reference_qwen3_vl import validate_artifact

    root = Path(
        os.environ.get(
            "QWEN3_VL_ORACLE",
            str(Path.home() / "nano-vllm-artifacts/qwen3-vl/oracle-reviewed"),
        )
    )
    folder = root / "hf-sdpa" / case
    return validate_artifact(json.loads((folder / "manifest.json").read_text()), folder)


def full_reference(directory):
    import torch
    from transformers import Qwen3VLForConditionalGeneration as HFModel
    from tests.reference_qwen3_vl import (
        DEFAULT_MODEL,
        provenance,
        hf_positions,
        sample_cache,
    )

    path = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    identity = provenance(path)
    model, info = HFModel.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
        output_loading_info=True,
    )
    assert (
        not info["missing_keys"]
        and not info["unexpected_keys"]
        and not info["mismatched_keys"]
    )
    model = model.cuda().eval()
    backend = model.model.language_model.config._attn_implementation
    assert backend == "flash_attention_2"
    identity.update(
        backend=backend,
        dtype=str(next(model.parameters()).dtype),
        gpu=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        TORCHDYNAMO_DISABLE=os.environ.get("TORCHDYNAMO_DISABLE"),
        cases={},
    )
    for case in CASES:
        oracle = load_oracle(case)
        inputs = {
            name.removeprefix("input."): value.cuda()
            for name, value in oracle.items()
            if name.startswith("input.")
        }
        positions, delta = hf_positions(model, inputs)
        assert torch.equal(positions.cpu(), oracle["positions"])
        assert delta.item() == 0
        saved = {}
        hooks = capture_layers(model.model.language_model, saved)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result = model(
                **inputs, position_ids=positions, use_cache=True, logits_to_keep=1
            )
            saved["logits"] = result.logits[:, -1].cpu()
            sample_cache(result.past_key_values, saved, "kv")
        for hook in hooks:
            hook.remove()
        saved["positions"] = positions[:, 0].cpu()
        torch.save(saved, directory / f"hf-{case}.pt")
        free, total = torch.cuda.mem_get_info()
        identity["cases"][case] = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "free_bytes": free,
            "total_bytes": total,
        }
        del result, inputs, oracle, saved
    (directory / "hf-identity.json").write_text(json.dumps(identity, indent=2))


def native_full(directory):
    import torch
    import torch.distributed as dist
    from transformers import AutoConfig
    from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from nanovllm.layers.mrope import build_mrope_positions
    from nanovllm.utils.context import reset_context
    from nanovllm.utils.loader import load_qwen3_vl_weights
    from tests.reference_qwen3_vl import DEFAULT_MODEL

    path = os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    with tempfile.TemporaryDirectory() as rendezvous:
        dist.init_process_group(
            "gloo", init_method=Path(rendezvous, "init").as_uri(), rank=0, world_size=1
        )
        try:
            with bf16_cuda():
                model = Qwen3VLForConditionalGeneration(
                    AutoConfig.from_pretrained(path, local_files_only=True)
                ).eval()
            report = load_qwen3_vl_weights(model, path, text_only=True)
            assert len(report.loaded_keys) == 310 and len(report.skipped_keys) == 315
            assert all(key.startswith("model.visual.") for key in report.skipped_keys)
            assert (
                model.lm_head.weight is model.model.language_model.embed_tokens.weight
            )
            text = model.model.language_model
            for layer in text.layers:
                layer.self_attn.attn.k_cache = torch.empty(
                    3, 256, 8, 128, device="cuda", dtype=torch.bfloat16
                )
                layer.self_attn.attn.v_cache = torch.empty_like(
                    layer.self_attn.attn.k_cache
                )
            stats = {
                "loaded_keys": report.loaded_keys,
                "skipped_keys": report.skipped_keys,
                "tied_aliases": report.tied_aliases,
                "cases": {},
            }
            for case in CASES:
                oracle = load_oracle(case)
                ids = oracle["input.input_ids"][0]
                positions, delta = build_mrope_positions(
                    ids.tolist(),
                    oracle["input.mm_token_type_ids"][0],
                    torch.empty(0, 3, dtype=torch.long),
                    2,
                )
                assert delta == 0 and torch.equal(positions, oracle["positions"][:, 0])
                length = len(ids)
                logical = torch.arange(length, device="cuda")
                # Exercise a physical layout different from logical token order.
                pages = torch.tensor([2, 0], device="cuda")
                slots = pages[logical // 256] * 256 + logical % 256
                saved = {}
                hooks = capture_layers(text, saved)
                for layer in text.layers:
                    layer.self_attn.attn.k_cache.fill_(float("nan"))
                    layer.self_attn.attn.v_cache.fill_(float("nan"))
                prefill_context(length, slots.to(torch.int32))
                torch.cuda.reset_peak_memory_stats()
                try:
                    with torch.inference_mode():
                        hidden = model(ids.cuda(), positions.cuda())
                        saved["logits"] = model.compute_logits(hidden).cpu()
                    assert saved["logits"].shape == (1, 151936)
                    sample = oracle["prefill.kv.sample_positions"]
                    saved["kv.sample_positions"] = sample
                    for index, layer in enumerate(text.layers):
                        for kind, cache in (
                            ("keys", layer.self_attn.attn.k_cache),
                            ("values", layer.self_attn.attn.v_cache),
                        ):
                            saved[f"kv.{index}.{kind}"] = cache.view(-1, 8, 128)[
                                slots[sample.cuda()]
                            ].cpu()
                    saved["positions"] = positions
                    torch.save(saved, directory / f"nano-{case}.pt")
                    free, total = torch.cuda.mem_get_info()
                    stats["cases"][case] = {
                        "length": length,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "free_bytes": free,
                        "total_bytes": total,
                    }
                finally:
                    reset_context()
                    for hook in hooks:
                        hook.remove()
                del hidden, oracle, saved
            stats.update(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            )
            (directory / "native.json").write_text(json.dumps(stats, indent=2))
        finally:
            reset_context()
            dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.checkpoint
def test_target_checkpoint_prefill_and_physical_kv():
    import torch
    from tests.reference_qwen3_vl import tensor_metrics, assert_aligned
    from tests.support import run_worker

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="full-", dir=EVIDENCE))
    for backend in ("hf", "nano"):
        result = run_worker(
            "tests.test_qwen3_vl_text",
            ["--full-worker", backend, "--output", str(directory)],
        )
        (directory / f"{backend}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (
            f"{backend} worker: {result.stdout}{result.stderr}"
        )
    metrics, failures, sdpa_diagnostics = {}, [], {}
    for case in CASES:
        actual = torch.load(
            directory / f"nano-{case}.pt", map_location="cpu", weights_only=True
        )
        reference = torch.load(
            directory / f"hf-{case}.pt", map_location="cpu", weights_only=True
        )
        assert actual.keys() == reference.keys()
        oracle = load_oracle(case)
        metrics[case] = {}
        for name, value in actual.items():
            limits = {name: {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}}
            if value.is_floating_point():
                metrics[case][name] = tensor_metrics(value, reference[name])
            try:
                assert_aligned(name, value, reference[name], limits)
            except AssertionError as error:
                failures.append(f"{case}: {error}")
        sdpa_diagnostics[case] = tensor_metrics(
            actual["logits"], oracle["prefill.logits"][:, -1]
        )
        del actual, reference, oracle
    (directory / "comparison.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "failures": failures,
                "sdpa_diagnostics": sdpa_diagnostics,
                "standard": "finite + exact same-device HF FlashAttention2; no proposed BF16 tolerances used",
            },
            indent=2,
        )
    )
    assert not failures, f"{directory}: {failures[:8]} ({len(failures)} total)"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--small-reference", type=Path)
    parser.add_argument("--full-worker", choices=("hf", "nano"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.small_reference and (not args.full_worker or args.output is None):
        parser.error(
            "choose --small-reference DIR or --full-worker hf|nano --output DIR"
        )
    import torch

    torch.set_num_threads(1)
    if args.small_reference:
        small_reference(args.small_reference)
    elif args.full_worker == "hf":
        full_reference(args.output)
    else:
        native_full(args.output)
