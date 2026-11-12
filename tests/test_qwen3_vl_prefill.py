"""Admitted Engine Prefill, independent HF and physical KV/limit checks."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.hf_reference
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13a"
)
CASES = ("image_square", "image_rect", "image_same_layout_other_pixels")
LIMIT_CASES = ("image_limit_4095", "complete_model_text_4095")
EXACT = {"max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}


def prepared_case(case):
    from nanovllm.utils.multimodal import PreparedMultimodalPrompt
    from tests.test_qwen3_vl_text import load_oracle

    oracle = load_oracle(case)
    return PreparedMultimodalPrompt(
        oracle["input.input_ids"][0].tolist(),
        oracle["input.pixel_values"],
        oracle["input.image_grid_thw"],
        oracle["input.mm_token_type_ids"][0],
    )


def limit_inputs(case):
    """Extend fixed CPU inputs, without writing or refreshing the formal Oracle."""
    import torch
    from tests.test_qwen3_vl_text import load_oracle

    if case == LIMIT_CASES[0]:
        oracle = load_oracle("image_limit")
        inputs = {
            key: value.clone()
            for key, value in oracle.items()
            if key.startswith("input.")
        }
        padding = 4095 - inputs["input.input_ids"].shape[1]
        inputs["input.input_ids"] = torch.cat(
            [inputs["input.input_ids"], torch.full((1, padding), 9707)], dim=1
        )
        inputs["input.mm_token_type_ids"] = torch.cat(
            [
                inputs["input.mm_token_type_ids"],
                torch.zeros((1, padding), dtype=torch.long),
            ],
            dim=1,
        )
    else:
        assert case == LIMIT_CASES[1]
        inputs = {
            "input.input_ids": torch.full((1, 4095), 9707, dtype=torch.long),
            "input.mm_token_type_ids": torch.zeros((1, 4095), dtype=torch.long),
        }
    inputs["input.attention_mask"] = torch.ones_like(inputs["input.input_ids"])
    return inputs


def run_prefill_workers(directory, *, limits=False):
    for backend in ("hf", "native"):
        command = [
            sys.executable,
            "-m",
            "tests.test_qwen3_vl_prefill",
            "--worker",
            backend,
            "--output",
            str(directory),
        ]
        if limits:
            command.append("--limits")
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (directory / f"{backend}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((directory / "native-report.json").read_text())


def cpu_engine():
    from transformers import AutoConfig
    from tests.test_qwen3_vl_input import cpu_engine as make_engine, model_path
    from types import SimpleNamespace

    return make_engine(
        AutoConfig.from_pretrained(model_path(), local_files_only=True),
        SimpleNamespace(tokenizer=None),
    )


class PrefillCPUTests(TestCase):
    def test_image_waits_on_cpu_without_kv_until_admitted(self):
        from nanovllm import SamplingParams
        from tests.test_qwen3_vl_scheduler import pool_state

        engine = cpu_engine()
        before = pool_state(engine.scheduler.block_manager)
        original = prepared_case("image_square")
        engine.add_request(original, SamplingParams(max_tokens=1))
        seq = engine.scheduler.waiting[0]
        self.assertEqual(pool_state(engine.scheduler.block_manager), before)
        self.assertEqual(seq.multimodal.pixel_values.device.type, "cpu")
        self.assertNotEqual(
            seq.multimodal.pixel_values.data_ptr(), original.pixel_values.data_ptr()
        )
        self.assertEqual(seq.block_table, [])
        self.assertEqual(len(engine.scheduler.running), 0)

    def test_legacy_unpickled_sequence_runs_without_multimodal_attributes(self):
        # Legacy TP's wire representation deliberately has no multimodal fields.
        import pickle
        from types import SimpleNamespace
        import torch
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.engine.sequence import Sequence
        from nanovllm.layers.sampler import Sampler

        seq = pickle.loads(pickle.dumps(Sequence([10, 11])))
        assert not hasattr(seq, "multimodal")
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(is_multimodal=False)
        runner.rank = 0
        runner.prepare_prefill = lambda seqs: (
            torch.tensor([10, 11]),
            torch.tensor([0, 1]),
        )
        runner.prepare_sample = lambda seqs: torch.tensor([0.0])
        runner.run_model = lambda *args: torch.tensor([[1.0, 3.0]])
        runner.sampler = Sampler()
        self.assertEqual(runner.run([seq], True), [1])

    def test_chunk_and_image_decode_rejected_before_cuda_or_encoder(self):
        from types import SimpleNamespace
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.engine.sequence import Sequence

        prompt = prepared_case("image_square")
        seq = Sequence(prompt.token_ids, multimodal=prompt)
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(is_multimodal=True)
        seq.num_scheduled_tokens = len(seq) - 1
        with pytest.raises(ValueError, match="complete"):
            runner.run([seq], True)
        seq.num_scheduled_tokens = len(seq)
        with pytest.raises(ValueError, match="admitted"):
            runner.run([seq], True)
        seq.num_scheduled_tokens = 1
        with pytest.raises(NotImplementedError, match="[Dd]ecode|Task14"):
            runner.run([seq], False)


@pytest.mark.parametrize(
    "case",
    ["pixel_rows", "mask_mismatch", "missing_grid", "over_budget", "orphan_image"],
)
def test_invalid_input_keeps_queue_and_kv_unchanged(case):
    from nanovllm import SamplingParams
    from tests.test_qwen3_vl_input import invalid_prompt
    from tests.test_qwen3_vl_scheduler import pool_state

    engine = cpu_engine()
    before = pool_state(engine.scheduler.block_manager)
    with pytest.raises(ValueError):
        engine.add_request(invalid_prompt(case), SamplingParams(max_tokens=1))
    assert engine.is_finished()
    assert pool_state(engine.scheduler.block_manager) == before


@pytest.mark.gpu
@pytest.mark.checkpoint
class PrefillRoutingTests(TestCase):
    def test_real_image_prefill(self):
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="routing-", dir=EVIDENCE))
        report = run_prefill_workers(directory)
        self.assertEqual(report["loaded_keys_count"], 625)
        self.assertEqual(report["skipped_keys"], [])
        self.assertEqual(report["kv_shape"], [2, 28, 2, 256, 8, 128])
        self.assertEqual(set(report["cases"]), set(CASES))
        self.assertTrue(all(r["vision_calls"] == 1 for r in report["cases"].values()))


@pytest.mark.gpu
@pytest.mark.checkpoint
class PrefillLimitTests(TestCase):
    def test_maximum_image_and_complete_model_text_prefill(self):
        evidence = EVIDENCE.parent / "task13c"
        evidence.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="limits-", dir=evidence))
        report = run_prefill_workers(directory, limits=True)
        self.assertEqual(report["loaded_keys_count"], 625)
        self.assertEqual(report["skipped_keys"], [])
        self.assertEqual(report["kv_shape"], [2, 28, 16, 256, 8, 128])
        self.assertEqual(set(report["cases"]), set(LIMIT_CASES))
        self.assertEqual(report["cases"][LIMIT_CASES[0]]["visual_tokens"], 256)
        self.assertEqual(report["cases"][LIMIT_CASES[0]]["vision_calls"], 1)
        self.assertEqual(report["cases"][LIMIT_CASES[1]]["vision_calls"], 0)
        self.assertTrue(report["over_limit_rejected_before_allocation"])


def check_limit_rejections(engine, runner, model, report, fault):
    import torch
    from nanovllm import SamplingParams
    from tests.test_qwen3_vl_scheduler import pool_state

    image = prepared_case("image_limit")
    locations = image.mm_token_type_ids.nonzero().flatten().tolist()
    # A coherent V257/P1028 input isolates the visual budget from grid/row errors.
    end = locations[-1] + 1
    too_visual = replace(
        image,
        token_ids=image.token_ids[:end]
        + [image.token_ids[locations[0]]]
        + image.token_ids[end:],
        pixel_values=torch.cat([image.pixel_values, image.pixel_values[:4]]),
        image_grid_thw=torch.tensor([[1, 2, 514]]),
        mm_token_type_ids=torch.cat(
            [
                image.mm_token_type_ids[:end],
                torch.ones(1, dtype=torch.long),
                image.mm_token_type_ids[end:],
            ]
        ),
    )
    inputs = limit_inputs(LIMIT_CASES[0])
    over_length = replace(
        image,
        token_ids=inputs["input.input_ids"][0].tolist() + [9707],
        mm_token_type_ids=torch.cat(
            [
                inputs["input.mm_token_type_ids"][0],
                torch.zeros(1, dtype=torch.long),
            ]
        ),
    )
    before_pool = pool_state(engine.scheduler.block_manager)
    runner.kv_cache.fill_(17)
    rejected = []
    if fault == "limits":
        # Process-local diagnostic: emulate a missing Scheduler admission guard.
        engine.scheduler.add = lambda seq: engine.scheduler.waiting.append(seq)
    with (
        patch.object(
            model.model.visual, "forward", wraps=model.model.visual.forward
        ) as vision,
        patch.object(
            engine.scheduler.block_manager,
            "allocate",
            wraps=engine.scheduler.block_manager.allocate,
        ) as allocate,
    ):
        for name, prompt, tokens, reason in (
            ("V257", too_visual, 1, "visual token budget"),
            ("image_L4096_plus1", over_length, 1, "max_model_len"),
            ("text_L4096_plus1", [9707] * 4096, 1, "max_model_len"),
            ("text_L4095_plus2", [9707] * 4095, 2, "max_model_len"),
        ):
            try:
                engine.add_request(prompt, SamplingParams(max_tokens=tokens))
            except ValueError as error:
                assert reason in str(error), str(error)
                rejected.append(name)
            else:
                for seq in list(engine.scheduler.waiting):
                    engine.scheduler.abort(seq)
            assert engine.is_finished()
            assert pool_state(engine.scheduler.block_manager) == before_pool
            assert bool((runner.kv_cache == 17).all()), name
        over_limit_rejected_before_allocation = (
            len(rejected) == 4 and allocate.call_count == 0 and vision.call_count == 0
        )
    report["rejected_limits"] = rejected
    report["over_limit_rejected_before_allocation"] = (
        over_limit_rejected_before_allocation
    )
    TestCase().assertTrue(over_limit_rejected_before_allocation, report)


def worker(backend, directory, fault=None, *, limits=False):
    import torch
    from nanovllm import SamplingParams
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import get_context
    from nanovllm.utils.multimodal import PreparedMultimodalPrompt
    from tests.test_qwen3_vl_fusion import capture_fusion
    from tests.test_qwen3_vl_text import load_oracle
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
    if limits:
        assert os.environ.get("TORCHDYNAMO_DISABLE") == "1"
    identity = provenance(path)
    identity.update(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dtype="torch.bfloat16",
        language_backend="flash_attention_2",
        vision_backend="sdpa",
        TORCHDYNAMO_DISABLE=os.environ.get("TORCHDYNAMO_DISABLE"),
        tensor_parallel_size=1,
        enforce_eager=True,
        limits=limits,
        allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        source_sha256={
            str(p): digest_file(p)
            for p in (
                Path(__file__),
                Path("nanovllm/engine/model_runner.py"),
                Path("nanovllm/engine/llm_engine.py"),
            )
        },
    )
    (directory / f"{backend}-identity.json").write_text(json.dumps(identity, indent=2))
    report = {"cases": {}, "failures": []}
    engine = None
    if backend == "hf":
        from transformers import Qwen3VLForConditionalGeneration as HFModel

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
    else:
        engine = LLMEngine(
            path,
            max_model_len=4096 if limits else 512,
            max_num_batched_tokens=4096 if limits else 128,
            max_num_seqs=1,
            enforce_eager=True,
            max_visual_tokens=256 if limits else 96,
        )
        runner = engine.model_runner
        model = runner.model
        report.update(
            loaded_keys_count=len(runner.weight_load_report.loaded_keys),
            skipped_keys=runner.weight_load_report.skipped_keys,
            kv_shape=list(runner.kv_cache.shape),
            budget={
                "max_model_len": engine.config.max_model_len,
                "max_num_batched_tokens": engine.config.max_num_batched_tokens,
                "max_visual_tokens": engine.config.max_visual_tokens,
            },
            profile=runner.vl_memory_profile,
        )
        assert report["loaded_keys_count"] == 625 and report["skipped_keys"] == []
        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
        # A real allocated/deallocated text request rotates the free list, so the first
        # image is placed in block1. Subsequent images alternate physical blocks.
        dummy = Sequence([42])
        engine.scheduler.block_manager.allocate(dummy, 0)
        engine.scheduler.block_manager.deallocate(dummy)
    torch.cuda.reset_peak_memory_stats()
    try:
        cases = LIMIT_CASES if limits else CASES
        for case in cases[:1] if fault and fault != "limits" else cases:
            oracle = limit_inputs(case) if limits else load_oracle(case)
            has_image = "input.pixel_values" in oracle
            tokens = oracle["input.input_ids"][0].tolist()
            prompt = (
                PreparedMultimodalPrompt(
                    tokens,
                    oracle["input.pixel_values"],
                    oracle["input.image_grid_thw"],
                    oracle["input.mm_token_type_ids"][0],
                )
                if has_image
                else tokens
            )
            length = len(tokens)
            locations = (
                oracle["input.mm_token_type_ids"][0].nonzero().flatten().tolist()
            )
            selected = sorted({0, length - 1, *locations[:1], *locations[-1:]})
            if limits:
                selected = sorted(set(selected + [254, 255, 256, 257, 511, 512, 4093]))
                assert length == 4095
                torch.cuda.reset_peak_memory_stats()
            logits_key = (
                "image_prefill_logits" if has_image else "complete_model_text_logits"
            )
            saved = {}
            handles = capture_fusion(model.model.language_model, saved)
            if limits:
                frequencies = (
                    model.model.language_model.rotary_emb.inv_freq.expand(28, -1)
                    if backend == "hf"
                    else torch.stack(
                        [
                            layer.self_attn.rotary_emb.inv_freq
                            for layer in model.model.language_model.layers
                        ]
                    )
                )
                assert frequencies.dtype == torch.float32
                saved["text_inv_freq"] = frequencies.cpu().clone()
                del frequencies
            inputs = (
                {
                    k.removeprefix("input."): v.cuda()
                    for k, v in oracle.items()
                    if k.startswith("input.")
                }
                if backend == "hf"
                else None
            )
            with torch.inference_mode():
                if backend == "hf":
                    positions, delta = hf_positions(model, inputs)
                    saved["positions"] = positions.reshape(3, -1).cpu().clone()
                    result = model(
                        **inputs,
                        position_ids=positions,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    saved[logits_key] = result.logits[:, -1].cpu().clone()
                    saved["selected_kv"] = torch.stack(
                        [
                            torch.stack(
                                [
                                    getattr(layer, which)[0, :, selected]
                                    .permute(1, 0, 2)
                                    .cpu()
                                    .clone()
                                    for layer in result.past_key_values.layers
                                ]
                            )
                            for which in ("keys", "values")
                        ]
                    )
                    del result
                else:
                    runner.kv_cache.fill_(17)
                    original_run_model = runner.run_model
                    metadata = {}

                    def observe(ids, positions, is_prefill, saved=saved, **kwargs):
                        context = get_context()
                        saved["positions"] = positions.cpu().clone()
                        metadata["slots"] = context.slot_mapping.cpu().tolist()
                        if fault == "positions":
                            positions = positions.clone()
                            positions[1, locations] += 1
                        elif fault == "fusion":
                            kwargs["visual_mask"] = kwargs["visual_mask"].roll(1)
                        elif fault == "kv_mapping":
                            if limits:
                                context.slot_mapping.copy_(
                                    context.slot_mapping.roll(256)
                                )
                            else:
                                context.slot_mapping.sub_(int(context.slot_mapping[0]))
                        logits = original_run_model(
                            ids, positions, is_prefill, **kwargs
                        )
                        saved[logits_key] = logits.cpu().clone()
                        # Sample physical slots while the scheduler still owns the table.
                        table = engine.scheduler.running[0].block_table.copy()
                        metadata["block_table"] = table
                        physical = [table[i // 256] * 256 + i % 256 for i in selected]
                        saved["selected_kv"] = (
                            runner.kv_cache.reshape(2, 28, -1, 8, 128)[:, :, physical]
                            .cpu()
                            .clone()
                        )
                        metadata["physical_selected"] = physical
                        return logits

                    runner.run_model = observe
                    with patch.object(
                        model.model.visual, "forward", wraps=model.model.visual.forward
                    ) as vision_forward_spy:
                        engine.add_request(
                            prompt,
                            SamplingParams(
                                temperature=0, max_tokens=1, ignore_eos=True
                            ),
                        )
                        if has_image:
                            assert (
                                engine.scheduler.waiting[
                                    0
                                ].multimodal.pixel_values.device.type
                                == "cpu"
                            )
                        assert not engine.scheduler.waiting[0].block_table
                        assert vision_forward_spy.call_count == 0
                        outputs, count = engine.step()
                        # Required behavior RED reaches this real encoder-call assertion.
                        try:
                            TestCase().assertEqual(
                                vision_forward_spy.call_count, int(has_image)
                            )
                        except AssertionError as error:
                            report["failures"].append(
                                f"{case}: vision_forward_spy: {error}"
                            )
                        metadata["vision_calls"] = vision_forward_spy.call_count
                        if limits:
                            assert outputs[0][1] == [int(saved[logits_key].argmax())]
                            metadata["sampled_tokens"] = outputs[0][1]
                        assert (
                            count == length
                            and len(outputs[0][1]) == 1
                            and engine.is_finished()
                        )
                    runner.run_model = original_run_model
                    assert metadata["slots"] == [
                        metadata["block_table"][i // 256] * 256 + i % 256
                        for i in range(length)
                    ]
                    if limits:
                        assert len(metadata["block_table"]) == 16
                        assert metadata["block_table"] != list(range(16))
                        assert len(set(metadata["slots"])) == length
                        assert not engine.scheduler.block_manager.used_block_ids
                    elif case == CASES[0]:
                        assert metadata["block_table"] == [1], metadata
                    # No stale writes into any unselected/unallocated physical slots.
                    used = set(metadata["slots"])
                    unused = [
                        i
                        for i in range(runner.kv_cache.shape[2] * 256)
                        if i not in used
                    ]
                    metadata["unused_slots_unchanged"] = bool(
                        (
                            runner.kv_cache.reshape(2, 28, -1, 8, 128)[:, :, unused]
                            == 17
                        ).all()
                    )
                    if not metadata["unused_slots_unchanged"]:
                        report["failures"].append(
                            f"{case}: unexpected KV write outside admitted slots"
                        )
            for handle in handles:
                handle.remove()
            torch.save(saved, directory / f"{backend}-{case}.pt")
            record = {
                "length": length,
                "max_new_tokens": 1,
                "visual_tokens": len(locations),
                "grid": oracle["input.image_grid_thw"].tolist() if has_image else [],
                "selected_logical": selected,
                "input_digest": tensor_digest(
                    {k: v for k, v in oracle.items() if k.startswith("input.")}
                ),
                "metrics": {},
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "free_total_bytes": torch.cuda.mem_get_info(),
            }
            if backend == "native":
                record.update(metadata)
                hf_record = json.loads((directory / "hf-report.json").read_text())[
                    "cases"
                ][case]
                assert record["input_digest"] == hf_record["input_digest"]
                expected = torch.load(directory / f"hf-{case}.pt", weights_only=True)
                assert saved.keys() == expected.keys()
                for key in saved:
                    record["metrics"][key] = tensor_metrics(saved[key], expected[key])
                    try:
                        assert_aligned(key, saved[key], expected[key], {key: EXACT})
                    except AssertionError as error:
                        report["failures"].append(f"{case}: {error}")
                values, indices = expected[logits_key][0].float().topk(5)
                record.update(
                    top5_ids=indices.tolist(),
                    hf_top5=values.tolist(),
                    native_top5=saved[logits_key][0, indices].float().tolist(),
                    top1_margin=float(values[0] - values[1]),
                    argmax_equal=bool(
                        expected[logits_key].argmax() == saved[logits_key].argmax()
                    ),
                )
                if limits:
                    free, total = torch.cuda.mem_get_info()
                    stats = torch.cuda.memory_stats()
                    margin = free - (
                        stats["allocated_bytes.all.peak"]
                        - stats["allocated_bytes.all.current"]
                    )
                    record.update(measured_free_margin=margin, total_bytes=total)
                    assert margin >= engine.config.kv_cache_reserve_bytes, record
                del expected
            report["cases"][case] = record
            (directory / f"{backend}-report.json").write_text(
                json.dumps(report, indent=2)
            )
            del inputs, saved, handles
        if engine is not None and not report["failures"] and limits:
            TestCase().assertEqual(
                report["cases"][LIMIT_CASES[0]]["visual_tokens"], 256
            )
            check_limit_rejections(engine, runner, model, report, fault)
        if engine is not None and not report["failures"] and not limits:
            # Invalid/missing image fields, V/L/batched budget and chunk never run vision
            # and leave the real GPU KV tensor bitwise intact.
            from tests.test_qwen3_vl_input import invalid_prompt

            before = runner.kv_cache.clone()
            with patch.object(
                model.model.visual, "forward", wraps=model.model.visual.forward
            ) as spy:
                for invalid in (
                    invalid_prompt("pixel_rows"),
                    invalid_prompt("missing_grid"),
                    invalid_prompt("over_budget"),
                    prepared_case("image_limit"),
                    replace(prepared_case("image_square"), token_ids=[10] * 513),
                ):
                    with pytest.raises(ValueError):
                        engine.add_request(invalid, SamplingParams(max_tokens=1))
                    assert engine.is_finished()
                for ids, tokens in (([10] * 512, 1), ([10] * 2, 511), ([10] * 129, 1)):
                    with pytest.raises(ValueError):
                        engine.add_request(ids, SamplingParams(max_tokens=tokens))
                image = prepared_case("image_square")
                seq = Sequence(image.token_ids, multimodal=image)
                seq.num_scheduled_tokens = len(seq) - 1
                with pytest.raises(ValueError, match="complete"):
                    runner.run([seq], True)
                seq.num_scheduled_tokens = len(seq)
                with pytest.raises(ValueError, match="admitted"):
                    runner.run([seq], True)
                with pytest.raises(NotImplementedError, match="[Dd]ecode|Task14"):
                    runner.run([seq], False)
                assert spy.call_count == 0
                assert torch.equal(runner.kv_cache, before)
            del before
            report["invalid_no_encoder_or_kv_side_effect"] = True
            # The public Engine path must fail explicitly when a caller asks for
            # a second image token; it must never silently use text-only positions.
            with patch.object(
                model.model.visual, "forward", wraps=model.model.visual.forward
            ) as spy:
                engine.add_request(
                    prepared_case("image_square"),
                    SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
                )
                engine.step()
                before = runner.kv_cache.clone()
                with pytest.raises(NotImplementedError, match="[Dd]ecode|Task14"):
                    engine.step()
                assert spy.call_count == 1 and torch.equal(runner.kv_cache, before)
                assert engine.is_finished()
                del before
            report["public_image_decode_guard"] = True
            engine.exit()
        report.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            free_total_bytes=torch.cuda.mem_get_info(),
        )
        (directory / f"{backend}-report.json").write_text(json.dumps(report, indent=2))
        assert not report["failures"], "\n".join(report["failures"])
    finally:
        if engine is not None:
            engine.exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("hf", "native"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fault", choices=("positions", "fusion", "kv_mapping", "limits")
    )
    parser.add_argument("--limits", action="store_true")
    args = parser.parse_args()
    worker(args.worker, args.output, args.fault, limits=args.limits)
