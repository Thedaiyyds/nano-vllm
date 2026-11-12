"""Joint VL activation profiling and full-context KV capacity (Task13b)."""

import hashlib
import os
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.hf_reference
EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / ".superpowers/sdd/2026-10-29-qwen3-vl-implementation/task13b"
)


@pytest.mark.gpu
@pytest.mark.checkpoint
class MemoryTests(TestCase):
    def test_joint_profile_and_allocated_kv_margin(self):
        import torch
        from nanovllm.config import Config
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
        from tests.test_qwen3_vl_runner import MODEL

        config = Config(str(MODEL), max_num_seqs=1, enforce_eager=True)
        grids = []
        output_refs = []
        lengths = []
        patches = []
        pre_cases = []
        original_run = ModelRunner.run_model
        original_info = torch.cuda.mem_get_info

        def run_model(owner, *args, **kwargs):
            lengths.append(args[0].numel())
            result = original_run(owner, *args, **kwargs)
            self.assertTrue(torch.isfinite(result).all().item())
            output_refs.append(weakref.ref(result))
            stats = torch.cuda.memory_stats()
            pre_cases.append(
                dict(
                    length=args[0].numel(),
                    logits_shape=list(result.shape),
                    finite=True,
                    peak_allocated=stats["allocated_bytes.all.peak"],
                    peak_reserved=stats["reserved_bytes.all.peak"],
                )
            )
            return result

        def info(*args, **kwargs):
            self.assertTrue(
                all(ref() is None for ref in output_refs),
                "live profile output would cancel activation reserve",
            )
            return original_info(*args, **kwargs)

        original = Qwen3VLForConditionalGeneration.encode_images

        def encode(model, pixels, grid):
            grids.append(grid.tolist())
            patches.append(pixels.shape[0])
            return original(model, pixels, grid)

        runner = None
        try:
            with (
                patch.object(Qwen3VLForConditionalGeneration, "encode_images", encode),
                patch.object(ModelRunner, "run_model", run_model),
                patch("torch.cuda.mem_get_info", info),
            ):
                runner = ModelRunner(config, 0, None)
            # Effective initial RED: old warmup executes no vision at all.
            self.assertEqual(len(grids), 3)
            self.assertEqual(lengths, [4096] * 4)
            self.assertEqual(patches, [1024] * 3)
            self.assertEqual(grids, [[[1, 32, 32]], [[1, 2, 512]], [[1, 512, 2]]])
            self.assertEqual(
                runner.kv_cache.shape[2], (config.max_model_len + 255) // 256
            )
            profile = runner.vl_memory_profile
            self.assertTrue(all(type(v) is int for v in profile.values()))
            allocated = []
            for grid in [None, (32, 32), (2, 512), (512, 2)]:
                torch.cuda.reset_peak_memory_stats()
                with patch.object(ModelRunner, "run_model", run_model):
                    runner._run_vl_memory_case(grid)
                self.assertTrue(all(ref() is None for ref in output_refs))
                torch.cuda.synchronize()
                stats = torch.cuda.memory_stats()
                free, total = torch.cuda.mem_get_info()
                increment = (
                    stats["allocated_bytes.all.peak"]
                    - stats["allocated_bytes.all.current"]
                )
                margin = free - increment
                self.assertGreaterEqual(margin, config.kv_cache_reserve_bytes)
                allocated.append(
                    dict(
                        grid=grid,
                        length=pre_cases[-1]["length"],
                        logits_shape=pre_cases[-1]["logits_shape"],
                        finite=True,
                        peak_allocated=stats["allocated_bytes.all.peak"],
                        peak_reserved=stats["reserved_bytes.all.peak"],
                        free_after=free,
                        measured_free_margin=margin,
                    )
                )
            from tests.reference_qwen3_vl import provenance

            identity = provenance(str(MODEL))
            identity.update(
                dtype=str(config.model_dtype),
                vision_backend="sdpa",
                language_backend="flash_attention_2",
                dynamo_disabled=os.environ.get("TORCHDYNAMO_DISABLE"),
                tensor_parallel_size=1,
                eager=config.enforce_eager,
                loaded_keys_count=len(runner.weight_load_report.loaded_keys),
                skipped_keys=list(runner.weight_load_report.skipped_keys),
            )
            self.assertEqual(identity["loaded_keys_count"], 625)
            self.assertEqual(identity["skipped_keys"], [])
            source = {
                str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in [
                    Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1]
                    / "nanovllm/engine/model_runner.py",
                ]
            }
            EVIDENCE.mkdir(parents=True, exist_ok=True)
            (EVIDENCE / "resources.json").write_text(
                json.dumps(
                    dict(
                        identity=identity,
                        source_sha256=source,
                        profile=profile,
                        pre_kv_cases=pre_cases[:4],
                        allocated_kv=allocated,
                        kv_shape=list(runner.kv_cache.shape),
                        grids=grids,
                        gpu=torch.cuda.get_device_name(),
                        total_bytes=total,
                    ),
                    indent=2,
                )
            )
        finally:
            if runner is not None:
                runner.exit()


class BudgetIntegrationTests(TestCase):
    def runner(self):
        from nanovllm.engine.model_runner import ModelRunner
        from nanovllm.layers.attention import Attention
        from tests.test_qwen3_vl_runner import budget_config

        runner = ModelRunner.__new__(ModelRunner)
        runner.config = budget_config(
            gpu_memory_utilization=1.0, kv_cache_reserve_bytes=1000
        )
        runner.block_size = 256
        attention = Attention(16, 128, 1.0, 8)
        runner.model = SimpleNamespace(
            model=SimpleNamespace(
                language_model=SimpleNamespace(modules=lambda: [attention] * 28)
            )
        )
        runner.vl_memory_profile = dict(
            peak_allocated=500,
            peak_reserved=700,
            free_after=2000,
            reserve_bytes=runner.config.kv_cache_reserve_bytes,
            activation_bytes=400,
            total_bytes=3000,
            available_bytes=-1,
        )
        return runner

    def test_negative_net_budget_is_rejected_without_clamping(self):
        from nanovllm.engine.model_runner import select_kv_block_count

        runner = self.runner()
        seen = []

        def select(config, available):
            seen.append(available)
            return select_kv_block_count(config, available)

        with (
            patch("nanovllm.engine.model_runner.select_kv_block_count", select),
            patch("torch.cuda.mem_get_info", return_value=(1399, 3000)),
            patch(
                "torch.cuda.memory_stats",
                return_value={
                    "allocated_bytes.all.peak": 100,
                    "allocated_bytes.all.current": 0,
                },
            ),
        ):
            with self.assertRaisesRegex(ValueError, r"nonnegative.*-1"):
                runner._allocate_vl_kv_cache()
        self.assertEqual(seen, [-1])

    def test_net_budget_is_consumed_once_and_insufficient_capacity_rejected(self):
        from nanovllm.engine.model_runner import kv_block_bytes, select_kv_block_count

        runner = self.runner()
        required = 16 * kv_block_bytes(
            runner.config.text_config, 256, runner.config.model_dtype
        )
        for budget in [required - 1, required]:
            runner.vl_memory_profile["available_bytes"] = budget

            def select(config, available):
                self.assertEqual(available, budget)
                count = select_kv_block_count(config, available)
                self.assertEqual(count, 16)
                raise RuntimeError("capacity accepted before GPU allocation")

            with (
                patch("nanovllm.engine.model_runner.select_kv_block_count", select),
                patch(
                    "torch.cuda.mem_get_info",
                    return_value=(budget + 1400, budget + 3000),
                ),
                patch(
                    "torch.cuda.memory_stats",
                    return_value={
                        "allocated_bytes.all.peak": 100,
                        "allocated_bytes.all.current": 0,
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError if budget < required else RuntimeError,
                    "insufficient|capacity accepted",
                ):
                    runner._allocate_vl_kv_cache()

    def test_profile_subtracts_activation_and_reserve_once(self):
        from nanovllm.engine.model_runner import ModelRunner

        runner = self.runner()
        runner.config.max_num_batched_tokens = 4096
        runner.config.max_visual_tokens = 256
        runner.config.hf_config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=64),
            image_token_id=80,
            vision_start_token_id=81,
            vision_end_token_id=82,
            video_token_id=83,
        )
        runner._run_vl_memory_case = lambda grid: None
        stats = {
            "allocated_bytes.all.peak": 500,
            "allocated_bytes.all.current": 100,
            "reserved_bytes.all.peak": 700,
        }
        with (
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.memory_stats", return_value=stats),
            patch("torch.cuda.mem_get_info", return_value=(2000, 3000)),
        ):
            result = ModelRunner.profile_vl_memory(runner)
        self.assertEqual(result["activation_bytes"], 400)
        self.assertEqual(
            result["available_bytes"], 2000 - 400 - runner.config.kv_cache_reserve_bytes
        )
        self.assertEqual(result["reserve_bytes"], runner.config.kv_cache_reserve_bytes)
