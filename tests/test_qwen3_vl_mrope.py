"""Three-axis positions and text RoPE, independent of model weights/runner."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import pytest

pytestmark = pytest.mark.hf_reference


def text_config():
    from transformers import AutoConfig
    from tests.reference_qwen3_vl import DEFAULT_MODEL

    return AutoConfig.from_pretrained(
        os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL), local_files_only=True
    ).text_config


def hf_positions(ids, types, grids, merge=2):
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    # These HF methods only use the config; no model or weights are constructed.
    owner = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=merge))
    )
    owner.get_vision_position_ids = lambda *a, **kw: (
        Qwen3VLModel.get_vision_position_ids(owner, *a, **kw)
    )
    positions, delta = Qwen3VLModel.get_rope_index(
        owner, torch.tensor([ids]), types[None], image_grid_thw=grids
    )
    return positions[:, 0], delta.item()


def hf_rotated(config, positions, q, k):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    rope = Qwen3VLTextRotaryEmbedding(config).to(q.device)
    cos, sin = rope(q, positions[:, None])
    result = apply_rotary_pos_emb(q[None], k[None], cos, sin, unsqueeze_dim=2)
    return tuple(value[0] for value in result)


class MRopeTests(unittest.TestCase):
    def test_interleaved_axes(self):
        import torch
        from nanovllm.layers.mrope import Qwen3VLRotaryEmbedding

        config = text_config()
        positions = torch.tensor([[1, 3], [7, 11], [17, 23]])
        q = torch.cat((torch.ones(2, 2, 64), torch.zeros(2, 2, 64)), -1)
        k = q[:, :1].clone()
        actual = Qwen3VLRotaryEmbedding(config)(positions, q, k)
        expected = hf_rotated(config, positions, q, k)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        # Independent lane check: contiguous T/H/W sections would fail this.
        inv = 1.0 / (5000000 ** (torch.arange(0, 128, 2).float() / 128))
        for lane in range(64):
            axis = 1 if lane in range(1, 60, 3) else 2 if lane in range(2, 60, 3) else 0
            angle = positions[axis].float() * inv[lane]
            self.assertTrue(torch.equal(actual[0][:, 0, lane], angle.cos()))
            self.assertTrue(torch.equal(actual[0][:, 0, lane + 64], angle.sin()))

    def test_prompt_delta(self):
        import torch
        from nanovllm.layers.mrope import build_mrope_positions

        ids = [10, 151652] + [151655] * 6 + [151653, 11, 12]
        types = torch.tensor([0, 0] + [1] * 6 + [0] * 3)
        grids = torch.tensor([[1, 4, 6]])
        positions, delta = build_mrope_positions(ids, types, grids, 2)
        expected, hf_delta = hf_positions(ids, types, grids)
        self.assertEqual(tuple(positions.shape), (3, 11))
        self.assertEqual(delta, -3)
        self.assertEqual(delta, hf_delta)
        self.assertTrue(torch.equal(positions, expected))
        self.assertEqual(
            positions.tolist(),
            [
                [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7],
                [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7],
                [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7],
            ],
        )
        self.assertEqual(positions.dtype, torch.int64)
        self.assertEqual(positions.device.type, "cpu")

    def test_text_and_multiple_image_segments(self):
        import torch
        from nanovllm.layers.mrope import build_mrope_positions

        for types, grids in [
            ([0] * 257, []),
            ([0] + [1] * 6 + [0] * 2 + [1] * 4 + [0], [[1, 4, 6], [1, 4, 4]]),
        ]:
            ids = [10] * len(types)
            types = torch.tensor(types)
            grids = torch.tensor(grids, dtype=torch.long).reshape(-1, 3)
            actual, delta = build_mrope_positions(ids, types, grids, 2)
            expected, expected_delta = hf_positions(ids, types, grids)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(delta, expected_delta)
        self.assertEqual(
            build_mrope_positions(
                [1, 2],
                torch.zeros(2, dtype=torch.long),
                torch.empty(0, 3, dtype=torch.long),
                2,
            )[1],
            0,
        )

    def test_adjacent_images_follow_grid_boundaries(self):
        import torch
        from nanovllm.layers.mrope import build_mrope_positions

        positions, delta = build_mrope_positions(
            [151655] * 3,
            torch.ones(3, dtype=torch.long),
            torch.tensor([[1, 2, 4], [1, 2, 2]]),
            2,
        )
        self.assertEqual(positions.tolist(), [[0, 0, 2], [0, 0, 2], [0, 1, 2]])
        self.assertEqual(delta, 0)


@pytest.mark.parametrize(
    "case",
    [
        "text_short",
        "text_255",
        "text_256",
        "text_257",
        "image_square",
        "image_rect",
        "image_same_layout_other_pixels",
        "image_limit",
        "image_candy",
        "image_cat",
    ],
)
@pytest.mark.checkpoint
def test_positions_match_persistent_oracle(case):
    import torch
    from nanovllm.layers.mrope import build_mrope_positions
    from tests.reference_qwen3_vl import validate_artifact

    root = Path(
        os.environ.get(
            "QWEN3_VL_ORACLE",
            str(Path.home() / "nano-vllm-artifacts/qwen3-vl/oracle-reviewed"),
        )
    )
    folder = root / "hf-sdpa" / case
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        values = validate_artifact(
            json.loads((folder / "manifest.json").read_text()), folder
        )
        ids = values["input.input_ids"][0].tolist()
        positions, delta = build_mrope_positions(
            ids,
            values["input.mm_token_type_ids"][0],
            values.get("input.image_grid_thw", torch.empty(0, 3, dtype=torch.long)),
            2,
        )
        assert torch.equal(positions, values["positions"][:, 0])
        assert delta == values["rope_delta"].item()
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "types,grid,merge",
    [
        ([1] * 6, [], 2),
        ([0], [[1, 2, 2]], 2),
        ([1] * 5, [[1, 4, 6]], 2),
        ([1] * 7, [[1, 4, 6]], 2),
        ([1], [[-1, 2, 2]], 2),
        ([1], [[1, 0, 2]], 2),
        ([1], [[1, 3, 2]], 2),
        ([1], [[1, 2, 2]], 0),
        ([2], [], 2),
    ],
)
def test_invalid_grid_consumption_and_dimensions(types, grid, merge):
    import torch
    from nanovllm.layers.mrope import build_mrope_positions

    with pytest.raises(ValueError):
        build_mrope_positions(
            [1] * len(types),
            torch.tensor(types),
            torch.tensor(grid, dtype=torch.long).reshape(-1, 3),
            merge,
        )


def test_modality_length_mismatch():
    import torch
    from nanovllm.layers.mrope import build_mrope_positions

    with pytest.raises(ValueError, match="mm_token_type_ids"):
        build_mrope_positions(
            [1, 2], torch.tensor([0]), torch.empty(0, 3, dtype=torch.long), 2
        )


@pytest.mark.gpu
class MRopeGPUTests(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def record_gpu_metadata(self, request):
        import torch

        torch.cuda.reset_peak_memory_stats()
        yield
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        request.node.user_properties.extend(
            [
                ("gpu", torch.cuda.get_device_name()),
                ("torch", torch.__version__),
                ("cuda", torch.version.cuda),
                ("dtype", "torch.bfloat16"),
                ("backend", "PyTorch eager; same-device HF component reference"),
                ("TORCHDYNAMO_DISABLE", os.environ.get("TORCHDYNAMO_DISABLE")),
                ("peak_allocated_bytes", torch.cuda.max_memory_allocated()),
                ("peak_reserved_bytes", torch.cuda.max_memory_reserved()),
                ("free_bytes_after", free),
                ("total_bytes", total),
                ("comparison", "finite and torch.equal; zero numerical difference"),
            ]
        )

    def test_bfloat16_noncontiguous_qk(self):
        import torch
        from nanovllm.layers.mrope import Qwen3VLRotaryEmbedding

        torch.manual_seed(2026)
        config = text_config()
        rope = Qwen3VLRotaryEmbedding(config).cuda()
        for length in (1, 11, 255, 256, 257):
            positions = torch.stack(
                (
                    torch.arange(length),
                    torch.arange(length) + 7,
                    torch.arange(length) + 17,
                )
            ).cuda()
            q = torch.randn(length, 16, 256, device="cuda", dtype=torch.bfloat16)[
                ..., ::2
            ]
            k = torch.randn(length, 8, 256, device="cuda", dtype=torch.bfloat16)[
                ..., ::2
            ]
            self.assertFalse(q.is_contiguous())
            before_q, before_k = q.clone(), k.clone()
            actual = rope(positions, q, k)
            expected = hf_rotated(config, positions, q, k)
            for a, b in zip(actual, expected):
                self.assertEqual(a.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue(torch.equal(a, b), f"length={length}")
            self.assertTrue(torch.equal(q, before_q))
            self.assertTrue(torch.equal(k, before_k))

    def test_decode_positions(self):
        import torch
        from nanovllm.layers.mrope import Qwen3VLRotaryEmbedding, build_mrope_positions

        positions, delta = build_mrope_positions(
            [1] * 11,
            torch.tensor([0, 0] + [1] * 6 + [0] * 3),
            torch.tensor([[1, 4, 6]]),
            2,
        )
        config = text_config()
        rope = Qwen3VLRotaryEmbedding(config).cuda()
        q = torch.ones(1, 16, 128, dtype=torch.bfloat16, device="cuda")
        k = q[:, :8]
        length = positions.shape[1]
        self.assertEqual([length + delta] * 3, [8, 8, 8])
        for step in range(32):
            decode_pos = torch.full((3, 1), length + delta + step, device="cuda")
            actual = rope(decode_pos, q, k)
            expected = hf_rotated(config, decode_pos, q, k)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
