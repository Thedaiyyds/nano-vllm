"""Qwen3-VL vision primitives and segmented noncausal ViT blocks."""

from math import isqrt

import torch
from torch import nn
from torch.nn import functional as F


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.in_channels = vision_config.in_channels
        self.embed_dim = vision_config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel, stride=kernel, bias=True
        )

    def forward(self, pixel_values):
        width = self.in_channels * self.temporal_patch_size * self.patch_size**2
        if (
            pixel_values.ndim != 2
            or pixel_values.shape[1] != width
            or not pixel_values.shape[0]
        ):
            raise ValueError("pixel_values must be nonempty [P,patch_width]")
        patches = pixel_values.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(patches.to(dtype=self.proj.weight.dtype)).reshape(
            -1, self.embed_dim
        )


def _validate_grid(grid, merge):
    if type(merge) is not int or merge <= 0:
        raise ValueError("merge_size must be a positive integer")
    if (
        not isinstance(grid, torch.Tensor)
        or grid.ndim != 2
        or grid.shape[1] != 3
        or not len(grid)
        or grid.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("grid_thw must be nonempty integer [I,3]")
    if (grid <= 0).any() or (grid[:, 1:] % merge).any():
        raise ValueError("grid dimensions must be positive and H/W merge-divisible")


def vision_position_ids(grid_thw, merge_size):
    """Spatial coordinates in Processor's block-row/block-col/in-row/in-col order."""
    _validate_grid(grid_thw, merge_size)
    positions = []
    for t, h, w in grid_thw.tolist():
        row, col = torch.meshgrid(
            torch.arange(h, device=grid_thw.device),
            torch.arange(w, device=grid_thw.device),
            indexing="ij",
        )
        block_shape = (h // merge_size, merge_size, w // merge_size, merge_size)
        positions.append(
            torch.stack(
                [
                    axis.reshape(block_shape).permute(0, 2, 1, 3).reshape(-1)
                    for axis in (row, col)
                ],
                dim=-1,
            ).repeat(t, 1)
        )
    return torch.cat(positions)


def interpolate_vision_positions(weight, grid_thw, merge_size):
    """HF four-tap bilinear sum: FP32 weights, align_corners=True, border clamp."""
    _validate_grid(grid_thw, merge_size)
    if weight.ndim != 2 or not weight.is_floating_point() or not weight.shape[0]:
        raise ValueError("position weight must be floating [square_size,hidden_size]")
    side = isqrt(weight.shape[0])
    if side * side != weight.shape[0]:
        raise ValueError("position weight must represent a square table")
    coordinates = vision_position_ids(grid_thw.to(weight.device), merge_size)
    offsets = torch.arange(2, device=weight.device)
    output = []
    cursor = 0
    for t, h, w in grid_thw.tolist():
        size = t * h * w
        denominator = torch.tensor([max(h - 1, 1), max(w - 1, 1)], device=weight.device)
        source = coordinates[cursor : cursor + size].float() * (side - 1) / denominator
        floor = source.floor()
        taps = (floor.long()[..., None] + offsets).clamp(0, side - 1)
        coefficients = (
            1 - (source[..., None] - floor[..., None] - offsets).abs()
        ).clamp(min=0)
        indices = (taps[:, 0, :, None] * side + taps[:, 1, None, :]).reshape(-1, 4)
        weights = (coefficients[:, 0, :, None] * coefficients[:, 1, None, :]).reshape(
            -1, 4
        )
        output.append((weight[indices] * weights[:, :, None]).sum(1))
        cursor += size
    return torch.cat(output)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        if vision_config.rope_parameters["rope_type"] != "axial":
            raise ValueError("Vision RoPE requires rope_type='axial'")
        self.head_dim = (
            getattr(vision_config, "head_dim", None)
            or vision_config.hidden_size // vision_config.num_attention_heads
        )
        if self.head_dim <= 0 or self.head_dim % 4:
            raise ValueError("Vision head_dim must be positive and divisible by four")
        spatial_dim = self.head_dim // 2
        inverse = 1.0 / (
            vision_config.rope_parameters["rope_theta"]
            ** (
                torch.arange(0, spatial_dim, 2, dtype=torch.float32, device="cpu")
                / spatial_dim
            )
        )
        # Immutable FP32 values survive parent_module.to(dtype=BF16), unlike a cast buffer.
        self._frequencies = tuple(inverse.tolist())

    def forward(self, position_ids):
        if (
            position_ids.ndim != 2
            or position_ids.shape[1] != 2
            or position_ids.dtype not in (torch.int32, torch.int64)
            or (position_ids < 0).any()
        ):
            raise ValueError("vision position_ids must be nonnegative integer [P,2]")
        inverse = torch.tensor(
            self._frequencies, device=position_ids.device, dtype=torch.float32
        )
        angles = position_ids[..., None].float() * inverse
        cos, sin = angles.cos(), angles.sin()
        return tuple(
            torch.cat([values[:, 0], values[:, 1], values[:, 0], values[:, 1]], dim=-1)
            for values in (cos, sin)
        )


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Rotate vision Q/K in FP32 and restore each input's original dtype."""
    if (
        q.ndim != 3
        or k.ndim != 3
        or q.shape[0] != k.shape[0]
        or q.shape[-1] != k.shape[-1]
        or q.shape[-1] % 2
        or cos.shape != (q.shape[0], q.shape[-1])
        or sin.shape != cos.shape
    ):
        raise ValueError("Expected matching [P,heads,D] Q/K and [P,D] cos/sin")

    def rotate(x):
        first, second = x.float().chunk(2, dim=-1)
        return (
            x.float() * cos[:, None].float()
            + torch.cat((-second, first), -1) * sin[:, None].float()
        ).to(x.dtype)

    return rotate(q), rotate(k)


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.dim = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        if self.num_heads <= 0 or self.dim <= 0 or self.dim % self.num_heads:
            raise ValueError("vision hidden_size must be divisible by num_heads")
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.dim:
            raise ValueError("vision hidden_states must be [P,hidden_size]")
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.numel() < 2
        ):
            raise ValueError("cu_seqlens must be integer cumulative segment boundaries")
        boundaries = cu_seqlens.tolist()
        lengths = [end - start for start, end in zip(boundaries, boundaries[1:])]
        if (
            boundaries[0] != 0
            or boundaries[-1] != hidden_states.shape[0]
            or any(length <= 0 for length in lengths)
        ):
            raise ValueError("cu_seqlens must cover every patch in nonempty segments")
        count = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(count, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        splits = [
            tensor.transpose(0, 1).unsqueeze(0).split(lengths, dim=2)
            for tensor in (q, k, v)
        ]
        output = [
            F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False, scale=self.head_dim**-0.5
            )
            .transpose(1, 2)
            .contiguous()
            for q, k, v in zip(*splits)
        ]
        return self.proj(torch.cat(output, dim=1).reshape(count, self.dim).contiguous())


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.linear_fc1 = nn.Linear(
            vision_config.hidden_size, vision_config.intermediate_size, bias=True
        )
        self.linear_fc2 = nn.Linear(
            vision_config.intermediate_size, vision_config.hidden_size, bias=True
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, hidden_states):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, vision_config):
        super().__init__()
        self.norm1 = nn.LayerNorm(vision_config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(vision_config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(vision_config)
        self.mlp = Qwen3VLVisionMLP(vision_config)

    def forward(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, position_embeddings
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))
