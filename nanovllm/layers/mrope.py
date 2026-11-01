"""Qwen3-VL prompt positions and interleaved text RoPE (no KV state)."""

import torch
from torch import nn


def build_mrope_positions(
    token_ids: list[int],
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    """Build CPU [3,L] positions; consume each image's exact grid-sized span.

    Token values and image placeholder syntax are validated at the request
    boundary (Task 9). Here modality types determine text/image segments.
    """
    length = len(token_ids)
    if (
        mm_token_type_ids.device.type != "cpu"
        or mm_token_type_ids.dtype not in (torch.int32, torch.int64)
        or mm_token_type_ids.shape != (length,)
    ):
        raise ValueError("mm_token_type_ids must be a CPU integer tensor of shape [L]")
    if (
        image_grid_thw.device.type != "cpu"
        or image_grid_thw.dtype not in (torch.int32, torch.int64)
        or image_grid_thw.ndim != 2
        or image_grid_thw.shape[1] != 3
    ):
        raise ValueError("image_grid_thw must be a CPU integer tensor of shape [I,3]")
    if type(spatial_merge_size) is not int or spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size={spatial_merge_size!r} must be positive")
    if (image_grid_thw <= 0).any() or (
        image_grid_thw[:, 1:] % spatial_merge_size != 0
    ).any():
        raise ValueError(
            "image_grid_thw dimensions must be positive and H/W merge-divisible"
        )
    types = mm_token_type_ids.tolist()
    if any(kind not in (0, 1) for kind in types):
        raise ValueError("mm_token_type_ids supports only text=0 and image=1")

    positions = torch.empty((3, length), dtype=torch.int64, device="cpu")
    cursor = current_pos = image_index = 0
    while cursor < length:
        if types[cursor] == 0:
            end = cursor + 1
            while end < length and types[end] == 0:
                end += 1
            positions[:, cursor:end] = torch.arange(
                current_pos, current_pos + end - cursor, device="cpu"
            )
            current_pos += end - cursor
        else:
            if image_index >= len(image_grid_thw):
                raise ValueError("image_grid_thw exhausted before image tokens")
            t, h, w = image_grid_thw[image_index].tolist()
            h, w = h // spatial_merge_size, w // spatial_merge_size
            end = cursor + t * h * w
            if end > length or any(kind != 1 for kind in types[cursor:end]):
                raise ValueError("image token span does not match image_grid_thw")
            grid = torch.meshgrid(
                torch.arange(t, device="cpu"),
                torch.arange(h, device="cpu"),
                torch.arange(w, device="cpu"),
                indexing="ij",
            )
            positions[:, cursor:end] = torch.stack(grid).reshape(3, -1) + current_pos
            current_pos += max(h, w)
            image_index += 1
        cursor = end
    if image_index != len(image_grid_thw):
        raise ValueError("image_grid_thw has unconsumed rows")
    delta = int(positions.max()) + 1 - length if length else 0
    return positions, delta


class Qwen3VLRotaryEmbedding(nn.Module):
    def __init__(self, text_config):
        super().__init__()
        self.head_dim = text_config.head_dim
        parameters = text_config.rope_parameters
        self.sections = parameters["mrope_section"]
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim!r} must be positive and even")
        if (
            len(self.sections) != 3
            or any(type(size) is not int or size <= 0 for size in self.sections)
            or sum(self.sections) != self.head_dim // 2
        ):
            raise ValueError(
                f"mrope_section={self.sections!r} must cover half head_dim"
            )
        if parameters["rope_type"] != "default":
            raise ValueError(f"Unsupported rope_type={parameters['rope_type']!r}")
        inv_freq = 1.0 / (
            parameters["rope_theta"]
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate [N,heads,D] Q/K with HF's BF16 rounding order."""
        if (
            q.ndim != 3
            or k.ndim != 3
            or q.shape[0] != k.shape[0]
            or q.shape[-1] != self.head_dim
            or k.shape[-1] != self.head_dim
            or q.device != k.device
            or q.dtype != k.dtype
            or not q.is_floating_point()
        ):
            raise ValueError(
                "q/k must have matching floating dtype/device and [N,heads,head_dim]"
            )
        if (
            positions.shape != (3, q.shape[0])
            or positions.device != q.device
            or positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("positions must be integer [3,N] on the q/k device")
        freqs = positions[..., None].float() * self.inv_freq.to(
            device=q.device, dtype=torch.float32
        )
        cos, sin = freqs.cos(), freqs.sin()
        for values in (cos, sin):
            for axis in (1, 2):
                lanes = slice(axis, self.sections[axis] * 3, 3)
                values[0, :, lanes] = values[axis, :, lanes]
        cos = torch.cat((cos[0], cos[0]), -1).to(q.dtype)[:, None]
        sin = torch.cat((sin[0], sin[0]), -1).to(q.dtype)[:, None]

        def rotate(x):
            first, second = x.chunk(2, dim=-1)
            return x * cos + torch.cat((-second, first), dim=-1) * sin

        return rotate(q), rotate(k)
