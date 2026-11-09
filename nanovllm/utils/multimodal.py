"""CPU request input boundary for Qwen3-VL."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass
class PreparedMultimodalPrompt:
    token_ids: list[int]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    mm_token_type_ids: torch.Tensor

    def copy(self) -> "PreparedMultimodalPrompt":
        """Own CPU storage, including when the caller retains tensor views."""
        tensors = (self.pixel_values, self.image_grid_thw, self.mm_token_type_ids)
        if any(
            not isinstance(t, torch.Tensor) or t.device.type != "cpu" for t in tensors
        ):
            raise ValueError("Prepared inputs must contain CPU tensors")
        return PreparedMultimodalPrompt(
            self.token_ids.copy(), *(tensor.detach().clone() for tensor in tensors)
        )


def _visual_limit(value):
    if type(value) is not int or not 1 <= value <= 256:
        raise ValueError("max_visual_tokens must be an integer in [1,256]")


def _token_ids(token_ids, hf_config):
    vocabulary = hf_config.text_config.vocab_size
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            type(token) is not int or not 0 <= token < vocabulary for token in token_ids
        )
    ):
        raise ValueError("token_ids must be a nonempty list of valid integer IDs")


def validate_text_tokens(token_ids, hf_config):
    """Reject VL control tokens without their image data at the engine boundary."""
    _token_ids(token_ids, hf_config)
    reserved = {
        hf_config.image_token_id,
        hf_config.video_token_id,
        hf_config.vision_start_token_id,
        hf_config.vision_end_token_id,
    }
    if reserved.intersection(token_ids):
        raise ValueError("Text input contains orphan image/video placeholders")


def prepare_multimodal(
    messages: list[dict], processor, *, max_visual_tokens: int = 256
) -> PreparedMultimodalPrompt:
    """Prepare one static PIL image under content['image'], using HF on CPU.

    Paths, URLs, encoded sources and their aliases must be decoded by the caller;
    rejecting them here prevents HF's RGB conversion from hiding animation.
    """
    from PIL import Image

    _visual_limit(max_visual_tokens)
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be one nonempty conversation")
    images = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Batched conversations are unsupported")
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("Message content must be text or a list")
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in ("text", "image"):
                raise ValueError(
                    "Only text and static image content is supported; no video"
                )
            if item["type"] == "image":
                images += 1
                if any(
                    key in item for key in ("url", "path", "base64")
                ) or not isinstance(item.get("image"), Image.Image):
                    raise ValueError(
                        "Image content requires a decoded PIL image under 'image'; source aliases are unsupported"
                    )
                if getattr(item.get("image"), "n_frames", 1) != 1:
                    raise ValueError("Only static images are supported")
    if images != 1:
        raise ValueError("Exactly one image is required")

    image_processor = processor.image_processor
    factor = image_processor.patch_size * image_processor.merge_size
    max_pixels = max_visual_tokens * factor**2
    output = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "return_mm_token_type_ids": True,
            "padding": False,
            "images_kwargs": {
                "size": {
                    "shortest_edge": min(
                        image_processor.size["shortest_edge"], max_pixels
                    ),
                    "longest_edge": max_pixels,
                }
            },
        },
    )
    if any(key in output for key in ("pixel_values_videos", "video_grid_thw")):
        raise ValueError("Video input is unsupported")
    ids, mask, types = (
        output.get(key) for key in ("input_ids", "attention_mask", "mm_token_type_ids")
    )
    if any(
        not isinstance(t, torch.Tensor) or t.device.type != "cpu"
        for t in (ids, mask, types)
    ):
        raise ValueError("Processor must return CPU IDs, attention and modality masks")
    if (
        ids.ndim != 2
        or ids.shape[0] != 1
        or mask.shape != ids.shape
        or types.shape != ids.shape
    ):
        raise ValueError("Processor must return a single unpadded sequence")
    if ids.dtype not in (torch.int32, torch.int64) or not (mask == 1).all():
        raise ValueError("Padding and non-integer IDs are unsupported")
    prepared = PreparedMultimodalPrompt(
        ids[0].tolist(),
        output.get("pixel_values"),
        output.get("image_grid_thw"),
        types[0],
    )
    config = SimpleNamespace(
        image_token_id=processor.image_token_id,
        video_token_id=processor.video_token_id,
        vision_start_token_id=processor.vision_start_token_id,
        vision_end_token_id=processor.vision_end_token_id,
        text_config=SimpleNamespace(vocab_size=len(processor.tokenizer)),
        vision_config=SimpleNamespace(
            spatial_merge_size=image_processor.merge_size,
            in_channels=3,
            patch_size=image_processor.patch_size,
            temporal_patch_size=image_processor.temporal_patch_size,
        ),
    )
    validate_multimodal(prepared, config, max_visual_tokens=max_visual_tokens)
    return prepared.copy()


def validate_multimodal(
    prompt: PreparedMultimodalPrompt, hf_config, *, max_visual_tokens: int
) -> None:
    """Validate before queueing or allocating KV. Empty image fields mean text."""
    _visual_limit(max_visual_tokens)
    if not isinstance(prompt, PreparedMultimodalPrompt):
        raise ValueError("Expected PreparedMultimodalPrompt")
    _token_ids(prompt.token_ids, hf_config)
    pixels, grid, types = (
        prompt.pixel_values,
        prompt.image_grid_thw,
        prompt.mm_token_type_ids,
    )
    if any(
        not isinstance(t, torch.Tensor) or t.device.type != "cpu"
        for t in (pixels, grid, types)
    ):
        raise ValueError("Prepared inputs must contain CPU tensors")
    integer_types = (torch.int32, torch.int64)
    if grid.dtype not in integer_types or grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError("image_grid_thw must be integer [I,3]")
    if types.dtype not in integer_types or types.shape != (len(prompt.token_ids),):
        raise ValueError("mm_token_type_ids must be integer [L]")
    vision = hf_config.vision_config
    # Sequence's config-free interface targets Qwen3-VL's 2x2 merge contract.
    if vision.spatial_merge_size != 2:
        raise ValueError("Only Qwen3-VL spatial_merge_size=2 is supported")
    width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    if (
        pixels.ndim != 2
        or pixels.shape[1] != width
        or not pixels.is_floating_point()
        or not torch.isfinite(pixels).all()
    ):
        raise ValueError("pixel_values must be finite floating [P,patch_width]")
    ids = torch.tensor(prompt.token_ids, dtype=torch.int64, device="cpu")
    visual_mask = ids == hf_config.image_token_id
    if not torch.equal(types, visual_mask.to(types.dtype)):
        raise ValueError("Modality types must exactly match image placeholders")
    if not len(grid):
        if pixels.shape[0]:
            raise ValueError("Pixels without image grid")
        validate_text_tokens(prompt.token_ids, hf_config)
        return
    if grid.shape[0] != 1:
        raise ValueError("Exactly one image grid is supported")
    t, h, w = grid[0].tolist()
    if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2:
        raise ValueError(
            "Static image grid requires T=1 and positive merge-divisible H/W"
        )
    patches = t * h * w
    visual_tokens = patches // 4
    if pixels.shape[0] != patches or visual_tokens > max_visual_tokens:
        raise ValueError("Pixel row count or visual token budget mismatch")
    locations = visual_mask.nonzero().flatten().tolist()
    if len(locations) != visual_tokens or locations != list(
        range(locations[0], locations[0] + visual_tokens)
    ):
        raise ValueError("Image placeholders must form one grid-sized span")
    start, end = locations[0] - 1, locations[-1] + 1
    tokens = prompt.token_ids
    if (
        start < 0
        or end >= len(tokens)
        or tokens[start] != hf_config.vision_start_token_id
        or tokens[end] != hf_config.vision_end_token_id
        or tokens.count(hf_config.vision_start_token_id) != 1
        or tokens.count(hf_config.vision_end_token_id) != 1
        or hf_config.video_token_id in tokens
    ):
        raise ValueError("Incomplete image placeholder syntax or unsupported video")
