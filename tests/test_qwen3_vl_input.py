"""CPU input contracts: real pinned Processor, owned data, no model allocation."""

from dataclasses import replace
import os
from types import SimpleNamespace
from unittest import TestCase

import pytest

pytestmark = [pytest.mark.hf_reference, pytest.mark.checkpoint]


def model_path():
    from tests.reference_qwen3_vl import DEFAULT_MODEL

    return os.environ.get("QWEN3_VL_MODEL", DEFAULT_MODEL)


@pytest.fixture(scope="module")
def resources():
    import torch
    from transformers import AutoConfig, AutoProcessor

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield (
        AutoConfig.from_pretrained(model_path(), local_files_only=True),
        AutoProcessor.from_pretrained(model_path(), local_files_only=True),
    )
    torch.set_num_threads(threads)


def messages(size=(384, 256)):
    from PIL import Image

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look carefully: "},
                {"type": "image", "image": Image.new("RGB", size, (180, 40, 20))},
                {"type": "text", "text": "Describe the picture briefly."},
            ],
        }
    ]


def small_prompt():
    import torch
    from nanovllm.utils.multimodal import PreparedMultimodalPrompt

    return PreparedMultimodalPrompt(
        [10, 151652] + [151655] * 6 + [151653, 11, 12],
        torch.arange(24 * 1536, dtype=torch.float32).reshape(24, 1536),
        torch.tensor([[1, 4, 6]]),
        torch.tensor([0, 0] + [1] * 6 + [0, 0, 0]),
    )


def text_prompt(ids=None):
    import torch
    from nanovllm.utils.multimodal import PreparedMultimodalPrompt

    ids = [10, 11] if ids is None else ids
    return PreparedMultimodalPrompt(
        ids,
        torch.empty(0, 1536),
        torch.empty(0, 3, dtype=torch.long),
        torch.zeros(len(ids), dtype=torch.long),
    )


class InputTests(TestCase):
    @pytest.fixture(autouse=True)
    def setup_resources(self, resources):
        self.config, self.processor = resources

    def test_processor_grid_and_validation(self):
        import torch
        from nanovllm.utils.multimodal import prepare_multimodal, validate_multimodal
        from nanovllm.engine.sequence import Sequence
        from tests.test_qwen3_vl_mrope import hf_positions

        for size, grid, rows, count in [
            ((256, 256), [[1, 16, 16]], 256, 64),
            ((384, 256), [[1, 16, 24]], 384, 96),
            ((512, 512), [[1, 32, 32]], 1024, 256),
        ]:
            with self.subTest(size=size):
                conversation = messages(size)
                expected = self.processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"return_mm_token_type_ids": True},
                )
                prepared = prepare_multimodal(conversation, self.processor)
                validate_multimodal(prepared, self.config, max_visual_tokens=256)
                self.assertEqual(prepared.token_ids, expected["input_ids"][0].tolist())
                self.assertEqual(prepared.image_grid_thw.tolist(), grid)
                self.assertEqual(tuple(prepared.pixel_values.shape), (rows, 1536))
                self.assertEqual(
                    prepared.token_ids.count(self.config.image_token_id), count
                )
                for field in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
                    actual = getattr(prepared, field)
                    reference = (
                        expected[field][0]
                        if field == "mm_token_type_ids"
                        else expected[field]
                    )
                    self.assertTrue(torch.equal(actual, reference), field)
                    self.assertEqual(actual.dtype, reference.dtype)
                    self.assertEqual(actual.device.type, "cpu")
                seq = Sequence(prepared.token_ids, multimodal=prepared)
                positions, delta = hf_positions(
                    prepared.token_ids,
                    prepared.mm_token_type_ids,
                    prepared.image_grid_thw,
                )
                self.assertTrue(torch.equal(seq.prompt_positions, positions))
                self.assertEqual(seq.rope_delta, delta)

    def test_request_owns_cpu_state(self):
        import torch
        from nanovllm.engine.sequence import Sequence

        prepared = small_prompt()
        original_ids = prepared.token_ids.copy()
        original_pixels = prepared.pixel_values.clone()
        seq = Sequence(original_ids, multimodal=prepared)
        prepared.token_ids[0] = 99
        prepared.pixel_values.fill_(0)
        prepared.image_grid_thw.fill_(2)
        prepared.mm_token_type_ids.fill_(0)
        original_ids[1] = 99
        self.assertEqual(seq.token_ids, [10, 151652] + [151655] * 6 + [151653, 11, 12])
        self.assertEqual(seq.multimodal.token_ids, seq.token_ids)
        self.assertTrue(torch.equal(seq.multimodal.pixel_values, original_pixels))
        self.assertEqual(seq.multimodal.image_grid_thw.tolist(), [[1, 4, 6]])
        self.assertEqual(
            seq.visual_mask.tolist(), [False, False] + [True] * 6 + [False] * 3
        )
        self.assertEqual(seq.rope_delta, -3)
        self.assertEqual(
            seq.prompt_positions.tolist(),
            [
                [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7],
                [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7],
                [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7],
            ],
        )
        other = Sequence(small_prompt().token_ids, multimodal=small_prompt())
        seq.multimodal.pixel_values.fill_(-1)
        self.assertFalse(
            torch.equal(seq.multimodal.pixel_values, other.multimodal.pixel_values)
        )
        plain = Sequence([1, 2])
        self.assertIsNone(plain.multimodal)
        self.assertIsNone(plain.prompt_positions)
        self.assertIsNone(plain.visual_mask)
        self.assertEqual(plain.rope_delta, 0)

    def test_resize_respects_explicit_visual_budget(self):
        from nanovllm.utils.multimodal import prepare_multimodal

        prepared = prepare_multimodal(
            messages((1024, 1024)), self.processor, max_visual_tokens=64
        )
        expected = self.processor.apply_chat_template(
            messages((1024, 1024)),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "return_mm_token_type_ids": True,
                "padding": False,
                "images_kwargs": {
                    "size": {"shortest_edge": 65536, "longest_edge": 65536}
                },
            },
        )
        import torch

        self.assertEqual(prepared.image_grid_thw.tolist(), [[1, 16, 16]])
        self.assertEqual(prepared.token_ids.count(151655), 64)
        self.assertTrue(torch.equal(prepared.pixel_values, expected["pixel_values"]))


def invalid_prompt(case):
    import torch

    prompt = small_prompt()
    if case == "time":
        return replace(prompt, image_grid_thw=torch.tensor([[2, 4, 6]]))
    if case == "odd_height":
        return replace(prompt, image_grid_thw=torch.tensor([[1, 3, 6]]))
    if case == "odd_width":
        return replace(prompt, image_grid_thw=torch.tensor([[1, 4, 5]]))
    if case == "zero_grid":
        return replace(prompt, image_grid_thw=torch.tensor([[1, 0, 6]]))
    if case == "missing_grid":
        return replace(prompt, image_grid_thw=torch.empty(0, 3, dtype=torch.long))
    if case == "multiple_images":
        return replace(prompt, image_grid_thw=torch.tensor([[1, 4, 6], [1, 4, 6]]))
    if case == "pixel_rows":
        return replace(prompt, pixel_values=prompt.pixel_values[:-1])
    if case == "pixel_width":
        return replace(prompt, pixel_values=prompt.pixel_values[:, :-1])
    if case == "pixel_dtype":
        return replace(prompt, pixel_values=prompt.pixel_values.long())
    if case == "pixel_nan":
        return replace(prompt, pixel_values=prompt.pixel_values.fill_(float("nan")))
    if case == "pixel_inf":
        return replace(prompt, pixel_values=prompt.pixel_values.fill_(float("inf")))
    if case == "grid_dtype":
        return replace(prompt, image_grid_thw=prompt.image_grid_thw.float())
    if case == "grid_rank":
        return replace(prompt, image_grid_thw=prompt.image_grid_thw[0])
    if case == "mask_dtype":
        return replace(prompt, mm_token_type_ids=prompt.mm_token_type_ids.bool())
    if case == "mask_length":
        return replace(prompt, mm_token_type_ids=prompt.mm_token_type_ids[:-1])
    if case == "mask_values":
        return replace(prompt, mm_token_type_ids=prompt.mm_token_type_ids * 2)
    if case == "mask_mismatch":
        return replace(prompt, mm_token_type_ids=prompt.mm_token_type_ids.roll(1))
    if case == "empty":
        return text_prompt([])
    if case == "orphan_image":
        return text_prompt([151655])
    if case == "orphan_start":
        return text_prompt([151652, 10])
    if case == "orphan_end":
        return text_prompt([10, 151653])
    if case == "video":
        return text_prompt([151656])
    if case == "negative_id":
        return text_prompt([-1])
    if case == "boolean_id":
        return text_prompt([True])
    if case == "oov_id":
        return text_prompt([151936])
    if case == "missing_start":
        prompt.token_ids[1] = 10
    if case == "missing_end":
        prompt.token_ids[-3] = 10
    if case == "extra_start":
        prompt.token_ids[0] = 151652
    if case == "split_span":
        prompt.token_ids[3], prompt.token_ids[-1] = (
            prompt.token_ids[-1],
            prompt.token_ids[3],
        )
    if case == "over_budget":
        return replace(
            prompt,
            token_ids=[151652] + [151655] * 272 + [151653],
            pixel_values=torch.zeros(1088, 1536),
            image_grid_thw=torch.tensor([[1, 32, 34]]),
            mm_token_type_ids=torch.tensor([0] + [1] * 272 + [0]),
        )
    return prompt


INVALID_CASES = [
    "time",
    "odd_height",
    "odd_width",
    "zero_grid",
    "missing_grid",
    "multiple_images",
    "pixel_rows",
    "pixel_width",
    "pixel_dtype",
    "pixel_nan",
    "pixel_inf",
    "grid_dtype",
    "grid_rank",
    "mask_dtype",
    "mask_length",
    "mask_values",
    "mask_mismatch",
    "empty",
    "orphan_image",
    "orphan_start",
    "orphan_end",
    "video",
    "negative_id",
    "boolean_id",
    "oov_id",
    "missing_start",
    "missing_end",
    "extra_start",
    "split_span",
    "over_budget",
]


@pytest.mark.parametrize("case", INVALID_CASES)
def test_invalid_prepared_rejected(case, resources):
    from nanovllm.utils.multimodal import validate_multimodal

    with pytest.raises(ValueError):
        validate_multimodal(invalid_prompt(case), resources[0], max_visual_tokens=256)


@pytest.mark.parametrize("limit", [0, -1, 257, 1.5, True])
def test_invalid_visual_budget(limit, resources):
    from nanovllm.utils.multimodal import prepare_multimodal, validate_multimodal

    with pytest.raises(ValueError):
        validate_multimodal(small_prompt(), resources[0], max_visual_tokens=limit)
    with pytest.raises(ValueError):
        prepare_multimodal(messages(), resources[1], max_visual_tokens=limit)


@pytest.mark.parametrize("case", ["empty", "text_only", "two_images", "video", "batch"])
def test_unsupported_messages(case, resources):
    from nanovllm.utils.multimodal import prepare_multimodal

    conversation = messages()
    if case == "empty":
        conversation = []
    if case == "text_only":
        conversation[0]["content"] = [{"type": "text", "text": "hello"}]
    if case == "two_images":
        conversation[0]["content"].append(conversation[0]["content"][1])
    if case == "video":
        conversation[0]["content"][1] = {"type": "video", "video": "unused.mp4"}
    if case == "batch":
        conversation = [conversation, conversation]
    with pytest.raises(ValueError):
        prepare_multimodal(conversation, resources[1])


@pytest.mark.parametrize("case", ["padding", "batch", "video_fields"])
def test_processor_output_not_silently_normalized(case, resources):
    from nanovllm.utils.multimodal import prepare_multimodal

    class CorruptedProcessor:
        def __getattr__(self, key):
            return getattr(resources[1], key)

        def apply_chat_template(self, *args, **kwargs):
            output = resources[1].apply_chat_template(*args, **kwargs)
            if case == "padding":
                output["attention_mask"][0, 0] = 0
            if case == "batch":
                output["input_ids"] = output["input_ids"].repeat(2, 1)
            if case == "video_fields":
                output["video_grid_thw"] = output["image_grid_thw"]
            return output

    with pytest.raises(ValueError):
        prepare_multimodal(messages(), CorruptedProcessor())


def cpu_engine(config, processor, *, multimodal=True):
    from nanovllm.engine.llm_engine import LLMEngine
    from tests.test_qwen3_vl_scheduler import vl_scheduler

    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        is_multimodal=multimodal, hf_config=config, max_visual_tokens=256
    )
    engine.tokenizer = processor.tokenizer
    engine.scheduler = vl_scheduler()
    return engine


@pytest.mark.parametrize("kind", ["ids", "string", "prepared"])
@pytest.mark.parametrize("bad_id", [151652, 151653, 151655, 151656])
def test_engine_rejects_orphans_before_queue_or_kv(kind, bad_id, resources):
    from tests.test_qwen3_vl_scheduler import pool_state

    engine = cpu_engine(*resources)
    before = pool_state(engine.scheduler.block_manager)
    prompt = [10, bad_id]
    if kind == "string":
        prompt = resources[1].tokenizer.decode(prompt)
    if kind == "prepared":
        prompt = text_prompt(prompt)
    with pytest.raises(ValueError):
        engine.add_request(prompt, __import__("nanovllm").SamplingParams(max_tokens=1))
    assert engine.is_finished()
    assert pool_state(engine.scheduler.block_manager) == before


def test_engine_valid_prepared_explicitly_blocked_before_kv(resources):
    from nanovllm import SamplingParams
    from tests.test_qwen3_vl_scheduler import pool_state

    engine = cpu_engine(*resources)
    before = pool_state(engine.scheduler.block_manager)
    with pytest.raises(NotImplementedError, match="[Ii]mage|图像"):
        engine.generate([small_prompt()], SamplingParams(max_tokens=1), use_tqdm=False)
    assert engine.is_finished()
    assert pool_state(engine.scheduler.block_manager) == before


def test_engine_text_queue_and_legacy_unchanged(resources):
    from nanovllm import SamplingParams

    engine = cpu_engine(*resources)
    engine.add_request([10, 11], SamplingParams(max_tokens=1))
    engine.add_request("hello", SamplingParams(max_tokens=1))
    engine.add_request(text_prompt(), SamplingParams(max_tokens=1))
    assert len(engine.scheduler.waiting) == 3
    assert all(not seq.block_table for seq in engine.scheduler.waiting)
    assert engine.scheduler.waiting[2].prompt_positions.tolist() == [[0, 1]] * 3
    legacy = cpu_engine(*resources, multimodal=False)
    legacy.add_request([10, 11], SamplingParams(max_tokens=1))
    assert legacy.scheduler.waiting[0].token_ids == [10, 11]
    with pytest.raises(ValueError):
        legacy.add_request(small_prompt(), SamplingParams(max_tokens=1))


@pytest.mark.parametrize("prompt", [[], ""])
def test_engine_empty_input_rejected_before_queue(prompt, resources):
    from nanovllm import SamplingParams

    engine = cpu_engine(*resources)
    with pytest.raises(ValueError):
        engine.add_request(prompt, SamplingParams(max_tokens=1))
    assert engine.is_finished()


@pytest.mark.parametrize(
    "field", ["pixel_values", "image_grid_thw", "mm_token_type_ids"]
)
def test_non_cpu_storage_rejected(field, resources):
    from nanovllm.utils.multimodal import validate_multimodal
    from nanovllm.engine.sequence import Sequence

    prompt = small_prompt()
    # Real meta tensors exercise device validation without needing CUDA.
    setattr(prompt, field, getattr(prompt, field).to("meta"))
    with pytest.raises(ValueError, match="CPU"):
        validate_multimodal(prompt, resources[0], max_visual_tokens=256)
    with pytest.raises(ValueError, match="CPU"):
        Sequence(prompt.token_ids, multimodal=prompt)


def test_lower_visual_budget_and_merge_contract(resources):
    from nanovllm.utils.multimodal import validate_multimodal

    prompt = small_prompt()
    validate_multimodal(prompt, resources[0], max_visual_tokens=6)
    with pytest.raises(ValueError, match="budget"):
        validate_multimodal(prompt, resources[0], max_visual_tokens=5)
    config = SimpleNamespace(**resources[0].to_dict())
    config.text_config = resources[0].text_config
    config.vision_config = SimpleNamespace(spatial_merge_size=4)
    with pytest.raises(ValueError, match="merge"):
        validate_multimodal(prompt, config, max_visual_tokens=256)


def test_sequence_ids_must_match_owned_prompt():
    from nanovllm.engine.sequence import Sequence

    with pytest.raises(ValueError, match="match"):
        Sequence([10, 11], multimodal=small_prompt())


@pytest.mark.parametrize("case", ["image_square", "image_rect", "image_limit"])
def test_prepared_matches_persistent_oracle(case, resources):
    import json
    from pathlib import Path
    import torch
    from tests.reference_qwen3_vl import CASES, validate_artifact
    from nanovllm.utils.multimodal import prepare_multimodal

    spec = json.loads(CASES.read_text())[case]
    conversation = messages(tuple(spec["size"]))
    from PIL import Image

    conversation[0]["content"][1]["image"] = Image.new(
        "RGB", tuple(spec["size"]), tuple(spec["color"])
    )
    prepared = prepare_multimodal(conversation, resources[1])
    folder = (
        Path(
            os.environ.get(
                "QWEN3_VL_ORACLE",
                str(Path.home() / "nano-vllm-artifacts/qwen3-vl/oracle-reviewed"),
            )
        )
        / "hf-sdpa"
        / case
    )
    expected = validate_artifact(
        json.loads((folder / "manifest.json").read_text()), folder
    )
    assert prepared.token_ids == expected["input.input_ids"][0].tolist()
    assert torch.equal(prepared.pixel_values, expected["input.pixel_values"])
    assert torch.equal(prepared.image_grid_thw, expected["input.image_grid_thw"])
    assert torch.equal(
        prepared.mm_token_type_ids, expected["input.mm_token_type_ids"][0]
    )


@pytest.mark.parametrize("source", ["pil", "data_uri"])
@pytest.mark.parametrize("key", ["image", "url", "path", "base64"])
def test_animation_never_silently_flattened(source, key, resources):
    import base64
    from io import BytesIO
    from PIL import Image
    from nanovllm.utils.multimodal import prepare_multimodal

    buffer = BytesIO()
    Image.new("RGB", (256, 256), "red").save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (256, 256), "blue")],
        duration=100,
    )
    encoded = buffer.getvalue()
    animation = Image.open(BytesIO(encoded))
    assert animation.n_frames == 2
    value = (
        animation
        if source == "pil"
        else "data:image/png;base64," + base64.b64encode(encoded).decode()
    )
    conversation = messages()
    conversation[0]["content"][1] = {"type": "image", key: value}
    with pytest.raises(ValueError):
        prepare_multimodal(conversation, resources[1])


@pytest.mark.parametrize("key", ["image", "url", "path", "base64"])
def test_static_source_forms_explicitly_bounded(key, resources):
    from nanovllm.utils.multimodal import prepare_multimodal

    conversation = messages()
    picture = conversation[0]["content"][1]["image"]
    conversation[0]["content"][1] = {"type": "image", key: picture}
    if key == "image":
        assert prepare_multimodal(
            conversation, resources[1]
        ).image_grid_thw.tolist() == [[1, 16, 24]]
    else:
        with pytest.raises(ValueError):
            prepare_multimodal(conversation, resources[1])


def test_static_encoded_source_explicitly_rejected(resources):
    import base64
    from io import BytesIO
    from nanovllm.utils.multimodal import prepare_multimodal

    conversation = messages()
    buffer = BytesIO()
    conversation[0]["content"][1]["image"].save(buffer, format="PNG")
    conversation[0]["content"][1]["image"] = (
        "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    )
    with pytest.raises(ValueError):
        prepare_multimodal(conversation, resources[1])
