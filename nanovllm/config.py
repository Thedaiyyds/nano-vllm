import os
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig, GenerationConfig, PretrainedConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    text_config: PretrainedConfig = field(init=False)
    model_dtype: torch.dtype = field(init=False)
    is_multimodal: bool = field(init=False)
    eos_token_ids: frozenset[int] = field(init=False)
    max_visual_tokens: int = 256
    kv_cache_reserve_bytes: int = 1073741824

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        self.hf_config = AutoConfig.from_pretrained(self.model, local_files_only=True)
        model_type = self.hf_config.model_type
        if model_type not in ("qwen3", "qwen3_vl"):
            raise ValueError(f"Unsupported model_type={model_type!r}")
        self.is_multimodal = model_type == "qwen3_vl"
        self.text_config = (
            self.hf_config.text_config if self.is_multimodal else self.hf_config
        )
        self.model_dtype = (
            self.text_config.dtype or self.hf_config.dtype or torch.float32
        )
        if os.path.isfile(os.path.join(self.model, "generation_config.json")):
            generation_config = GenerationConfig.from_pretrained(
                self.model, local_files_only=True
            )
        else:
            generation_config = GenerationConfig.from_model_config(self.hf_config)
        eos = generation_config.eos_token_id
        self.eos_token_ids = frozenset(
            [] if eos is None else [eos] if isinstance(eos, int) else eos
        )

        if self.is_multimodal:
            for name, expected in (
                ("tensor_parallel_size", 1),
                ("max_num_seqs", 1),
                ("enforce_eager", True),
                ("model_dtype", torch.bfloat16),
            ):
                value = getattr(self, name)
                if value != expected:
                    raise ValueError(f"VL {name}={value!r}; requires {expected!r}")
            for name in (
                "max_model_len",
                "max_num_batched_tokens",
                "max_visual_tokens",
                "kv_cache_reserve_bytes",
            ):
                value = getattr(self, name)
                if value <= 0:
                    raise ValueError(f"VL {name}={value!r}; must be positive")
            for name, limit in (("max_model_len", 4096), ("max_visual_tokens", 256)):
                value = getattr(self, name)
                if value > limit:
                    raise ValueError(f"VL {name}={value!r}; maximum is {limit}")
        else:
            assert 1 <= self.tensor_parallel_size <= 8
        self.max_model_len = min(
            self.max_model_len, self.text_config.max_position_embeddings
        )
