"""Native Qwen3-VL vision/text fusion with explicit HF residual rounding order."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.mrope import Qwen3VLRotaryEmbedding


def _split_projection(x, weight, bias, sizes):
    # Keep packed storage/loading, but match HF's independent GEMM rounding.
    weights = weight.split(sizes, dim=0)
    biases = [None] * len(sizes) if bias is None else bias.split(sizes, dim=0)
    return torch.cat([F.linear(x, w, b) for w, b in zip(weights, biases)], dim=-1)


class Qwen3VLQKVParallelLinear(QKVParallelLinear):
    def forward(self, x):
        sizes = [self.num_heads * self.head_size] + [
            self.num_kv_heads * self.head_size
        ] * 2
        return _split_projection(x, self.weight, self.bias, sizes)


class Qwen3VLMergedColumnParallelLinear(MergedColumnParallelLinear):
    def forward(self, x):
        return _split_projection(x, self.weight, self.bias, self.output_sizes)


class Qwen3VLMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported VL hidden_act={config.hidden_act!r}")
        self.gate_up_proj = Qwen3VLMergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = Qwen3VLQKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rotary_emb = Qwen3VLRotaryEmbedding(config)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads,
        )

    def forward(self, positions, hidden_states):
        q, k, v = self.qkv_proj(hidden_states).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        output = self.attn(q, k, v)
        return self.o_proj(output.flatten(1, -1))


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3VLTextAttention(config)
        self.mlp = Qwen3VLMLP(config)

    def forward(self, positions, hidden_states):
        # Existing RMSNorm mutates FP32 inputs; keep the residual independent.
        norm_input = (
            hidden_states.clone()
            if hidden_states.dtype == torch.float32
            else hidden_states
        )
        attention = self.self_attn(positions, self.input_layernorm(norm_input))
        hidden_states = hidden_states + attention
        norm_input = (
            hidden_states.clone()
            if hidden_states.dtype == torch.float32
            else hidden_states
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(norm_input))


class Qwen3VLTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [Qwen3VLDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions, *, visual_features=None, visual_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        if visual_features is not None:
            main, deepstack = visual_features
            hidden_states[visual_mask] = main
        for index, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states)
            if visual_features is not None and index < 3:
                hidden_states[visual_mask] = (
                    hidden_states[visual_mask] + deepstack[index]
                )
        return self.norm(hidden_states)


class Qwen3VLForConditionalGeneration(nn.Module):
    def __init__(self, config, *, text_only=True):
        super().__init__()
        if dist.get_world_size() != 1:
            raise ValueError("Qwen3-VL text model currently requires TP=1")
        if not config.tie_word_embeddings:
            raise ValueError("Qwen3-VL requires tied embedding/LM head")
        self.config = config
        text = config.text_config
        self.model = nn.Module()
        if not text_only:
            from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionModel

            self.model.visual = Qwen3VLVisionModel(config.vision_config)
        self.model.language_model = Qwen3VLTextModel(text)
        self.lm_head = ParallelLMHead(text.vocab_size, text.hidden_size)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(self, input_ids, positions, *, visual_features=None, visual_mask=None):
        # Validate all three branches before any decoder can write language KV.
        if (visual_features is None) != (visual_mask is None):
            raise ValueError(
                "visual_features and visual_mask must be provided together"
            )
        if visual_features is not None:
            if (
                not isinstance(visual_mask, torch.Tensor)
                or visual_mask.dtype != torch.bool
                or visual_mask.ndim != 1
                or visual_mask.shape != input_ids.shape
                or visual_mask.device != input_ids.device
            ):
                raise ValueError("visual_mask must be bool [N] on the input device")
            if not isinstance(visual_features, tuple) or len(visual_features) != 2:
                raise ValueError(
                    "visual_features must be (main, list[three DeepStack tensors])"
                )
            main, deepstack = visual_features
            if not isinstance(deepstack, list) or len(deepstack) != 3:
                raise ValueError(
                    "visual_features requires exactly three DeepStack tensors"
                )
            shape = (int(visual_mask.sum()), self.config.text_config.hidden_size)
            dtype = self.model.language_model.embed_tokens.weight.dtype
            for feature in (main, *deepstack):
                if (
                    not isinstance(feature, torch.Tensor)
                    or feature.shape != shape
                    or feature.device != input_ids.device
                    or feature.dtype != dtype
                ):
                    raise ValueError(
                        "all visual features must match [V, hidden_size], input device and embedding dtype"
                    )
        return self.model.language_model(
            input_ids,
            positions,
            visual_features=visual_features,
            visual_mask=visual_mask,
        )

    def encode_images(self, pixel_values, grid_thw):
        return self.model.visual(pixel_values, grid_thw)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)
