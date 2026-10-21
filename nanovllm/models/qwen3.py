import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0

        #tp，每个rank拆分出自己需要处理的head大小
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        #该rank需要处理的q和kv的hidden_size部分大小（即多个rank最终拼接最后一维组成hidden_size）
        #这里attention之后的Q dmodel = (total_num_heads) * self.head_size，kv dmodel = (2 * total_num_kv_heads) * self.head_size
        #q_size = Q dmodel // tp_size,kv_size = kv dmodel
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        #tp计算qkv，并且先不all_reduce,每个rank值处理部分head
        #hidden_states (total_tokens, hidden_size)
        qkv = self.qkv_proj(hidden_states) #(total_tokens, (total_num_heads + 2 * total_num_kv_heads) * self.head_size) // tp_size)
        #拆分qkv
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim) # (total_tokens, num_heads, head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim) # (total_tokens, num_kv_heads, head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim) # (total_tokens, num_kv_heads, head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q) # (total_tokens, num_heads, head_dim)
            k = self.k_norm(k) # (total_tokens, num_kv_heads, head_dim)
        #对head_dim所在维度进行rotate
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v) # (total_tokens, num_heads， head_dim)
        # flatten之后为(total_tokens, num_heads *  head_dim)
        # 这里和cs336其实一样，后面乘以了一个oweight，将不同head中的信息组合，并且所有rank在这里做all_reduce拿到完整信息
        # 只不过这里先在不同rank中先乘以oweight再all_reduce，O @ W.T == [O1,O2 ... ,On] @ [w1,w2 ... wn].T == ∑OnWnT，类似于等价计算
        output = self.o_proj(o.flatten(1, -1)) # (total_tokens, hidden_size)
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # x (total_tokens, hidden_size)
        #同时tp计算gate和up
        gate_up = self.gate_up_proj(x) # (total_tokens,intermediate_size * 2 // tp_size)
        #Sigu(gate) * up
        x = self.act_fn(gate_up) # (total_tokens,intermediate_size * 2 // tp_size)
        #all_reduce所有rank，类似于o_proj的处理
        x = self.down_proj(x) # (total_tokens,hidden_size)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        #QHead == KVHead MHA
        #QHead > KVHead (KVHead == 1 MQA KVHead > 1 GQA)
        #hidden_size（即 d_model）：Transformer 各层传递的特征维度。
        #intermediate_size（即 d_ff）：MLP 内部进行特征变换时使用的中间维度
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        #RMSNorm未使用tp
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        #这里使用preNorm，一个transformer里面4个RMSNorm，input_layernorm，q_norm，k_norm，post_attention_layernorm
        #第一层输入residual为None，之后每一层residual先加上input,这样有机会融合多个操作进一个kernel，Fused Add + RMSNorm
        #每次归一化顺手把残存加法给做了
        #Attention之前的pre_Norm
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        #Attention
        hidden_states = self.self_attn(positions, hidden_states)# (total_tokens, hidden_size)
        #post_attention_layernorm 实际上就是 FFN（MLP）之前的 Pre-Norm
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)# (total_tokens, hidden_size)
        #FFN
        hidden_states = self.mlp(hidden_states) # (total_tokens, hidden_size)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        #FinalLayerNorm，最后再norm一次并且将residual加上
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class Qwen3ForCausalLM(nn.Module):
    #映射nano_vllm参数到hugging face参数
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        #权重共享，输入 Embedding 与输出 LM Head 共用权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        #计算logits
        return self.lm_head(hidden_states)
