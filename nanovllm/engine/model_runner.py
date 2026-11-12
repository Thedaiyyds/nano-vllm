import gc
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_vl import Qwen3VLForConditionalGeneration
from nanovllm.layers.attention import Attention
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model, load_qwen3_vl_weights


def kv_block_bytes(text_config, block_size: int, dtype: torch.dtype) -> int:
    """Bytes for both K and V across all language layers (unsharded, TP=1)."""
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = text_config.hidden_size // text_config.num_attention_heads
    for name, value in (
        ("block_size", block_size),
        ("num_hidden_layers", text_config.num_hidden_layers),
        ("num_key_value_heads", text_config.num_key_value_heads),
        ("head_dim", head_dim),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return (
        2
        * text_config.num_hidden_layers
        * block_size
        * text_config.num_key_value_heads
        * head_dim
        * dtype.itemsize
    )


def select_kv_block_count(config: Config, available_bytes: int) -> int:
    """Require the complete context capacity from a net, already-reserved budget."""
    for name, value in (
        ("max_model_len", config.max_model_len),
        ("kvcache_block_size", config.kvcache_block_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if type(available_bytes) is not int or available_bytes < 0:
        raise ValueError(
            f"available_bytes must be a nonnegative integer net KV budget, got {available_bytes!r}"
        )
    num_blocks = (
        config.max_model_len + config.kvcache_block_size - 1
    ) // config.kvcache_block_size
    required_bytes = num_blocks * kv_block_bytes(
        config.text_config, config.kvcache_block_size, config.model_dtype
    )
    if available_bytes < required_bytes:
        raise ValueError(
            f"insufficient KV budget: need {required_bytes} bytes for {num_blocks} blocks, "
            f"available {available_bytes}"
        )
    return num_blocks


class ModelRunner:
    def _load_model(self):
        config = self.config
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(config.model_dtype)
        try:
            if config.hf_config.model_type == "qwen3_vl":
                self.model = Qwen3VLForConditionalGeneration(
                    config.hf_config, text_only=False
                )
                # HF initializes non-weight text frequencies on CPU before moving
                # them to CUDA. GPU pow differs by FP32 ULPs that affect long RoPE.
                from nanovllm.layers.mrope import Qwen3VLRotaryEmbedding

                with torch.device("cpu"):
                    inv_freq = Qwen3VLRotaryEmbedding(config.text_config).inv_freq
                for layer in self.model.model.language_model.layers:
                    layer.self_attn.rotary_emb.inv_freq.copy_(inv_freq)
                self.weight_load_report = load_qwen3_vl_weights(
                    self.model, config.model
                )
            else:
                self.model = Qwen3ForCausalLM(config.hf_config)
                load_model(self.model, config.model)
        finally:
            torch.set_default_dtype(default_dtype)

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self._closed = False
        self._owns_process_group = False
        self._device_set = False
        self._initialized = False
        default_dtype = torch.get_default_dtype()
        default_device = torch.get_default_device()
        try:
            try:
                dist.init_process_group(
                    "nccl",
                    "tcp://localhost:2333",
                    world_size=self.world_size,
                    rank=rank,
                )
                self._owns_process_group = True
                torch.cuda.set_device(rank)
                self._device_set = True
                torch.set_default_device("cuda")
                self._load_model()
                torch.set_default_dtype(config.model_dtype)
                self.sampler = Sampler()
                self.warmup_model()
                self.allocate_kv_cache()
                if not self.enforce_eager:
                    self.capture_cudagraph()
            finally:
                torch.set_default_device(default_device)
                torch.set_default_dtype(default_dtype)

            if self.world_size > 1:
                if rank == 0:
                    self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                    dist.barrier()
                else:
                    dist.barrier()
                    self.shm = SharedMemory(name="nanovllm")
            self._initialized = True
            if self.world_size > 1 and rank > 0:
                self.loop()
        except Exception as error:
            try:
                self.exit()
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    def exit(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            if hasattr(self, "shm"):
                self.shm.close()
                if self._initialized:
                    dist.barrier()
                if self.rank == 0:
                    self.shm.unlink()
            if self._device_set:
                torch.cuda.synchronize()
        finally:
            reset_context()
            for name in (
                "model",
                "kv_cache",
                "sampler",
                "graphs",
                "graph_pool",
                "graph_vars",
                "shm",
            ):
                self.__dict__.pop(name, None)
            try:
                if self._owns_process_group and dist.is_initialized():
                    dist.destroy_process_group()
            finally:
                self._owns_process_group = False
                if self._device_set:
                    # Parameters' bound weight_loader methods form module cycles.
                    gc.collect()
                    torch.cuda.empty_cache()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        if self.config.is_multimodal and not hasattr(self, "kv_cache"):
            self.vl_memory_profile = self.profile_vl_memory()
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        if self.config.is_multimodal:
            seq_len, num_seqs = max_model_len, 1
        else:
            seq_len = min(max_num_batched_tokens, max_model_len)
            num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def _vl_memory_grids(self):
        """Largest admissible square and both extreme Prepared grid directions."""
        config = self.config
        hf = config.hf_config
        length = min(config.max_model_len, config.max_num_batched_tokens)
        # Small test checkpoints may have reserved IDs outside their vocabulary;
        # they have no admissible image input. Do not bypass public validation.
        if any(
            not 0 <= token < hf.text_config.vocab_size
            for token in (
                hf.image_token_id,
                hf.vision_start_token_id,
                hf.vision_end_token_id,
            )
        ):
            return []
        visual = min(config.max_visual_tokens, length - 2)
        if visual < 1:
            return []
        side = int(visual**0.5)
        # Prepared permits merge-divisible extreme shapes beyond the natural
        # image Processor's smart_resize aspect bound; cover that wider contract.
        return [(2 * side, 2 * side), (2, 2 * visual), (2 * visual, 2)]

    @torch.inference_mode()
    def _run_vl_memory_case(self, grid):
        """Internal resource forward: validated inputs without request admission.

        Before allocation there is intentionally no KV. After allocation the
        same full Prefill writes real slots. No returned tensors can survive to
        the caller's current-allocation measurement.
        """
        from nanovllm.layers.mrope import build_mrope_positions
        from nanovllm.utils.multimodal import (
            PreparedMultimodalPrompt,
            validate_multimodal,
            validate_text_tokens,
        )

        config = self.config
        hf = config.hf_config
        length = min(config.max_model_len, config.max_num_batched_tokens)
        reserved = {
            hf.image_token_id,
            hf.video_token_id,
            hf.vision_start_token_id,
            hf.vision_end_token_id,
        }
        text_id = next(i for i in range(hf.text_config.vocab_size) if i not in reserved)
        tokens = [text_id] * length
        features = mask = None
        try:
            if grid is None:
                # Full context is conservative for eventual text Decode even
                # when admission limits individual Prefill to fewer tokens.
                tokens = [text_id] * config.max_model_len
                validate_text_tokens(tokens, hf)
                seq = Sequence(tokens)
                seq.num_scheduled_tokens = len(tokens)
                if hasattr(self, "kv_cache"):
                    seq.block_table = list(range(config.num_kvcache_blocks))
                self.run([seq], True)
                return
            else:
                h, w = grid
                visual = h * w // 4
                if visual + 2 > length:
                    raise ValueError("profile image exceeds admissible language length")
                tokens[: visual + 2] = (
                    [hf.vision_start_token_id]
                    + [hf.image_token_id] * visual
                    + [hf.vision_end_token_id]
                )
                vision = hf.vision_config
                width = (
                    vision.in_channels
                    * vision.temporal_patch_size
                    * vision.patch_size**2
                )
                pixels = torch.zeros(
                    (h * w, width), dtype=config.model_dtype, device="cpu"
                )
                grid_tensor = torch.tensor([[1, h, w]], dtype=torch.int64, device="cpu")
                types = torch.tensor(
                    [int(t == hf.image_token_id) for t in tokens], device="cpu"
                )
                prepared = PreparedMultimodalPrompt(tokens, pixels, grid_tensor, types)
                validate_multimodal(
                    prepared, hf, max_visual_tokens=config.max_visual_tokens
                )
                positions, _ = build_mrope_positions(tokens, types, grid_tensor, 2)
                positions = positions.to("cuda")
                pixels = pixels.to("cuda")
                features = self.model.encode_images(pixels, grid_tensor)
                mask = types.to(device="cuda", dtype=torch.bool)
            input_ids = torch.tensor(tokens, dtype=torch.int64, device="cuda")
            cu = torch.tensor([0, length], dtype=torch.int32, device="cuda")
            slots = (
                torch.arange(length, dtype=torch.int32, device="cuda")
                if hasattr(self, "kv_cache")
                else torch.empty(0, dtype=torch.int32, device="cuda")
            )
            set_context(True, cu, cu, length, length, slots)
            self.run_model(
                input_ids, positions, True, visual_features=features, visual_mask=mask
            )
        finally:
            reset_context()

    def profile_vl_memory(self) -> dict[str, int]:
        """Joint text/vision activation peak before KV allocation, in bytes."""
        if hasattr(self, "kv_cache"):
            raise ValueError("VL memory profile must precede KV allocation")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        grids = self._vl_memory_grids()
        for grid in [None, *grids]:
            self._run_vl_memory_case(grid)
        # The resource forward stack (including logits/features/pixels) has
        # ended before measuring current; live outputs cannot cancel reserve.
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        stats = torch.cuda.memory_stats()
        free, total = torch.cuda.mem_get_info()
        peak = stats["allocated_bytes.all.peak"]
        current = stats["allocated_bytes.all.current"]
        activation = peak - current
        reserve = self.config.kv_cache_reserve_bytes
        available = (
            int(total * self.config.gpu_memory_utilization)
            - (total - free)
            - activation
            - reserve
        )
        return dict(
            peak_allocated=peak,
            peak_reserved=stats["reserved_bytes.all.peak"],
            free_after=free,
            reserve_bytes=reserve,
            current_allocated=current,
            activation_bytes=activation,
            available_bytes=available,
            total_bytes=total,
            image_case_count=len(grids),
        )

    def allocate_kv_cache(self):
        config = self.config
        if config.is_multimodal:
            self._allocate_vl_kv_cache()
            return
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _allocate_vl_kv_cache(self):
        config = self.config
        text = config.text_config
        attentions = [
            m
            for m in self.model.model.language_model.modules()
            if isinstance(m, Attention)
        ]
        if len(attentions) != text.num_hidden_layers:
            raise ValueError(
                f"VL language Attention count {len(attentions)} != {text.num_hidden_layers}"
            )
        if not hasattr(self, "vl_memory_profile"):
            self.vl_memory_profile = self.profile_vl_memory()
        free, total = torch.cuda.mem_get_info()
        # Refresh free/config reserve (also on explicit reallocation), while
        # reusing the measured activation increment, never a stale net budget.
        available = (
            int(total * config.gpu_memory_utilization)
            - (total - free)
            - self.vl_memory_profile["activation_bytes"]
            - config.kv_cache_reserve_bytes
        )
        num_blocks = select_kv_block_count(config, available)
        self.kv_cache = torch.empty(
            2,
            text.num_hidden_layers,
            num_blocks,
            self.block_size,
            text.num_key_value_heads,
            text.head_dim,
            dtype=config.model_dtype,
            device="cuda",
        )
        config.num_kvcache_blocks = num_blocks
        for i, attention in enumerate(attentions):
            attention.k_cache = self.kv_cache[0, i]
            attention.v_cache = self.kv_cache[1, i]

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        if self.config.is_multimodal:
            if len(seqs) != 1 or any(
                seq.num_cached_tokens != 0 or seq.num_scheduled_tokens != seq.num_tokens
                for seq in seqs
            ):
                raise ValueError("VL Prefill requires one complete, uncached sequence")
            if (
                seqs[0].multimodal is not None
                and len(seqs[0].multimodal.image_grid_thw)
                and not seqs[0].block_table
            ):
                raise ValueError("Image Prefill requires an admitted request")
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:  # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = (
                        seq.block_table[i] * self.block_size + end - i * self.block_size
                    )
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        if self.config.is_multimodal:
            prompt_positions = seqs[0].prompt_positions
            positions = (
                prompt_positions[:, : seqs[0].num_scheduled_tokens].to(
                    input_ids.device, non_blocking=True
                )
                if prompt_positions is not None
                else positions.unsqueeze(0).expand(3, -1)
            )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        if self.config.is_multimodal and len(seqs) != 1:
            raise ValueError("VL Decode requires one active sequence")
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        if self.config.is_multimodal:
            positions = positions.unsqueeze(0).expand(3, -1)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        *,
        visual_features=None,
        visual_mask=None,
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if visual_features is not None or visual_mask is not None:
                hidden = self.model(
                    input_ids,
                    positions,
                    visual_features=visual_features,
                    visual_mask=visual_mask,
                )
            else:
                hidden = self.model(input_ids, positions)
            return self.model.compute_logits(hidden)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, : context.block_tables.size(1)] = (
                context.block_tables
            )
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        try:
            image_seqs = [
                seq
                for seq in seqs
                if self.config.is_multimodal
                and seq.multimodal is not None
                and len(seq.multimodal.image_grid_thw)
            ]
            if image_seqs:
                if not is_prefill:
                    raise NotImplementedError("Image Decode is deferred until Task14")
            input_ids, positions = (
                self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            )
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            if image_seqs:
                seq = image_seqs[0]
                pixels = seq.multimodal.pixel_values.to(
                    device=input_ids.device, dtype=self.config.model_dtype
                )
                features = self.model.encode_images(
                    pixels, seq.multimodal.image_grid_thw
                )
                mask = seq.visual_mask.to(input_ids.device)
                logits = self.run_model(
                    input_ids,
                    positions,
                    is_prefill,
                    visual_features=features,
                    visual_mask=mask,
                )
            else:
                logits = self.run_model(input_ids, positions, is_prefill)
            return (
                self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            )
        finally:
            reset_context()

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
