import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        #初始化分布式进程组，确认多进程通信地址，例如后面all_reduce()需要多进程进行通信交换信息
        #并且只有运行model的时候才开启分布式，之前的阶段都是由一个主进程运行
        #如果用gloo，则是cpu分布式计算，做实验可以用这个模拟多gpu
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        #给对应rank的进程相应的GPU
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        #将本地预训练的权重加载到刚才创建的模型中，每个rank都会调用
        load_model(self.model, config.model)
        self.sampler = Sampler()
        #通过warp up来计算kv cache可分配的内存
        self.warmup_model()
        #分配kv cache
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            #主进程进入rank0，创建共享内存
            if rank == 0:
                #创建共享内存，向其他rank发送seq等消息
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
            #其他rank从内存中读取参数
                #所有rank到达barrier才能进行下一步
                dist.barrier()
                #为当前rank设置共享内存
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            #读数据，如果还没写就先阻塞
            method_name, args = self.read_shm()
            #除rank0以外的rank只执行计算，返回值不接收
            self.call(method_name, *args)
            if method_name == "exit":
                break
    #其他rank从share momery中读取参数
    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        #等待事件发生
        self.event.wait()
        #读取数据
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    #rank0写入shard memory传递参数给其他rank
    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        #序列化成字节数据
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        #由rank0触发所有事件
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    #模拟一次较大规模的 Forward，测量显存峰值，根据峰值估算需要预留的临时显存，把剩余可用预算分配给 KV Cache
    def warmup_model(self):
        #释放 PyTorch 缓存分配器中当前未被占用的缓存显存，例如用完的tensor但是gpu显存并未释放掉
        torch.cuda.empty_cache()
        #重置峰值统计，以便接下来测量这次模拟推理的显存峰值
        torch.cuda.reset_peak_memory_stats()
        #获取最大seq大小
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        #计算最大seq数量
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        #模拟峰值内存的计算
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        #当前设备显存使用情况，整张 GPU 上已被占用的显存，包括 PyTorch、CUDA 上下文和其他程序等
        free, total = torch.cuda.mem_get_info()
        used = total - free
        #统计 PyTorch 分配器峰值的活跃分配量
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        #统计 PyTorch 分配器当前的活跃分配量
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        #scheduler为seq分配一块block实际对应所有transformer层的kv cache大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        #分配多少个这样的kv block
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        #最终创建真正的kv cache，这里创建的num_kvcache_blocks对应block manager的控制信息block id
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        #这里需要padding才能输出才能转成二维tensor
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    #prepare_prefill会这一批所有batch中的seq都拼成一个，防止不通prompt需要padding导致的资源浪费
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0] #Cumulative
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            #计算当前seq需要处理的位置
            start = seq.num_cached_tokens
            #该seq需要使用的q大小
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            #该seq需要使用的k大小
            #这里qk不同因为新的q也需要使用之前的k，但是不需要使用之前的q
            seqlen_k = end
            #记录当前seq的信息
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            #warp不经过调度去分配block，没有分配物理block不用计算存储位置，这里其实是为了init中warm up的峰值内存计算
            if not seq.block_table:    # warmup
                continue
            #计算真实的kv cache存储位置，但是这里计算的是对于每层cache而言的相对位置，类比虚拟地址转换物理地址
            #slot = block_id * block_size + block_offset，一个slot大小为D = num_heads * head_dim
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        #处理Q需要使用到的K的token_num > 当前需要处理的Q的token_num，说明发生了prefix cache
        #context.block_tables 是本次 FlashAttention 是否需要通过分页缓存来读取 K、V 的执行信息
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # cuda传输cpu数据进入gpu需要先将数据暂存在锁页内存pin_memory中，然后传入gpu
        # pin_memory=True将数据直接放入锁页内存中，避免发生暂存拷贝的时间
        # non_blocking=True是让cpu不需要强制等待拷贝完成，异步执行
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        #将这些信息存入上下文
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    #Prefill：GPU 计算量大，Kernel Launch 开销所占比例通常较小。
    #Decode：单轮计算量较小，却要启动很多 Kernel，因此更容易从 CUDA Graph 中受益。
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            #这里有bug，假如len == 100，找不到适合的bs，触发 StopIteration 异常，无法选择合适的 CUDA Graph。
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            #向 CUDA 提交一次已经准备好的计算图执行请求，这样不用多次提交kernel导致GPU可能出现等待的空闲时间
            #CUDA Graph 确实可以看作一种提前准备好的 GPU 执行计划，捕获已有 GPU 操作、内存地址及其依赖关系，让它们能够被高效地重复执行。
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        #其他rank返回None
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        #捕获时使用的 Tensor 存储地址，必须在重放时仍然有效，且保持与捕获时一致
        #它们都是可以反复修改数值的固定 GPU 缓冲区
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
            #捕获Decode的Forward阶段
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            #第一次执行模型时，PyTorch、CUDA 库及部分 Kernel 可能需要进行初始化、选择算法或分配临时资源。
            #提前运行一次，能减少这些首次执行操作对后续捕获的影响。
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            #告诉 CUDA：接下来这段代码提交的 GPU 操作，请记录到 graph 中。
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            #CUDA Graph 执行需要的中间显存，减少多张 Graph 保留的显存开销，尝试复用一块显存池
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()
        #固定输入输出的缓冲区保存下来
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
