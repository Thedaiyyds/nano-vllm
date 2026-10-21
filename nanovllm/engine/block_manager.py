from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        #根据分配好的kv cache，这里创建控制信息
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        #不同位置出现的相同的block也不能共用kv cache
        #因为casual attention的存在，不同位置能看的历史上下文不同导致kv不同（可能第一层相同，后面的层受历史token影响导致不同）
        #但是和后面的块没有关系只需要看前缀，因为casual mask不允许前面的token关注后面的token
        if prefix != -1:
            #用prefix的hash值来将历史上下文纳入当前block标识
            #"little"表示小端存储
            h.update(prefix.to_bytes(8, "little"))
        #等同于hash[(prefix),token_ids]
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        #找第一个空闲block并分配
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        #将该空闲block之前的信息抹除
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        #但是此时还没分配kv cache block，只是关联了控制信息
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        #按block_size来处理seq
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            #判断是否有prefix cache，没有直接break，判定方式：hash表里面能查到就能命中，因为lazy delete
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            #有prefix cache，需要申请的block数量-1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        #判断剩余空闲block是否足够分配，不能返回-1，能返回能使用的prefix cache的block数量
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        #检查以前是否分配过block
        assert not seq.block_table
        h = -1
        #为每一个token block分配kv cache block，关联控制信息

        #先关联已经cache的block
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            #该kv cache block正在被使用则引用+1
            if block_id in self.used_block_ids:
                block.ref_count += 1
            #如果没有块正在使用但是其实kv cache block还没有被释放则将该block的控制信息重新改为占用
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        #为未cache的token block分配kv cache block
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                #没有block关联cache block则lazy delete（hash table指向还在）
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        #end用的也是整数除法，且没有加上block size - 1, 如果最后一块不完整则不会登记
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        #放入kv cache之后计算hash值，所以对应block的kv cache存在，这样hash值才为对应block存在的唯一标识
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
