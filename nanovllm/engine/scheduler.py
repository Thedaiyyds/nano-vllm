from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        #开始调度，prefill和decode阶段所有request处理完成才算finished
        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            #检查本轮剩余token预算
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            #检测prefix cache
            if not seq.block_table:
                #判断是否能分配kv cache，如果不能则prefill分配失败，直接break，尝试推进已有请求，才能可能释放kv cache空间
                #这里极端情况下可能出现bug
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                #计算要处理的token数量
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            #如果当前batch计算的token数量小于当前请求需要处理的token数量，则只有当前为第一个seq的时候才允许prefill
            #并且此次为chunked prefill,即允许分块，但只允许本轮第一个被选中的请求分块，否则要等到能将所有token放进去的时候才允许加入
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            #为没有分配过block的seq分配block（前面保证了block够分）
            #prefill只会分配一次block，不管是否prefill chunked，也就是说scheduler会提前分配所需要的block
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            #这里取remaining的情况对应上面只有当前为第一个seq的时候才允许prefill
            #设置该轮执行的token数量
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            #调度执行完之后如果处理完所有的该seq的token，下一次调度的时候就在running状态了，就可以被decode调度了
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            #放入该轮调度任务列表中
            scheduled_seqs.append(seq)
        #prefill和decode在一次schedule调度中是分开的，并且prefill优先
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            #接收一个RUNNING状态的seq
            seq = self.running.popleft()
            #循环判断是否有free block可以分配给当前seq，知道可分配或者
            # （注意lazy delete只是用的可以复用，不急着删，但是free block的数量是实实在在的代表了空闲块的存在）
            while not self.block_manager.can_append(seq):
                #如果还有RUNNING状态的seq，则抢占（队列后的seq优先）
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                #完善block的控制信息，如果缺少block则分配
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        #这里不修改状态，seq的decode阶段完成才修改状态
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    #抢占一个seq，将它打回WAITING态并且释放其拥有的所有block
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            #计算hash值，kv cache写入之后才登记，并且如果pre chunked会导致不完整块不会登记hash值，完整之后再登记
            #但是prefill chunked的非整块数据还是会存储在kv cache里面，但是只有原始的seq可以使用，因为seq的block_table能找到对应block
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            #将生成的新token拼接上去
            seq.append_token(token_id)
            #碰到eos或者到达最大token长度，代表执行完毕
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                #seq转为FINISHED状态
                seq.status = SequenceStatus.FINISHED
                #lazy delete block
                self.block_manager.deallocate(seq)
                #移除running状态
                self.running.remove(seq)
