"""Real CPU scheduler/block ownership checks; no model or Gloo substitute."""

from types import SimpleNamespace
from unittest import TestCase

import pytest

pytestmark = pytest.mark.hf_reference


def vl_scheduler(**overrides):
    from nanovllm.engine.scheduler import Scheduler

    values = dict(
        is_multimodal=True,
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        eos=99,
        eos_token_ids=frozenset({98, 99}),
        kvcache_block_size=256,
        num_kvcache_blocks=4,
    )
    values.update(overrides)
    return Scheduler(SimpleNamespace(**values))


def sequence(length=257, max_tokens=3):
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams

    return Sequence([1] * length, SamplingParams(max_tokens=max_tokens))


def pool_state(manager):
    return (
        list(manager.free_block_ids),
        set(manager.used_block_ids),
        [(b.ref_count, b.hash, b.token_ids[:]) for b in manager.blocks],
        dict(manager.hash_to_block_id),
    )


def assert_released(scheduler):
    manager = scheduler.block_manager
    assert scheduler.is_finished()
    assert not manager.used_block_ids
    assert len(manager.free_block_ids) == len(manager.blocks)
    assert all(b.ref_count == 0 for b in manager.blocks)


class SchedulerTests(TestCase):
    def test_serial_admission_and_capacity(self):
        # Waiting-first would steal a second pair of pages before Decode.
        scheduler = vl_scheduler()
        first, second = sequence(), sequence()
        scheduler.add(first)
        scheduler.add(second)
        scheduled, prefill = scheduler.schedule()
        self.assertEqual(scheduled, [first])
        self.assertTrue(prefill)
        self.assertEqual(first.num_scheduled_tokens, 257)
        self.assertEqual(second.block_table, [])
        scheduler.postprocess(scheduled, [8], prefill)
        scheduled, prefill = scheduler.schedule()
        self.assertEqual(scheduled, [first])
        self.assertFalse(prefill)
        self.assertEqual(second.block_table, [])
        self.assertEqual(list(scheduler.waiting), [second])
        self.assertEqual(list(scheduler.running), [first])
        self.assertEqual(scheduler.block_manager.hash_to_block_id, {})
        scheduler.postprocess(scheduled, [9], prefill)
        scheduled, prefill = scheduler.schedule()
        self.assertEqual(scheduled, [first])
        self.assertFalse(prefill)
        scheduler.postprocess(scheduled, [10], prefill)
        self.assertTrue(first.is_finished)
        self.assertEqual(second.block_table, [])
        self.assertFalse(scheduler.block_manager.used_block_ids)
        scheduled, prefill = scheduler.schedule()
        self.assertEqual(scheduled, [second])
        self.assertTrue(prefill)
        self.assertEqual(second.num_scheduled_tokens, 257)
        scheduler.postprocess(scheduled, [98], prefill)
        assert_released(scheduler)

    def test_no_prefix_lookup_or_publish(self):
        # Identical image placeholder IDs must not reuse the previous image KV.
        scheduler = vl_scheduler()
        for _ in range(2):
            seq = sequence(513, max_tokens=2)
            scheduler.add(seq)
            scheduled, prefill = scheduler.schedule()
            self.assertEqual(seq.num_cached_tokens, 0)
            self.assertEqual(seq.num_scheduled_tokens, 513)
            scheduler.postprocess(scheduled, [8], prefill)
            self.assertEqual(scheduler.block_manager.hash_to_block_id, {})
            scheduled, prefill = scheduler.schedule()
            scheduler.postprocess(scheduled, [9], prefill)
            self.assertEqual(scheduler.block_manager.hash_to_block_id, {})
            self.assertTrue(all(b.hash == -1 for b in scheduler.block_manager.blocks))
            assert_released(scheduler)


@pytest.mark.parametrize(
    "length,max_tokens,limits,reason",
    [
        (256, 1, dict(num_kvcache_blocks=1), "KV"),
        (257, 256, dict(num_kvcache_blocks=2), "KV"),
        (256, 1, dict(max_model_len=256), "max_model_len"),
        (257, 1, dict(max_num_batched_tokens=256), "max_num_batched_tokens"),
    ],
)
def test_impossible_request_rejected_without_mutation(
    length, max_tokens, limits, reason
):
    from nanovllm.engine.sequence import SequenceStatus

    scheduler = vl_scheduler(**limits)
    seq = sequence(length, max_tokens)
    before = pool_state(scheduler.block_manager)
    with pytest.raises(ValueError, match=reason):
        scheduler.add(seq)
    assert pool_state(scheduler.block_manager) == before
    assert seq.status == SequenceStatus.WAITING and seq.block_table == []
    assert_released(scheduler)


@pytest.mark.parametrize("length", [255, 256, 257])
def test_page_boundaries_and_repeated_queueing(length):
    # Fit total capacity exactly, even while another request temporarily owns it.
    scheduler = vl_scheduler(num_kvcache_blocks=2, max_model_len=512)
    requests = [sequence(length) for _ in range(3)]
    for seq in requests:
        scheduler.add(seq)
    for index, seq in enumerate(requests):
        for step in range(3):
            scheduled, prefill = scheduler.schedule()
            assert scheduled == [seq] and prefill == (step == 0)
            assert seq.num_scheduled_tokens == (length if prefill else 1)
            assert len(scheduler.running) == 1
            assert all(other.block_table == [] for other in requests[index + 1 :])
            # Hand-derived page counts for all three boundary paths.
            expected = {255: [1, 1, 2], 256: [1, 2, 2], 257: [2, 2, 2]}
            assert len(seq.block_table) == expected[length][step]
            assert all(
                scheduler.block_manager.blocks[b].ref_count == 1
                for b in seq.block_table
            )
            scheduler.postprocess(scheduled, [8], prefill)
        assert seq.num_completion_tokens == 3 and seq.is_finished
        assert not scheduler.block_manager.used_block_ids
    assert_released(scheduler)


def test_add_uses_total_pool_not_current_free_capacity():
    scheduler = vl_scheduler(num_kvcache_blocks=2, max_model_len=512)
    first, second = sequence(), sequence()
    scheduler.add(first)
    scheduled, prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [8], prefill)
    assert not scheduler.block_manager.free_block_ids
    before = pool_state(scheduler.block_manager)
    scheduler.add(second)
    assert pool_state(scheduler.block_manager) == before
    scheduler.abort(first)
    scheduled, prefill = scheduler.schedule()
    assert scheduled == [second] and prefill
    scheduler.postprocess(scheduled, [99], prefill)
    assert_released(scheduler)


@pytest.mark.parametrize(
    "phase", ["waiting", "scheduled_prefill", "decode", "finished"]
)
def test_abort_is_idempotent_and_keeps_other_request(phase):
    scheduler = vl_scheduler(num_kvcache_blocks=2)
    first, second = sequence(), sequence(max_tokens=1)
    scheduler.add(first)
    scheduler.add(second)
    if phase != "waiting":
        scheduled, prefill = scheduler.schedule()
        if phase in ("decode", "finished"):
            scheduler.postprocess(
                scheduled, [99 if phase == "finished" else 8], prefill
            )
    scheduler.abort(first)
    after = pool_state(scheduler.block_manager)
    scheduler.abort(first)
    assert pool_state(scheduler.block_manager) == after
    assert first.is_finished and first.block_table == []
    assert first.num_cached_tokens == 0 and first.num_scheduled_tokens == 0
    assert list(scheduler.waiting) == [second] and not scheduler.running
    assert not scheduler.block_manager.used_block_ids
    scheduled, prefill = scheduler.schedule()
    assert scheduled == [second] and prefill
    scheduler.postprocess(scheduled, [8], prefill)
    assert second.num_completion_tokens == 1
    assert_released(scheduler)


@pytest.mark.parametrize("phase", ["admission", "decode"])
def test_capacity_invariant_raises_without_preemption_and_can_abort(phase):
    from nanovllm.engine.sequence import SequenceStatus

    scheduler = vl_scheduler(num_kvcache_blocks=2)
    seq, queued = sequence(256), sequence(255, max_tokens=1)
    scheduler.add(seq)
    scheduler.add(queued)
    manager = scheduler.block_manager
    if phase == "decode":
        scheduled, prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [8], prefill)
    # Inject real ownership to simulate a broken pool invariant, without mocking it.
    outsider = sequence(1)
    manager.allocate(outsider, manager.can_allocate(outsider))
    before = pool_state(manager)
    with pytest.raises(RuntimeError, match="(?i)KV.*invariant"):
        scheduler.schedule()
    assert pool_state(manager) == before
    assert queued.block_table == []
    assert seq.status == (
        SequenceStatus.RUNNING if phase == "decode" else SequenceStatus.WAITING
    )
    scheduler.abort(seq)
    scheduler.abort(seq)
    manager.deallocate(outsider)
    scheduled, prefill = scheduler.schedule()
    assert scheduled == [queued] and prefill
    scheduler.postprocess(scheduled, [8], prefill)
    assert_released(scheduler)


def test_disabled_prefix_cache_does_not_look_up_existing_entry():
    from nanovllm.engine.block_manager import BlockManager

    manager = BlockManager(2, 256, enable_prefix_cache=False)
    seq = sequence()
    # A valid stale entry makes an accidental lookup observable independently
    # of the separate publication test.
    token_ids = [1] * 256
    h = manager.compute_hash(token_ids)
    manager.blocks[0].update(h, token_ids)
    manager.hash_to_block_id[h] = 0
    assert manager.can_allocate(seq) == 0
    manager.allocate(seq, 0)
    assert seq.num_cached_tokens == 0
    seq.num_scheduled_tokens = 257
    manager.hash_blocks(seq)
    assert manager.hash_to_block_id == {}
    manager.deallocate(seq)
    assert not manager.used_block_ids
    assert all(b.ref_count == 0 for b in manager.blocks)
