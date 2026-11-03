from __future__ import annotations

from dataclasses import dataclass

from verl.experimental.reward_loop import reward_loop as reward_loop_module
from verl.experimental.reward_loop.reward_loop import RewardLoopManager


class FakeData:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return FakeData(self.values[item])


@dataclass(frozen=True)
class FakeRef:
    sequence: int


class FakeRemoteMethod:
    def __init__(self, worker, results):
        self.worker = worker
        self.results = results

    def remote(self, chunk):
        ref = FakeRef(len(self.results))
        values = list(chunk.values)
        self.worker.submissions.append(values)
        self.results[ref] = [{"sample": value, "worker": self.worker.index} for value in values]
        return ref


class FakeWorker:
    def __init__(self, index, results):
        self.index = index
        self.submissions = []
        self.compute_score_batch = FakeRemoteMethod(self, results)


def test_dynamic_scheduler_preserves_order_and_bounds_worker_batch(monkeypatch):
    results = {}
    workers = [FakeWorker(index, results) for index in range(3)]
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.reward_loop_workers = workers

    # Complete the newest submitted ref first to emulate workers finishing at
    # different speeds.  Output order must still match the input order.
    monkeypatch.setattr(
        reward_loop_module.ray,
        "wait",
        lambda refs, num_returns=1: ([max(refs, key=lambda ref: ref.sequence)], []),
    )
    monkeypatch.setattr(reward_loop_module.ray, "get", lambda ref: results[ref])

    outputs = manager._compute_score_batches_dynamic(FakeData(range(11)), chunk_size=2)

    assert [output["sample"] for output in outputs] == list(range(11))
    assert sum(len(worker.submissions) for worker in workers) == 6
    assert all(len(chunk) <= 2 for worker in workers for chunk in worker.submissions)
    assert any(len(worker.submissions) > 1 for worker in workers)


def test_dynamic_scheduler_handles_empty_input_without_ray_calls(monkeypatch):
    manager = RewardLoopManager.__new__(RewardLoopManager)
    manager.reward_loop_workers = [object()]
    monkeypatch.setattr(
        reward_loop_module.ray,
        "wait",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ray.wait should not be called")),
    )

    assert manager._compute_score_batches_dynamic(FakeData([]), chunk_size=16) == []
