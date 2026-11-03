from __future__ import annotations

import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

from quantification.clients.openai_client import (
    OpenAICompatibleClient,
    _select_echoed_answer_logprobs,
)
from quantification.graph.structure import GraphStructure
from quantification.schemas import DependencyEdge, NodeOutput, ReasoningGraph
from quantification.utility.j_scorer import JScorer


class _CountingClient:
    def __init__(self) -> None:
        self.contexts: list[str] = []

    def format_context(self, question: str, prefix_outputs: list[str]) -> str:
        return "\n".join([question, *prefix_outputs])

    def compute_answer_cross_entropy(self, context: str, answer: str) -> float:
        del answer
        self.contexts.append(context)
        return float(len(self.contexts))


def test_dag_j_scoring_reuses_precomputed_baseline() -> None:
    graph = ReasoningGraph(
        question="question",
        ground_truth="42",
        nodes=[NodeOutput("a", "alpha"), NodeOutput("b", "beta")],
        edges=[DependencyEdge("a", "b")],
        execution_order=["a", "b"],
    )
    client = _CountingClient()
    scorer = JScorer(client, SimpleNamespace())

    after, before = scorer.compute_dag_aware_cross_entropies(
        graph, GraphStructure(graph), j_baseline=0.25
    )

    # Root context_before is exactly the question and must reuse J0 rather than
    # issuing a second Qwen30B request. The linear child also reuses J(a).
    assert client.contexts == ["question\nalpha", "question\nalpha\nbeta"]
    assert before["a"] == 0.25
    assert before["b"] == after["a"]


def test_exact_j_cache_coalesces_concurrent_duplicate_requests() -> None:
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.config = SimpleNamespace(request_timeout=1.0, max_retries=1)
    client._j_cache_max_entries = 16
    client._j_cache = OrderedDict()
    client._j_cache_lock = threading.Lock()
    client._j_inflight = {}

    calls = 0
    calls_lock = threading.Lock()

    def uncached(context: str, answer: str) -> float:
        nonlocal calls
        assert (context, answer) == ("same context", "42")
        with calls_lock:
            calls += 1
        time.sleep(0.1)
        return 3.25

    client._compute_answer_cross_entropy_uncached = uncached
    barrier = threading.Barrier(8)
    results: list[float] = []

    def worker() -> None:
        barrier.wait()
        results.append(client.compute_answer_cross_entropy("same context", "42"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert results == [3.25] * 8
    assert calls == 1
    # A later request is also served from the bounded exact cache.
    assert client.compute_answer_cross_entropy("same context", "42") == 3.25
    assert calls == 1


def test_select_echoed_answer_logprobs_uses_answer_character_span() -> None:
    # The newline token ends exactly at the answer boundary and must not be
    # included.  The two answer tokens are selected by their echoed offsets.
    prompt = "context\n-4"
    values, tokens = _select_echoed_answer_logprobs(
        rendered_prompt=prompt,
        answer="-4",
        echoed_text=prompt,
        tokens=["context", "\n", "-", "4"],
        token_logprobs=[None, -0.1, -4.25, -0.75],
        text_offsets=[0, 7, 8, 9],
    )
    assert tokens == ["-", "4"]
    assert values == [-4.25, -0.75]


def test_select_echoed_answer_logprobs_supports_boundary_overlap() -> None:
    # Some tokenizers merge leading whitespace with the first answer token.
    prompt = "Answer: 42"
    values, tokens = _select_echoed_answer_logprobs(
        rendered_prompt=prompt,
        answer="42",
        echoed_text=prompt,
        tokens=["Answer", ":", " 42"],
        token_logprobs=[None, -0.2, -1.5],
        text_offsets=[0, 6, 7],
    )
    assert tokens == [" 42"]
    assert values == [-1.5]
