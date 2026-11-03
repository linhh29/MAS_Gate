from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BCP_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "BrowseComp-Plus"
if str(BCP_ROOT) not in sys.path:
    sys.path.insert(0, str(BCP_ROOT))

import retrieval_flow


class _CharTokenizer:
    def __init__(self):
        self.last_call = None

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
        return_attention_mask=True,
        return_token_type_ids=True,
    ):
        self.last_call = {
            "truncation": truncation,
            "max_length": max_length,
            "return_attention_mask": return_attention_mask,
            "return_token_type_ids": return_token_type_ids,
        }
        ids = [ord(ch) for ch in str(text)]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in str(text)]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class _BorrowCheckingTokenizer(_CharTokenizer):
    def __init__(self):
        super().__init__()
        self._borrowed = threading.Lock()

    def __call__(self, *args, **kwargs):
        if not self._borrowed.acquire(blocking=False):
            raise RuntimeError("Already borrowed")
        try:
            time.sleep(0.002)
            return super().__call__(*args, **kwargs)
        finally:
            self._borrowed.release()

    def decode(self, *args, **kwargs):
        if not self._borrowed.acquire(blocking=False):
            raise RuntimeError("Already borrowed")
        try:
            time.sleep(0.001)
            return super().decode(*args, **kwargs)
        finally:
            self._borrowed.release()


class _FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, topk):
        self.calls.append((query, topk))
        r = len(self.calls)
        return [
            SimpleNamespace(docid=f"r{r}-d{i}", text=(f"R{r}D{i}" * 5000))
            for i in range(topk)
        ]


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        action = "final" if self.calls >= 5 else "search"
        distinct_queries = [
            "alpha academy merger history",
            "beta latin school founding",
            "gamma girls institution predecessor",
            "delta town education chronology",
        ]
        content = (
            '{"action":"final","search_query":"","reason":"done"}'
            if action == "final"
            else '{"action":"search","search_query":"%s","reason":"gap"}'
            % distinct_queries[self.calls - 1]
        )
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        )


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class BrowseCompRetrievalTokenBudgetTest(unittest.TestCase):
    def test_no_whitespace_document_is_true_512_token_truncated(self) -> None:
        tok = _CharTokenizer()
        text = "x" * 100_000
        out = retrieval_flow._truncate_text_by_tokens(text, 512, tokenizer=tok)
        self.assertEqual(len(tok.encode(out, add_special_tokens=False)), 512)
        self.assertEqual(tok.last_call["max_length"], 513)
        self.assertTrue(tok.last_call["truncation"])
        self.assertFalse(tok.last_call["return_attention_mask"])
        self.assertFalse(tok.last_call["return_token_type_ids"])

    def test_concurrent_first_load_is_coalesced(self) -> None:
        retrieval_flow._bcp_local_tokenizer_cached.cache_clear()
        calls = 0
        calls_lock = threading.Lock()
        sentinel = object()

        def slow_load(_model_path):
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.02)
            return sentinel

        try:
            with patch.object(retrieval_flow, "_load_bcp_local_tokenizer", side_effect=slow_load):
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
                    loaded = list(
                        pool.map(
                            retrieval_flow._bcp_local_tokenizer,
                            ["concurrent-test-model"] * 128,
                        )
                    )
            self.assertEqual(calls, 1)
            self.assertTrue(all(item is sentinel for item in loaded))
        finally:
            retrieval_flow._bcp_local_tokenizer_cached.cache_clear()

    def test_mixed_token_limits_do_not_concurrently_borrow_fast_tokenizer(self) -> None:
        tok = _BorrowCheckingTokenizer()

        def truncate(i):
            limit = 24 if i % 2 else 512
            text = ("x" if i % 3 else "abc def ") * 2000
            return retrieval_flow._truncate_text_by_tokens(text, limit, tokenizer=tok)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            outputs = list(pool.map(truncate, range(128)))
        self.assertEqual(len(outputs), 128)
        for i, output in enumerate(outputs):
            limit = 24 if i % 2 else 512
            self.assertLessEqual(len(tok.encode(output, add_special_tokens=False)), limit)

    def test_missing_tokenizer_fails_closed_without_approx_fallback(self) -> None:
        with patch.object(retrieval_flow, "_bcp_local_tokenizer", side_effect=OSError("missing")), patch.dict(
            retrieval_flow.os.environ,
            {"MAS_GATE_BCP_ALLOW_APPROX_TOKEN_FALLBACK": "0"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "requires the local Qwen tokenizer"):
                retrieval_flow._truncate_text_by_tokens("x" * 1000, 512)

    def test_five_rounds_keep_each_document_within_512_real_tokens(self) -> None:
        retriever = _FakeRetriever()
        client = _FakeClient()
        tok = _CharTokenizer()
        with patch.object(retrieval_flow, "_bcp_local_tokenizer", return_value=tok):
            contexts, trajectory, usage = asyncio.run(
                retrieval_flow.multi_round_retrieve_contexts(
                    client=client,
                    model="Qwen3-30B-A3B-Instruct-2507",
                    question="test question",
                    retriever=retriever,
                    retrieval_topk=5,
                    doc_max_tokens=512,
                    max_rounds=5,
                    evidence_policy="accumulated",
                    max_final_snippets=15,
                )
            )
        self.assertEqual(len(trajectory), 5)
        self.assertEqual(len(retriever.calls), 5)
        self.assertEqual(len(contexts), 15)
        self.assertEqual(usage["prompt_tokens"], 50)
        for block in contexts:
            content = block.split("] ", 1)[1]
            self.assertLessEqual(len(tok.encode(content, add_special_tokens=False)), 512)


if __name__ == "__main__":
    unittest.main()
