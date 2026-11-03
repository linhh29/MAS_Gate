from __future__ import annotations

import asyncio
import json
import math
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from mas_gate_rl import reward
from building.skill_mas import openai_async_client
from building.skill_mas.openai_async_client import AsyncOpenAIClient


class TerminalPolicyRewardTest(unittest.TestCase):
    def test_malformed_rollout_propagates_terminal_penalty_across_all_phases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_terminal_reward_") as td:
            root = Path(td)
            phase_file = root / "phase.json"
            info = {
                "dataset": "bcp",
                "question": "A local test question",
                "index": 7,
                "task_id": "bcp-7",
            }
            solution = "this is not Orchestra XML"
            env = {
                "MAS_GATE_REWARD_TRACE_ROOT": str(root / "traces"),
                "MAS_GATE_REWARD_PHASE_FILE": str(phase_file),
                "MAS_GATE_REWARD_ERROR_SCORE": "-1.0",
            }
            with patch.dict(os.environ, env, clear=False):
                phase_file.write_text('{"phase":"sub_agent"}', encoding="utf-8")
                phase1 = asyncio.run(
                    reward._compute_score_async("bcp", solution, "answer", info)
                )
                self.assertEqual(phase1["score"], 0.0)
                self.assertEqual(phase1["metric_key"], "sub_agent_invalid_policy")
                execution_path = Path(phase1["execution_sidecar"])
                execution = json.loads(execution_path.read_text(encoding="utf-8"))
                self.assertEqual(execution["status"], "invalid_policy_output")
                self.assertEqual(execution["terminal_score"], -1.0)
                self.assertNotIn("error", phase1)

                phase_file.write_text('{"phase":"qwen_j"}', encoding="utf-8")
                with patch.object(
                    reward,
                    "_get_quant_engine",
                    side_effect=AssertionError("Qwen J must be skipped for terminal policy output"),
                ):
                    phase2 = asyncio.run(
                        reward._compute_score_async("bcp", solution, "answer", info)
                    )
                self.assertEqual(phase2["score"], 0.0)
                self.assertEqual(phase2["metric_key"], "qwen_j_skipped_invalid_policy")
                self.assertTrue(Path(phase2["j_phase_sidecar"]).is_file())
                self.assertNotIn("error", phase2)

                phase_file.write_text('{"phase":"bge_impact"}', encoding="utf-8")
                # A terminal J sidecar returns before importing/calling BGE quantification.
                phase3 = asyncio.run(
                    reward._compute_score_async("bcp", solution, "answer", info)
                )
                self.assertEqual(phase3["score"], -1.0)
                self.assertEqual(phase3["metric_key"], "invalid_policy_output")
                self.assertEqual(phase3["quant_result"]["mean_impact"], -1.0)
                self.assertNotIn("error", phase3)

                normalized = reward._finalize_reward_result(
                    phase3,
                    data_source="bcp",
                    solution_str=solution,
                    extra_info=info,
                )
                self.assertEqual(normalized["score"], -1.0)
                self.assertEqual(normalized["metric_key"], "invalid_policy_output")
                self.assertEqual(normalized["error"], "")

    def test_full_phase_malformed_rollout_is_policy_penalty_not_service_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_full_reward_") as td:
            phase_file = Path(td) / "phase.json"
            phase_file.write_text('{"phase":"full"}', encoding="utf-8")
            env = {
                "MAS_GATE_REWARD_TRACE_ROOT": str(Path(td) / "traces"),
                "MAS_GATE_REWARD_PHASE_FILE": str(phase_file),
            }
            with patch.dict(os.environ, env, clear=False):
                result = asyncio.run(
                    reward._compute_score_async(
                        "aime", "malformed", "42", {"index": 1, "task_id": "aime-1"}
                    )
                )
            self.assertEqual(result["score"], -1.0)
            self.assertEqual(result["metric_key"], "invalid_policy_output")
            self.assertNotIn("error", result)


    def test_execution_failure_propagates_terminal_penalty_across_phases(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_execution_failure_") as td:
            root = Path(td)
            trace_root = root / "traces"
            phase_file = root / "phase.json"
            solution = "<orchestra><agents/></orchestra>"
            info = {"dataset": "aime", "question": "q", "index": 9, "task_id": "aime-9"}
            env = {
                "MAS_GATE_REWARD_TRACE_ROOT": str(trace_root),
                "MAS_GATE_REWARD_PHASE_FILE": str(phase_file),
                "MAS_GATE_REWARD_ERROR_SCORE": "-1.0",
            }
            with patch.dict(os.environ, env, clear=False):
                trace_path = trace_root / "trace.json"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text("{}", encoding="utf-8")
                execution_path = reward._write_execution_sidecar(
                    trace_root=trace_root,
                    idx=9,
                    task_id="aime-9",
                    solution_str=solution,
                    trace_path=trace_path,
                    judge_score=0.0,
                    success=False,
                    failure_reason="temporary sub-agent transport failure",
                )
                execution = json.loads(execution_path.read_text(encoding="utf-8"))
                self.assertEqual(execution["status"], "execution_failed")
                self.assertEqual(execution["terminal_score"], -1.0)
                self.assertIn("transport failure", execution["terminal_reason"])

                phase_file.write_text('{"phase":"qwen_j"}', encoding="utf-8")
                with patch.object(
                    reward,
                    "_get_quant_engine",
                    side_effect=AssertionError("Qwen J must be skipped after failed execution"),
                ):
                    phase2 = asyncio.run(
                        reward._compute_score_async("aime", solution, "42", info)
                    )
                self.assertEqual(phase2["score"], 0.0)
                self.assertTrue(Path(phase2["j_phase_sidecar"]).is_file())

                phase_file.write_text('{"phase":"bge_impact"}', encoding="utf-8")
                phase3 = asyncio.run(
                    reward._compute_score_async("aime", solution, "42", info)
                )
                self.assertEqual(phase3["score"], -1.0)
                self.assertEqual(phase3["metric_key"], "execution_failed")
                self.assertEqual(phase3["quant_result"]["status"], "execution_failed")



class RewardSidecarRecoveryTest(unittest.TestCase):
    def test_qwen_j_missing_trace_recovers_correctness_and_writes_terminal_sidecar(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_qwen_j_recovery_") as td:
            root = Path(td)
            trace_root = root / "traces"
            phase_file = root / "phase.json"
            solution = "<orchestra><agents/></orchestra>"
            info = {"dataset": "bcp", "question": "q", "index": 244, "task_id": "64"}
            missing_trace = trace_root / "sample_trace_json" / "missing.json"
            env = {
                "MAS_GATE_REWARD_TRACE_ROOT": str(trace_root),
                "MAS_GATE_REWARD_PHASE_FILE": str(phase_file),
                "MAS_GATE_REWARD_ERROR_SCORE": "-1.0",
                "MAS_GATE_REWARD_SIDECAR_READ_RETRIES": "1",
                "MAS_GATE_REWARD_SIDECAR_READ_INTERVAL": "0",
            }
            with patch.dict(os.environ, env, clear=False):
                reward._write_execution_sidecar(
                    trace_root=trace_root, idx=244, task_id="64", solution_str=solution,
                    trace_path=missing_trace, judge_score=0.0, success=True,
                    judge_score_available=False, exact_score=0.0,
                )
                reward._write_json_sidecar(
                    reward._bcp_judge_sidecar_path(trace_root, 244, "64", solution),
                    {
                        "schema_version": "mas_gate_bcp_llm_judge_state/1",
                        "status": "ok", "trace_json": str(missing_trace),
                        "judge_score": 1.0, "judge_score_available": True,
                    },
                )

                phase_file.write_text('{"phase":"qwen_j"}', encoding="utf-8")
                qwen_j = asyncio.run(
                    reward._compute_score_async("bcp", solution, "answer", info)
                )
                self.assertEqual(qwen_j["score"], -1.0)
                self.assertEqual(qwen_j["judge_score"], 1.0)
                self.assertTrue(qwen_j["error"])
                j_path = Path(qwen_j["j_phase_sidecar"])
                j_state = json.loads(j_path.read_text(encoding="utf-8"))
                self.assertEqual(j_state["status"], "qwen_j_failed")
                self.assertEqual(j_state["terminal_score"], -1.0)
                self.assertEqual(j_state["judge_score"], 1.0)

                phase_file.write_text('{"phase":"bge_impact"}', encoding="utf-8")
                bge = asyncio.run(
                    reward._compute_score_async("bcp", solution, "answer", info)
                )
                self.assertEqual(bge["score"], -1.0)
                self.assertEqual(bge["judge_score"], 1.0)
                self.assertEqual(bge["metric_key"], "qwen_j_failed")
                self.assertTrue(bge["quant_error"])
                self.assertNotIn("error", bge)

    def test_finalize_error_without_judge_score_is_finite_zero(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_finite_correctness_") as td:
            with patch.dict(os.environ, {"MAS_GATE_REWARD_TRACE_ROOT": td}, clear=False):
                normalized = reward._finalize_reward_result(
                    {"score": -1.0, "metric_key": "error_fallback", "error": "x"},
                    data_source="bcp", solution_str="solution",
                    extra_info={"index": 1, "task_id": "1"},
                )
        self.assertEqual(normalized["judge_score"], 0.0)
        self.assertTrue(math.isfinite(normalized["judge_score"]))

    def test_concurrent_same_path_json_sidecar_writes_remain_parseable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mas_gate_atomic_sidecar_") as td:
            path = Path(td) / "same.json"
            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(lambda i: reward._write_json_sidecar(path, {"writer": i}), range(128)))
            final = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn(final["writer"], range(128))
            self.assertEqual(list(path.parent.glob(path.name + ".tmp.*")), [])


class ExplicitHttpClientTest(unittest.TestCase):
    def test_skill_mas_explicit_http_client_closes_deterministically(self) -> None:
        async def exercise() -> None:
            client = AsyncOpenAIClient(
                "Qwen3-30B-A3B-Instruct-2507",
                api_key="EMPTY",
                base_url="http://127.0.0.1:9/v1",
                explicit_http_client=True,
            )
            raw = client._explicit_http_client
            self.assertIsNotNone(raw)
            self.assertFalse(raw.is_closed)
            await client.aclose()
            self.assertTrue(raw.is_closed)

        asyncio.run(exercise())

    def test_bcp_native_chat_client_uses_plain_httpx_client(self) -> None:
        async def exercise() -> None:
            with patch.dict(os.environ, {
                "MAS_GATE_SUB_AGENT_BASE_URL": "http://127.0.0.1:9/v1",
                "MAS_GATE_SUB_AGENT_API_KEY": "EMPTY",
            }):
                client = reward._make_openai_chat_client("Qwen3-30B-A3B-Instruct-2507")
                raw = client._client
                # It must not be OpenAI's AsyncHttpxClientWrapper with an async __del__.
                self.assertEqual(type(raw).__module__, "httpx")
                self.assertEqual(type(raw).__name__, "AsyncClient")
                await client.close()
                self.assertTrue(raw.is_closed)

        asyncio.run(exercise())


class ContextDiagnosticsTest(unittest.TestCase):
    class _CharTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return list(range(sum(len(str(m.get("content", ""))) for m in messages) + 10))

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) for ch in text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

    def test_context_compaction_records_removed_tokens_and_effective_budget(self) -> None:
        messages = [{"role": "user", "content": "x" * 300}]
        diagnostics = {}
        env = {
            "MAS_GATE_SUB_AGENT_MAX_MODEL_LEN": "300",
            "MAS_GATE_SUB_AGENT_CONTEXT_SAFETY_TOKENS": "20",
            "MAS_GATE_SUB_AGENT_MAX_TOKENS": "100",
            "MAS_GATE_SUB_AGENT_MIN_COMPLETION_TOKENS": "60",
        }
        with patch.dict(os.environ, env, clear=False), patch.object(
            openai_async_client,
            "_mas_gate_local_tokenizer",
            return_value=self._CharTokenizer(),
        ):
            effective, prompt_tokens = openai_async_client._mas_gate_cap_completion_to_context(
                messages, 100, diagnostics
            )
        self.assertTrue(diagnostics["context_compacted"])
        self.assertGreater(diagnostics["user_tokens_removed"], 0)
        self.assertEqual(diagnostics["prompt_tokens_after_compaction"], prompt_tokens)
        self.assertEqual(diagnostics["effective_max_tokens"], effective)
        self.assertEqual(effective, 100)
        self.assertTrue(diagnostics["output_budget_excludes_prompt"])
        self.assertIn("MAS-GATE CONTEXT COMPACTION", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
