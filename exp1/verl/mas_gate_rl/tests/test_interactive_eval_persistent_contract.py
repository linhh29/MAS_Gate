from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "datasets" / "run_local_alfworld_scienceworld_gate8b30b.sh"
RING = ROOT / "datasets" / "interactive_env" / "vllm_ring_controller.py"


def test_alfworld_scienceworld_formal_eval_is_persistent_only_and_training_aligned() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'VLLM_RING_BACKEND="${VLLM_RING_BACKEND:-sleep_wake}"' in text
    assert 'VLLM_RING_BACKEND" != "sleep_wake"' in text
    assert "--backend sleep_wake --sleep-level" in text
    assert "--enable-sleep-mode" in text
    assert 'SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-1}"' in text
    assert 'EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-1024}"' in text
    assert 'ORCH_MAX_NUM_SEQS="${ORCH_MAX_NUM_SEQS:-1024}"' in text
    assert 'SUB_MAX_NUM_SEQS="${SUB_MAX_NUM_SEQS:-1024}"' in text
    assert "--no-semantic-health" not in text
    assert "--fresh-" not in text
    assert "VLLM_FRESH" not in text
    assert "vllm_fresh" not in text


def test_interactive_formal_eval_checks_initial_and_after_wake_semantics() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "vllm_semantic_health.py" in text
    assert 'semantic_health orchestra "$ORCH_PORT" "$ORCH_MODEL_NAME" initial' in text
    assert 'semantic_health subagent "$SUB_PORT" "$SUB_MODEL_NAME" initial' in text
    assert 'semantic_health orchestra "$ORCH_PORT" "$ORCH_MODEL_NAME" after_wake' in text


def test_ring_checks_three_semantics_after_every_persistent_wake() -> None:
    text = RING.read_text(encoding="utf-8")
    assert 'await self._post(new, "/wake_up")' in text
    assert "await self._semantic_health(new)" in text
    assert '"Reply with exactly: HEALTHY"' in text
    assert '"What is 2+2? Reply with exactly: 4"' in text
    assert 'Return exactly this JSON object with no markdown:' in text
    assert 'finish_reason != "stop"' in text
