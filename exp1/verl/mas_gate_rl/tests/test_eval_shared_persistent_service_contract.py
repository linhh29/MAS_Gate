from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_capacity_monitor_deduplicates_shared_physical_endpoint():
    text = (ROOT / "monitor_eval_capacity.py").read_text(encoding="utf-8")
    assert "ROLE_URLS=" in text
    assert "URL_ROLES.setdefault(_url,[]).append(_role)" in text
    assert "URLS={'+'.join(_roles):_url" in text


def test_semantic_eval_resolves_judge_alias_before_ports_and_cleanup():
    text = (ROOT / "run_orchestra_eval.sh").read_text(encoding="utf-8")
    alias = text.index('if [[ "$MAS_GATE_EVAL_DATASET" == bcp || "$MAS_GATE_EVAL_DATASET" == hotpotqa ]]')
    ports = text.index('PORTS=("$ORCHESTRA_PORT" "$SUBAGENT_PORT" "$JUDGE_PORT")')
    assert alias < ports
    assert 'UNIQUE_PORTS+=("$p")' in text
    assert 'for p in "${UNIQUE_PORTS[@]}"' in text
    assert '[[ "$MAS_GATE_EVAL_DATASET" != bcp ]] || port_free "$JUDGE_PORT"' not in text


def test_formal_eval_defaults_to_fresh_phase_restart():
    text = (ROOT / "run_orchestra_eval.sh").read_text(encoding="utf-8")
    assert 'MAS_GATE_EVAL_FRESH_PHASES=${MAS_GATE_EVAL_FRESH_PHASES:-1}' in text
    assert 'source "$THIS_VERL/mas_gate_rl/run_orchestra_eval_fresh_phases.sh"' in text
    assert 'lifecycle=fresh_phase_restart' in text
    # The legacy backend remains available only through explicit opt-in and
    # retains strict post-wake semantic checks.
    assert 'lifecycle=sleep_wake (explicit legacy opt-in)' in text
    assert '--enable-sleep-mode' in text
    assert '/sleep?level=${VLLM_SLEEP_LEVEL:-1}' in text


def test_formal_entrypoints_default_to_exact_eleven_steps():
    repo = ROOT.parents[1]
    expected = 'EVAL_CHECKPOINT_STEPS=${EVAL_CHECKPOINT_STEPS:-4,7,11,15,18,22,25,29,33,36,40}'
    for relative in (
        'datasets/run_eval_hotpotqa_orchestra.sh',
        'datasets/run_eval_bcp_orchestra.sh',
        'datasets/run_eval_deepscaler_orchestra.sh',
        'datasets/run_eval_gpqa_orchestra.sh',
    ):
        text = (repo / relative).read_text(encoding='utf-8')
        assert expected in text, relative
        assert 'ARGS+=(--steps "$EVAL_CHECKPOINT_STEPS")' in text, relative
