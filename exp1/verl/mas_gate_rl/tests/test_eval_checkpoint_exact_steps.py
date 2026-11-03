import json
from pathlib import Path
import pytest
from mas_gate_rl import eval_checkpoint_sweep as sweep


def _model(root: Path, step: int):
    p = root / f'global_step_{step}' / 'actor' / 'huggingface'
    p.mkdir(parents=True)
    (p / 'config.json').write_text('{}')
    (p / 'model.safetensors').write_bytes(b'x')


def test_discovery_reads_native_global_step_layout(tmp_path):
    _model(tmp_path, 4); _model(tmp_path, 40)
    rows = sweep.discover_local_checkpoints(tmp_path, 'deepscaler', '1')
    assert [r['step'] for r in rows] == [4, 40]
    assert rows[0]['checkpoint_layout'] == 'verl_global_step'
    assert rows[-1]['kind'] == 'final'


def test_requested_steps_fail_fast_when_any_step_is_missing(tmp_path):
    records = [
        {"step": 4, "model_path": str(tmp_path / "step4")},
        {"step": 11, "model_path": str(tmp_path / "step11")},
    ]
    with pytest.raises(RuntimeError, match=r"Missing requested checkpoint steps.*\[7\].*discovered=\[4, 11\]"):
        sweep.select_requested_checkpoints(records, {4, 7, 11}, tmp_path)


def test_requested_steps_ignore_only_unrequested_extras(tmp_path):
    records = [{"step": step} for step in (4, 7, 11, 18)]
    selected, extras = sweep.select_requested_checkpoints(records, {4, 7, 11}, tmp_path)
    assert [row["step"] for row in selected] == [4, 7, 11]
    assert extras == [18]
