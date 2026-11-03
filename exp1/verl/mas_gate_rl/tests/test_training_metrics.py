import json

from omegaconf import OmegaConf

from mas_gate_rl.plot_training_metrics import merge_metric_rows, parse_metrics_jsonl
from mas_gate_rl.training_metrics import snapshot_checkpoint_metrics


def test_snapshot_checkpoint_metrics_follows_hf_weights(tmp_path):
    history = tmp_path / "training_metrics.jsonl"
    history.write_text('{"step":1,"data":{"actor/pg_loss":0.25}}\n', encoding="utf-8")
    config = OmegaConf.create({"algorithm": {"adv_estimator": "grpo"}})

    written = snapshot_checkpoint_metrics(
        checkpoint_root=tmp_path / "ckpt",
        step=1,
        metrics={"actor/pg_loss": 0.25, "actor/kl_loss": float("nan")},
        config=config,
        history_path=history,
    )

    root_metrics = tmp_path / "ckpt/global_step_1/metrics"
    hf_metrics = tmp_path / "ckpt/global_step_1/actor/huggingface"
    assert len(written) == 6
    assert (root_metrics / "training_metrics.jsonl").read_text() == history.read_text()
    assert (hf_metrics / "training_metrics.jsonl").read_text() == history.read_text()
    latest = json.loads((hf_metrics / "latest_training_metrics.json").read_text())
    assert latest["step"] == 1
    assert latest["data"]["actor/pg_loss"] == 0.25
    assert latest["data"]["actor/kl_loss"] is None
    saved_config = json.loads((hf_metrics / "training_config.json").read_text())
    assert saved_config["algorithm"]["adv_estimator"] == "grpo"


def test_snapshot_checkpoint_metrics_degrades_to_latest_on_history_eio(tmp_path, monkeypatch):
    history = tmp_path / "training_metrics.jsonl"
    history.write_text('{"step":0,"data":{}}\n', encoding="utf-8")
    original_read_bytes = type(history).read_bytes

    def fail_only_history(self):
        if self == history:
            raise OSError(5, "simulated PCache EIO")
        return original_read_bytes(self)

    monkeypatch.setenv("MAS_GATE_METRICS_IO_RETRIES", "2")
    monkeypatch.setenv("MAS_GATE_METRICS_IO_RETRY_SECONDS", "0")
    monkeypatch.setattr(type(history), "read_bytes", fail_only_history)
    written = snapshot_checkpoint_metrics(
        checkpoint_root=tmp_path / "ckpt",
        step=3,
        metrics={"actor/loss": -0.1},
        config=OmegaConf.create({"trainer": {"total_epochs": 1}}),
        history_path=history,
    )
    assert len(written) == 6
    copied = (tmp_path / "ckpt/global_step_3/metrics/training_metrics.jsonl").read_text()
    assert json.loads(copied) == {"step": 3, "data": {"actor/loss": -0.1}}


def test_plot_training_metrics_parses_checkpoint_jsonl():
    rows = parse_metrics_jsonl([
        '{"step":1,"data":{"actor/loss":-0.1,"actor/lr":5e-7,"metadata":"ignored","bad":null}}\n'
        '{"step":2,"data":{"actor/loss":-0.2,"reward_fusion/mixed_group_ratio":0.5}}\n'
    ])
    assert rows == [
        {"step": 1.0, "actor/loss": -0.1, "actor/lr": 5e-7},
        {"step": 2.0, "actor/loss": -0.2, "reward_fusion/mixed_group_ratio": 0.5},
    ]


def test_plot_training_metrics_prefers_more_complete_row():
    merged = merge_metric_rows(
        [{"step": 1.0, "actor/loss": -0.1}],
        [{"step": 1.0, "actor/loss": -0.2, "actor/lr": 5e-7}],
    )
    assert merged == [{"step": 1.0, "actor/loss": -0.2, "actor/lr": 5e-7}]


def test_plot_training_metrics_rejects_malformed_jsonl():
    import pytest

    with pytest.raises(ValueError, match="invalid metrics JSONL"):
        parse_metrics_jsonl(['{"step":1,"data":'])
