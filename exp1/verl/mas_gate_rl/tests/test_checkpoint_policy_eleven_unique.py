from mas_gate_rl.checkpoint_policy import checkpoint_steps, intermediate_checkpoint_steps


def test_full_40_step_run_has_ten_unique_prefinal_plus_final():
    intermediate = intermediate_checkpoint_steps(40, 10)
    assert intermediate == [4, 7, 11, 15, 18, 22, 25, 29, 33, 36]
    assert len(intermediate) == len(set(intermediate)) == 10
    assert all(step < 40 for step in intermediate)
    assert checkpoint_steps(40, 10) == intermediate + [40]
    assert len(checkpoint_steps(40, 10)) == 11
