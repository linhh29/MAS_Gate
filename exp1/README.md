# Experiment 1 · Design-time orchestration

The supplied design implementation is retained under `building/`, `quantification/`, and `verl/`.

- `building/skill_mas/build.py`: workflow construction.
- `building/formalize/`: schema and workflow compilation.
- `quantification/`: node utility and communication measurement.
- `verl/mas_gate_rl/`: reward adapters, evaluation and training launchers.
- `verl/examples/mas_gate/run_mas_gate_grpo.sh`: training entry wrapper.

Configure `MAS_GATE_ROOT` to this experiment directory and supply the dataset/model locations expected by the chosen launcher. Example deployment paths in upstream scripts are placeholders; benchmark data and weights are not included. Runtime launchers may start or stop workers, so review the selected script for your cluster before running it.

Dependencies are listed in `quantification/requirements.txt` and the preserved verl setup files. The upstream framework documentation remains at `verl/README.md`. Its authors and copyright notices identify that dependency, not this anonymous project's authors.

## Reward integration

Node utility and communication scoring are implemented in `quantification/metrics.py` and `quantification/unified.py`. The trainer also contains group-wise dense/correctness reward fusion in `verl/mas_gate_rl/reward_fusion.py`.

The supplied `verl/mas_gate_rl/reward.py` is a wrapper around `_reward_impl.cpython-312.pyc`; that implementation was absent from the supplied files. This blocks training and evaluation entry points that import the reward adapter. Recover its Python source to restore the original workflow execution, dataset judging and trainer integration. The quantification source alone is not a drop-in implementation of those interfaces.
