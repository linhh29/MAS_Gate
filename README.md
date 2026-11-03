# MAS-Gate

<div align="center">

**Unlocking Multi-Agent System Lifecycle via Node-Centric Quantification**

Design-time construction · Runtime intervention · Post-hoc attribution

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.17613)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/linhh29/MAS_Gate)
[![Project Blog](https://img.shields.io/badge/Blog-Project%20Page-0078D4?logo=readthedocs&logoColor=white)](https://linhh29.github.io/blog/MAS-Gate/index.html)
[![Online Demo](https://img.shields.io/badge/Demo-Online%20Demo-512BD4?logo=rocket&logoColor=white)](https://mas-gate-demo.hehailin.life/)

</div>

MAS-Gate uses node-level utility and communication influence to connect workflow construction, bounded runtime intervention, and failure attribution across the multi-agent system lifecycle. This repository contains research code for the three experiment phases.

**Start here:** try the [online demo](https://mas-gate-demo.hehailin.life/) to replay the paper case or run a live workflow, read the [project blog](https://linhh29.github.io/blog/MAS-Gate/index.html) for an illustrated overview, or follow the [experiment entry points](#experiment-entry-points) below to explore the code.

## Repository map

```text
exp1/                 Design-time workflow construction and training
  building/           Workflow planning and formalization
  quantification/     Utility and communication scoring
  verl/               Training implementation and preserved third-party notices
exp2/                 Runtime Editor (current implementation)
  gate_main/          Planning, preparation, GRPO training, prediction, edits
  gate_baselines/     Comparison-method reference implementations
  tests/              Dependency-free runtime/reward checks
  config.example.json Configuration with local placeholders
  pyproject.toml      Installable main and baseline packages
exp3/                 Post-hoc failure attribution
  mas_gate/           Quantification, local models and attribution CLI
  tests/              Attribution/model-contract tests
```

## Experiment entry points

| Experiment | Start here | Additional inputs |
|---|---|---|
| exp1: design | [exp1/README.md](exp1/README.md) | Dataset adapters, model paths, GPU environment |
| exp2: runtime | [exp2/README.md](exp2/README.md) | Frozen planner/worker, scoring models, Editor model, task JSONL |
| exp3: attribution | [exp3/README.md](exp3/README.md) | Benchmark traces, local scorer/NLI/localizer checkpoints |

Each experiment manages its own dependencies. Avoid installing all GPU stacks into one environment by default. The separately deployed [MAS-Gate Demo](https://mas-gate-demo.hehailin.life/) provides an API-based inference example.

## Current runtime protocol

The Editor checks a node's role, assigned task, constraints and completed output. It receives the current graph and observed history only. Ground-truth answers, quantification scores and future outputs are withheld. The action schema is `{action, target, payload}`; `payload` is validated by action type.

- `KEEP`: continue; no payload.
- `RETRY_CURRENT`: repeat the saved prompt/context if replay is safe.
- `BACKTRACK_PARENT`: replay a direct parent and affected completed descendants.
- `ANNOTATE_CURRENT`: retain raw output and append a bounded structured diagnosis quoting an observed span.
- `GUIDE_CHILD`: give a pending direct child one structured check.
- `INSERT_SUBTASK`: add a VERIFY, BRIDGE or SUPPLEMENT milestone and preserve the original upstream handoff.

One valid non-Keep intervention and at most four extra node executions are allowed. `exp2/gate_main/training.py` now implements the manuscript's additive child-strength formula. Post-hoc chain scoring remains in exp3.

## Validation

```bash
(cd exp2 && python3 -m unittest discover -s tests -v)
(cd exp3 && python3 -m unittest discover -s tests -v)
```

Public upstream attribution is retained, as described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## exp1 source limitation

The supplied `exp1/verl/mas_gate_rl/reward.py` loads `_reward_impl.cpython-312.pyc`, which was not included in the source material. Training and evaluation paths that import this adapter cannot run as supplied. Its Python implementation must be recovered before claiming full exp1 reproduction support.

The node-scoring source is present in `exp1/quantification/`, including `QuantificationEngine` and the unified trace-scoring functions. It computes the scoring signals; it does not replace the missing workflow execution and trainer adapter. exp2 and exp3 use separate entry points.

## License

Project-owned code is licensed under **Apache-2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Existing third-party licenses remain in place. External data, model weights and the paper retain their own terms.
