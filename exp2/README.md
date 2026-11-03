# MAS-Gate Runtime Editor: configurable research code

This folder contains the main Runtime Editor method chain. A frozen Orchestra
provides one planned DAG per task. A frozen worker executes that DAG once to
create a no-edit reference. The trainable Editor observes each completed node,
chooses `KEEP` or one of five interventions, and receives a paired reward from
the edited and no-edit continuations. The actor is updated with turn-local
group-relative policy optimization (GRPO). Reference answers and MAS-Gate
scores are used by the trainer, never in the Editor prompt.

The implementation is intentionally compact. It can run on LLM-only workflows
after model paths, service endpoints, and task data are supplied. Its
single-process LoRA trainer expresses the Editor decision and reward logic;
larger runs may replace this training backend with a distributed one. Stateful
environments need a task-specific worker with safe snapshot/replay semantics.

## Setup

From this directory, create a Python environment and install the package:

```bash
python -m pip install -e .
cp config.example.json config.local.json
```

Edit `config.local.json` to point to the trainable Editor base model, frozen
scoring LLM, frozen NLI model, and OpenAI-compatible frozen Orchestra and
sub-agent services. The service keys are read from `PLANNER_API_KEY` and
`WORKER_API_KEY`; no key belongs in
the JSON file. `config.local.json`, `data/`, and `runs/` are ignored by Git.

The default `HFQuantifier` loads the scoring LLM and NLI model locally. These
models need their own memory, separate from the actor. For large models or a
phase-separated cluster, provide `quantifier.adapter` as described below.

## Training input

Start with a raw training JSONL containing `id`, `task`, and `answer`. The
planning command adds a frozen `graph` and writes the file named by
`data_file`; neither the planner nor the Editor receives `answer`.
`data_file` has one task per line with:

| Field | Meaning |
| --- | --- |
| `id` | Stable, unique task identifier. |
| `task` | Global task shown to the MAS and Editor. |
| `answer` | Training reference answer; withheld from Editor observations. |
| `graph.nodes` | Frozen Orchestra DAG nodes. |

Each node has `id`, `role`, `task`, `parents` (node ID list), `constraints`
(string list), `is_final` (exactly one true), and `replay_safe` (boolean).
Parents must precede their children in the DAG. The standard worker supports
text-only nodes. Set `replay_safe=false` for nodes with tool or environment
side effects unless a custom worker can restore the prior state. No training
or test examples are included in this release.

## Run order

1. Start the frozen Orchestra service at `planner.base_url` and the frozen
   sub-agent service at `worker.base_url`. Freeze a workflow for each raw
   training task:

   ```bash
   python -m gate_main.plan --config config.local.json \
     --input data/raw_train.jsonl --output data/train.jsonl
   ```

   The generic planner is for text-only workflows. To preserve a task's own
   Orchestra design, set `planner.adapter` to `your_module:PlannerClass`;
   its constructor receives `config=planner_options` and its
   `plan(task)` method returns a validated `Graph`. Do not change the
   Orchestra or sub-agents while training the Editor.
2. Build and score one no-intervention reference for every row:

   ```bash
   python -m gate_main.prepare --config config.local.json
   ```

   This saves the full DAG, node outputs, original prompts, correctness, and
   mean node impact. Existing references are kept; training checks a hash of
   each input row before use.
3. Train the Editor:

   ```bash
   python -m gate_main.train --config config.local.json
   ```

   The `--resume-from runs/editor/step-N` option restores the LoRA adapter,
   optimizer, and recorded RNG state. Restarting a stateful external service
   may still change its future responses; an exact resume requires a
   deterministic service or a cached continuation backend.
4. Apply a saved adapter without a reference answer or quantifier. Supply
   `data/predict.jsonl` with `id`, `task`, and `graph` fields (no `answer`),
   then run:

   ```bash
   python -m gate_main.predict --config config.local.json \
     --checkpoint runs/editor/step-1 \
     --data data/predict.jsonl --output runs/predictions.jsonl
   ```

   This executes each DAG as nodes complete. The Editor sees the planned graph
   and the observed output prefix, then chooses an action at each gate. It
   stops making decisions after its first valid non-`KEEP` intervention.
   Invalid model output falls back to `KEEP` and is marked in the decision log.
   The output file contains predictions and action decisions, not benchmark
   metrics. Use the benchmark's scorer separately if needed.

For a small configuration check, set `batch_tasks=1`, `rollouts_per_task=4`,
`steps=1`, and `checkpoint_steps=[1]`. Then scale the task batch, rollout
group, and total steps to match the available hardware and data.

## What happens in one training step

For each task, the Editor samples a full task-level episode. A `KEEP` action
has local reward zero and reveals the next executed node; the first valid
non-`KEEP` action applies one edit and ends Editor decisions. Invalid JSON or
an illegal action receives the configured fixed penalty. For a valid edit,
only the affected completed nodes and the remaining suffix are executed.
The paired return is

```text
r = c(edited) - c(no_edit)
    + beta_q * (Q(edited) - Q(no_edit))
    - lambda_cost * action_cost
```

Here `Q` is the mean completed-node impact, with
`U_i=tanh((J(C_i+,y)-J(C_i-,y))/abs(J_0))`,
`K_i=sum_{j in children(i)} R_ij(1+G_j)`, and `I_i=U_i(1+K_i)`.
The frozen scoring LLM supplies teacher-forced `J`; the frozen NLI model
supplies entailment probability `R`. Rewards are normalized across sampled
episodes for the same task and gate ordinal. Each advantage trains only that
gate's Editor action tokens. Earlier `KEEP` actions do not inherit a later
intervention reward.

The six actions and their legality checks are in `gate_main/editor.py`; their
effects and replay logic are in `gate_main/runtime.py`. The actual model input
is assembled by `gate_main/prompts.py`. The training loop is
`gate_main/train.py`.

## Experiment adapters

The default `exact_answer` scorer is only appropriate for tasks with a clean
final answer. Set `correctness_adapter` to `your_module:function`; it must
accept `(prediction, answer)` and return a number in `[0,1]`. Use the paper's
benchmark-specific success criterion when reproducing a reported result.

For tool-using or interactive tasks, set `worker.adapter` to
`your_module:WorkerClass`. Its constructor receives `config=worker_options`;
its `run(node, task, upstream, guidance="", fixed_prompt=None)` method returns
`(output, prompt)`. If it marks nodes replay-safe, it should also implement
`prepare_replay(gate_state, action)` to restore the environment state before
`RETRY_CURRENT` or `BACKTRACK_PARENT`.

For a remote or specialized scoring service, set `quantifier.adapter` to
`your_module:QuantifierClass`. Its constructor receives
`config=quantifier_options`; its `score(task, answer, graph, outputs)` method
returns mean node impact `Q`. Keep the same quantifier for reference and edited
traces. Never include its outputs in Editor prompts.

The example JSON uses a one-task batch for accessibility. The formal
HotpotQA/DeepScaleR protocol used 10,000 raw tasks, 256 tasks per step,
16 rollouts per task, 39 steps, and checkpoints at steps 4, 8, ..., 36, 39;
the 16-row tail was dropped. ALFWorld used 4,032 tasks and 15 steps. These
settings need the benchmark assets, the original frozen workflow generator,
and substantially more compute than the example configuration.


## Release layout

Both `gate_main` and `gate_baselines` are included by the package configuration. Baseline interfaces have their own README. Runtime unit tests live in `tests/`. The API-based web interface is distributed separately as MAS-Gate Demo. Run `python3 -m unittest discover -s tests -v` from this directory. Project-owned code is Apache-2.0; see `LICENSE`.
