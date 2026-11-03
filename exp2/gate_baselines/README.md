# Runtime comparison methods

`output_repair.py` contains Self-Refine and AgentDropoutV2-style references; `graph_methods.py` contains dynamic-team and learned-pruning references; `strong_llm_editor.py` provides an external model Editor. Shared interfaces are in `protocol.py`.

These compact implementations require a caller-provided model client and task-specific evaluation harness. They are not automatically run by the main Editor CLI. The package is included in the exp2 installation and no longer excluded by Git.
