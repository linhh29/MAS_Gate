# Experiment 3 · Post-hoc attribution

This experiment scores complete trajectories and compares direct versus risk-guided localization. It uses a chronological-chain approximation, reference-answer likelihood changes, frozen NLI uptake and downstream reach. Reference task answers are available to the localizers; attribution gold labels are excluded from their prompts.

```bash
python3 -m mas_gate --help
python3 -m unittest discover -s tests -v
```

The CLI lists preparation, quantification, localization and evaluation commands. `requirements.txt` records the supplied GPU stack. Heavy dependencies are imported only when constructing model wrappers. Provide existing compatible local model directories and benchmark files; no weights are bundled or automatically downloaded by these wrappers.

Who&When uses zero-based steps. AEGIS uses maximum eligible-step risk per agent, with User/System turns excluded as targets. Candidates prioritize inspection while the complete trajectory remains visible.
