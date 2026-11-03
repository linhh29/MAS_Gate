"""Construct replaceable worker and scoring components from JSON config."""

from __future__ import annotations

from .backends import OpenAIWorker, load_symbol
from .quality import HFQuantifier, exact_answer


def worker_from_config(config: dict):
    options = config["worker"]
    if options.get("adapter"):
        return load_symbol(options["adapter"])(config=options)
    return OpenAIWorker(**{key: value for key, value in options.items() if key != "adapter"})


def quantifier_from_config(config: dict):
    options = config["quantifier"]
    if options.get("adapter"):
        return load_symbol(options["adapter"])(config=options)
    return HFQuantifier(**{key: value for key, value in options.items() if key != "adapter"})


def correctness_from_config(config: dict):
    spec = config.get("correctness_adapter")
    return load_symbol(spec) if spec else exact_answer
