"""CPU-only contracts for model adapters and answer-target construction."""

from types import SimpleNamespace
import unittest

from mas_gate.models import (
    ANSWER_SYSTEM_PROMPT, AnswerScorer, Localizer, NLI, engine_kwargs, prepare_answer_tokens,
)


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, add_answer_bos=False):
        self.calls = []
        self.add_answer_bos = add_answer_bos

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return "header:" + messages[1]["content"] + ":assistant"

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        ids = [20, 21] if text == "target" else [10, 11, 12]
        if text == "target" and self.add_answer_bos and kwargs.get("add_special_tokens", True):
            ids = [1] + ids
        return SimpleNamespace(input_ids=[SimpleNamespace(tolist=lambda: ids)])


class ModelContracts(unittest.TestCase):
    def test_task_readout_and_separate_answer_tokens(self):
        tokenizer = FakeTokenizer()
        context, answer = prepare_answer_tokens(tokenizer, "question\nstep", "target", 6)
        self.assertEqual(context, [10, 11, 12])
        self.assertEqual(answer, [20, 21])
        self.assertEqual(tokenizer.messages, [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": "question\nstep"},
        ])
        self.assertEqual(tokenizer.template_kwargs, {
            "tokenize": False, "add_generation_prompt": True, "enable_thinking": False,
        })
        self.assertEqual([text for text, _ in tokenizer.calls], [
            "header:question\nstep:assistant", "target", "target",
        ])
        self.assertFalse(tokenizer.calls[-1][1]["add_special_tokens"])

    def test_answer_special_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError, "special tokens"):
            prepare_answer_tokens(FakeTokenizer(add_answer_bos=True), "question", "target")

    def test_j_overlength_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            prepare_answer_tokens(FakeTokenizer(), "question", "target", 5)

    def test_empty_answer_and_context_rejected(self):
        for context, answer in (("", "target"), ("question", ""), ("question", "  ")):
            with self.subTest(context=context, answer=answer):
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    prepare_answer_tokens(FakeTokenizer(), context, answer)

    def test_nonfinite_memory_fraction_rejected(self):
        for fraction in (float("nan"), float("inf"), 0.0):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    engine_kwargs("/local/checkpoint", "qwen30", gpu_memory_utilization=fraction)

    def test_five_localizer_engine_profiles(self):
        shared = {
            "model": "/local/checkpoint", "runner": "generate", "trust_remote_code": True,
            "tensor_parallel_size": 1, "gpu_memory_utilization": 0.9,
            "enforce_eager": True, "disable_custom_all_reduce": True,
            "max_model_len": 81920, "max_num_seqs": 1, "max_num_batched_tokens": 2048,
        }
        multimodal_off = {
            "language_model_only": True,
            "limit_mm_per_prompt": {"image": 0, "video": 0}, "skip_mm_profiling": True,
        }
        extras = {
            "qwen30": multimodal_off,
            "qwen35_4b": {**multimodal_off, "gdn_prefill_backend": "triton"},
            "qwen36_27b": {**multimodal_off, "gdn_prefill_backend": "triton"},
            "mistral24": {
                **multimodal_off, "tokenizer_mode": "mistral", "config_format": "hf",
                "load_format": "hf", "hf_overrides": {"tie_word_embeddings": False},
            },
            "llama8": {},
        }
        for profile, extra in extras.items():
            with self.subTest(profile=profile):
                self.assertEqual(
                    engine_kwargs("/local/checkpoint", profile, gpu_memory_utilization=0.9),
                    {**shared, **extra},
                )

    def test_localizer_refuses_long_prompt_before_generation(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return list(range(10))

        class Engine:
            def chat(self, **kwargs):
                raise AssertionError("overlength requests must not reach the engine")

        localizer = Localizer.__new__(Localizer)
        localizer._llm = Engine()
        localizer._tokenizer = Tokenizer()
        localizer.max_model_len = 265
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            localizer.generate([{"role": "user", "content": "question"}])

    def test_teacher_forcing_scores_only_answer_positions(self):
        try:
            import torch  # The adapter preserves the recorded float32 reduction.
        except ImportError:
            self.skipTest("PyTorch is optional for the lightweight CPU test suite")

        class Engine:
            def generate(self, **kwargs):
                self.request = kwargs
                return [SimpleNamespace(prompt_logprobs=[
                    None, {11: -50.0}, {12: -50.0},
                    {20: SimpleNamespace(logprob=-0.25)},
                    {21: SimpleNamespace(logprob=-0.75)},
                ])]

        scorer = AnswerScorer.__new__(AnswerScorer)
        scorer.tokenizer = FakeTokenizer()
        scorer.max_model_len = 6
        scorer._llm = Engine()
        scorer._sampling = "scoring-params"
        scorer._tokens_prompt = dict
        self.assertEqual(scorer("question", "target"), 0.5)
        self.assertEqual(
            scorer._llm.request["prompts"], [{"prompt_token_ids": [10, 11, 12, 20, 21]}]
        )

    def test_nli_pair_batching_and_truncation_contract(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional for the lightweight CPU test suite")

        class Batch(dict):
            def to(self, device):
                self.device = device
                return self

        class Tokenizer:
            def __init__(self):
                self.calls = []

            def __call__(self, premises, hypotheses, **kwargs):
                self.calls.append((premises, hypotheses, kwargs))
                return Batch(rows=len(premises))

        class Model:
            def __call__(self, rows):
                return SimpleNamespace(logits=torch.tensor([[0.0, 0.0]] * rows))

        nli = NLI.__new__(NLI)
        nli.device, nli.max_length, nli.batch_size = "cpu", 8192, 32
        nli._tokenizer, nli._model, nli._entailment_id = Tokenizer(), Model(), 0
        pairs = [(f"premise-{i}", f"hypothesis-{i}") for i in range(33)]
        self.assertEqual(nli.probabilities(pairs), [0.5] * 33)
        self.assertEqual([len(call[0]) for call in nli._tokenizer.calls], [32, 1])
        for premises, hypotheses, kwargs in nli._tokenizer.calls:
            self.assertEqual(kwargs, {
                "return_tensors": "pt", "truncation": True,
                "max_length": 8192, "padding": True,
            })
            self.assertTrue(premises[0].startswith("premise-"))
            self.assertTrue(hypotheses[0].startswith("hypothesis-"))
        self.assertEqual(nli.probabilities([]), [])


if __name__ == "__main__":
    unittest.main()
