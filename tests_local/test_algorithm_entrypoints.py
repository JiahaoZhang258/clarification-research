import unittest
from pathlib import Path

from clarify.algorithms.distilled_clarifier import DISTILLED_FINAL_PROMPT, DistilledClarifier
from clarify.algorithms.original_cot_clarifier import OriginalCoTClarifier
from generate_responses import _load_clarification_algorithm
from test_contrast_questions import ScriptedEnv

ROOT = Path(__file__).resolve().parents[1]


class AlgorithmEntrypointTests(unittest.TestCase):
    def test_sdk_can_load_exactly_one_class_from_each_new_entrypoint(self):
        for name, expected in (
            ("original_cot_clarifier.py", "OriginalCoTClarifier"),
            ("distilled_clarifier.py", "DistilledClarifier"),
        ):
            cls = _load_clarification_algorithm(str(ROOT / "clarify/algorithms" / name))
            self.assertEqual(cls.__name__, expected)

    def test_embedded_distilled_prompt_matches_evaluated_frozen_asset(self):
        frozen = (ROOT / "reproducibility/frozen_prompts/distilled_v1.txt").read_text().strip()
        self.assertEqual(DISTILLED_FINAL_PROMPT, frozen)

    def test_adapters_preserve_draft_question_and_answer_integration(self):
        original, distilled = ScriptedEnv(), ScriptedEnv()
        problem = {"prompt": "Transform x.", "entry_point": "f"}
        OriginalCoTClarifier().run(original, problem)
        DistilledClarifier().run(distilled, problem)
        self.assertEqual(original.prompts[:2], distilled.prompts[:2])
        self.assertEqual(original.questions, distilled.questions)
        self.assertEqual(len(distilled.prompts), 3)
        self.assertIn("User answer: Return the square.", distilled.prompts[-1])
        self.assertNotEqual(original.prompts[-1], distilled.prompts[-1])


if __name__ == "__main__":
    unittest.main()
