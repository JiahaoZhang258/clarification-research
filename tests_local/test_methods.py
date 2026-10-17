import json
import unittest

from clarify.algorithms.assumption_audit import AssumptionAudit
from clarify.algorithms.example_confirmation import ExampleConfirmation

CODE = "```python\ndef unique(xs):\n    return list(dict.fromkeys(xs))\n```"
PROBLEM = {"prompt": "Remove duplicate items.", "entry_point": "unique", "hidden": "SECRET"}
PROPOSAL = {
    "ambiguity": "duplicate policy",
    "interpretation_a": "keep one",
    "interpretation_b": "remove repeated values",
    "input": "[1,1,2]",
    "output_a": "[1,2]",
    "output_b": "[2]",
    "question": "Should a repeated value be kept once, e.g. [1,1,2] becomes [1,2]?",
}


class FakeEnvironment:
    def __init__(self, responses, turns=1):
        self.responses = iter(responses)
        self.turns = turns
        self.messages = []
        self.questions = []

    def can_ask(self):
        return len(self.questions) < self.turns

    def llm(self, messages):
        self.messages.append(messages)
        return next(self.responses)

    def ask_human(self, question):
        if not self.can_ask():
            raise AssertionError("Exceeded clarification budget")
        self.questions.append(question)
        return "Keep each value once, in its original order."


class MethodTests(unittest.TestCase):
    def test_example_confirms_one_rule_and_uses_answer(self):
        question = PROPOSAL["question"]
        env = FakeEnvironment(
            [
                json.dumps(PROPOSAL),
                json.dumps({"ask": True, "question": question}),
                CODE,
            ]
        )
        self.assertEqual(ExampleConfirmation().run(env, PROBLEM), CODE)
        self.assertEqual(env.questions, [question])
        final = json.dumps(env.messages[-1])
        self.assertIn("Keep each value once", final)
        self.assertNotIn("SECRET", json.dumps(env.messages))
        self.assertNotIn("interpretation_b", final)

    def test_example_does_not_treat_string_false_as_permission(self):
        env = FakeEnvironment(
            [
                json.dumps(PROPOSAL),
                '{"ask":"false","question":"Unnecessary?"}',
                CODE,
            ]
        )
        ExampleConfirmation().run(env, PROBLEM)
        self.assertEqual(env.questions, [])

    def test_example_rejects_non_distinguishing_outputs(self):
        proposal = {**PROPOSAL, "output_b": PROPOSAL["output_a"]}
        env = FakeEnvironment([json.dumps(proposal), CODE])
        ExampleConfirmation().run(env, PROBLEM)
        self.assertEqual(env.questions, [])
        self.assertEqual(len(env.messages), 2)

    def test_malformed_proposal_falls_back_to_implementation(self):
        env = FakeEnvironment(["not JSON", CODE])
        self.assertEqual(ExampleConfirmation().run(env, PROBLEM), CODE)
        self.assertEqual(env.questions, [])

    def test_audit_requires_real_code_evidence(self):
        audit = {
            "ask": True,
            "assumption": "ignore case",
            "code_evidence": "xs.lower()",
            "alternative": "case-sensitive",
            "question": "Ignore case?",
        }
        env = FakeEnvironment([CODE, json.dumps(audit), CODE])
        AssumptionAudit().run(env, PROBLEM)
        self.assertEqual(env.questions, [])

    def test_audit_asks_once_and_uses_answer(self):
        audit = {
            "ask": True,
            "assumption": "keep one",
            "code_evidence": "dict.fromkeys(xs)",
            "alternative": "remove all repeated values",
            "question": "Keep each value once?",
        }
        env = FakeEnvironment([CODE, json.dumps(audit), CODE])
        self.assertEqual(AssumptionAudit().run(env, PROBLEM), CODE)
        self.assertEqual(env.questions, [audit["question"]])
        self.assertIn("Keep each value once", json.dumps(env.messages[-1]))
        self.assertNotIn("SECRET", json.dumps(env.messages))

    def test_both_obey_zero_question_budget(self):
        for cls in (ExampleConfirmation, AssumptionAudit):
            with self.subTest(cls=cls):
                env = FakeEnvironment([CODE], turns=0)
                self.assertEqual(cls().run(env, PROBLEM), CODE)
                self.assertEqual(env.questions, [])
                self.assertEqual(len(env.messages), 1)


if __name__ == "__main__":
    unittest.main()
