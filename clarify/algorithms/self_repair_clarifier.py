"""Self-test-and-repair layered on the pinned first-place algorithm.

After the usual targeted clarification and final generation, this subclass writes
a handful of assert tests from the clarified specification, runs the code on them
via the inherited harness (``env.exec_code``), and regenerates with the failing
asserts as feedback. A repair is kept only when the regenerated code passes at
least as many of those self-written asserts as the code it replaces.

Only the post-generation verify/repair step is added; the draft, question and
final generation are inherited unchanged.
"""

from __future__ import annotations

from typing import Any

from clarify.algorithms.zarutskiy_clarifier import (
    FINAL_PROMPT,
    TESTS_PROMPT,
    ZarutskiyClarifier,
    call_llm,
    extract_code,
    plain_question,
)
from clarify.env import ClarificationEnvironment, TooManyQuestionException


class SelfRepairClarifier(ZarutskiyClarifier):
    DEFAULT_CONFIG = {
        **ZarutskiyClarifier.DEFAULT_CONFIG,
        # How many regeneration rounds the verify/repair step may attempt.
        "repair_attempts": 2,
    }

    def _clarification_block(self, question: str | None, answer: str | None) -> str:
        if not question:
            return ""
        return f"\nClarification from the user:\nQuestion: {question}\nAnswer: {answer}\n"

    def _spec_tests(
        self, env: ClarificationEnvironment, problem: dict[str, Any], question, answer
    ) -> list[str]:
        response = call_llm(
            env,
            TESTS_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{clarification}", self._clarification_block(question, answer))
            .replace("{num_tests}", str(self.cfg("num_tests"))),
        )
        block = extract_code(response) or ""
        return [line.strip() for line in block.splitlines() if line.strip().startswith("assert ")]

    def _run_tests(
        self, env: ClarificationEnvironment, code: str, entry: str, asserts: list[str]
    ) -> tuple[list | None, list[str]]:
        result = self.run_harness(env, [code], entry, "asserts", asserts)
        rows = result.get("rows", [])
        if not rows or rows[0] is None:
            return None, []
        row = rows[0]
        failing = [stmt for i, stmt in enumerate(asserts) if i >= len(row) or row[i] != 1]
        return row, failing

    def _regenerate(
        self, env, problem, question, answer, failing: list[str]
    ) -> str | None:
        content = (
            FINAL_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", problem["entry_point"])
            .replace("{question}", question or "")
            .replace("{answer}", answer or "")
            + "\n\nYour previous implementation failed these self-written tests:\n"
            + "\n".join("Failing test: " + stmt for stmt in failing)
            + "\n\nFix the implementation so it passes them. "
            "Enclose your solution in ```python and ```."
        )
        new_code, _ = self.generate_code(env, content)
        return self.clean(new_code, problem) if new_code else None

    def repair(
        self, env: ClarificationEnvironment, problem: dict[str, Any], question, answer, code: str
    ) -> str:
        entry = problem["entry_point"]
        asserts = self._spec_tests(env, problem, question, answer)
        rounds = 0
        regenerated = False
        for _ in range(max(0, int(self.cfg("repair_attempts")))):
            if not asserts:
                break
            row, failing = self._run_tests(env, code, entry, asserts)
            rounds += 1
            if row is None or not failing:
                break
            regenerated = True
            new_code = self._regenerate(env, problem, question, answer, failing)
            if not new_code:
                break
            new_row, _ = self._run_tests(env, new_code, entry, asserts)
            if new_row is None:
                break
            if sum(1 for r in new_row if r == 1) >= sum(1 for r in row if r == 1):
                code = new_code
        self.last_repair = {
            "self_tests": len(asserts),
            "rounds": rounds,
            "regenerated": regenerated,
        }
        return code

    def refine(
        self, env: ClarificationEnvironment, problem: dict[str, Any], candidate: str
    ) -> str:
        question: str | None = None
        answer: str | None = None
        if env.can_ask():
            question = self.question_targeted(env, problem, candidate)
            if self.cfg("plain_questions"):
                question = plain_question(question)
            if question is not None:
                try:
                    answer = env.ask_human(question)
                except TooManyQuestionException:
                    question, answer = None, None
        if question is None and int(self.cfg("num_final")) <= 1:
            return candidate
        codes = [self.clean(c, problem) for c in self.final_code(env, problem, question, answer)]
        code = self.select(env, problem, codes, question, answer)
        return self.repair(env, problem, question, answer, code)
