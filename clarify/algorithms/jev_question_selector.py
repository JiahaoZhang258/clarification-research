"""JEV-assisted question selection layered on the pinned first-place algorithm.

The base :class:`ZarutskiyClarifier` asks a single ``targeted`` question. This
subclass instead generates up to ``max_questions`` candidate questions with the
multi-question prompt and picks one, either deterministically (``selection="first"``,
the model's own top-ranked candidate) or with a decision model (``selection="jev"``,
via ``env.decide``).

Only the question-generation/selection step changes; the draft, ``ask_human``,
final generation, sanitizer and candidate selection are all inherited unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from clarify.algorithms.zarutskiy_clarifier import ZarutskiyClarifier, plain_question
from clarify.env import ClarificationEnvironment, TooManyQuestionException


class JevQuestionSelector(ZarutskiyClarifier):
    DEFAULT_CONFIG = {
        **ZarutskiyClarifier.DEFAULT_CONFIG,
        # How to choose among the candidate questions from the multi-question prompt:
        #   "first" -> the model's own top-ranked candidate (no decision model)
        #   "jev"   -> env.decide with a "choice" question (requires a decision model)
        "selection": "jev",
    }

    def _candidate_questions(
        self, env: ClarificationEnvironment, problem: dict[str, Any], candidate: str
    ) -> list[str]:
        """Return the individual questions the multi-question prompt produced."""
        joined = self.question_multi(env, problem, candidate)
        if not joined:
            return []
        questions: list[str] = []
        for line in joined.splitlines():
            match = re.match(r"^\d+[.)]\s*(.+)$", line.strip())
            if match:
                question = match.group(1).strip()
                if question:
                    questions.append(question)
        return questions[: max(1, int(self.cfg("max_questions")))]

    def _select(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        candidate: str,
        questions: list[str],
    ) -> str:
        """Pick one question and record the decision on ``self.last_selection``."""
        probabilities = None
        confidence = None
        if len(questions) == 1 or self.cfg("selection") == "first":
            index = 0
        else:
            criteria = {f"q{i + 1}": q for i, q in enumerate(questions)}
            context = (
                "Problem:\n"
                + problem["prompt"]
                + "\n\nDraft implementation:\n```python\n"
                + candidate
                + "\n```"
            )
            try:
                answers = env.decide(
                    questions={
                        "pick": {
                            "type": "choice",
                            "instructions": (
                                "Each option is one clarifying question about the problem. "
                                "Choose the single question whose answer would most reduce the "
                                "uncertainty about the correct behavior and most likely prevent a "
                                "wrong implementation. Prefer questions about required behavior "
                                "(a value, a rule, a return type, an edge-case convention), never "
                                "implementation details or test cases."
                            ),
                            "criteria": criteria,
                        }
                    },
                    context=context,
                )
                pick = answers["pick"]["choice"]
                probabilities = answers["pick"].get("probabilities")
                confidence = answers["pick"].get("confidence")
                index = next(
                    (i for i in range(len(questions)) if f"q{i + 1}" == pick), 0
                )
            except Exception:
                # Decision model unavailable or failed; fall back to the model's own choice.
                index = 0
        self.last_selection = {
            "questions": questions,
            "chosen_index": index,
            "chosen": questions[index],
            "probabilities": probabilities,
            "confidence": confidence,
        }
        return questions[index]

    def refine(
        self, env: ClarificationEnvironment, problem: dict[str, Any], candidate: str
    ) -> str:
        question: str | None = None
        answer: str | None = None
        if env.can_ask():
            questions = self._candidate_questions(env, problem, candidate)
            if questions:
                question = self._select(env, problem, candidate, questions)
            else:
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
        return self.select(env, problem, codes, question, answer)
