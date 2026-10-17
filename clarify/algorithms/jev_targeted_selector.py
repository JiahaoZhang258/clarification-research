"""JEV question selection over multiple targeted-question samples.

Layers on :class:`JevQuestionSelector`: instead of generating candidates with the
multi-question prompt, it samples the inherited ``targeted`` question (assume-framing)
``max_questions`` times and lets the decision model pick one.
"""

from __future__ import annotations

from typing import Any

from clarify.algorithms.jev_question_selector import JevQuestionSelector
from clarify.env import ClarificationEnvironment


class JevTargetedSelector(JevQuestionSelector):
    def _candidate_questions(
        self, env: ClarificationEnvironment, problem: dict[str, Any], candidate: str
    ) -> list[str]:
        limit = max(1, int(self.cfg("max_questions")))
        questions: list[str] = []
        seen: set[str] = set()
        for _ in range(limit):
            question = self.question_targeted(env, problem, candidate)
            if question and question not in seen:
                seen.add(question)
                questions.append(question)
        return questions
