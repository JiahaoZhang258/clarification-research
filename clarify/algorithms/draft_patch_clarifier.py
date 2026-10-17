"""Draft-patching final generation layered on the pinned first-place algorithm.

Instead of discarding the draft and regenerating from scratch (``integrate="final"``),
this subclass keeps the draft and asks the model to revise it minimally against the
clarification answer. The hypothesis: regenerating from scratch sometimes introduces
bugs the draft did not have, while a minimal patch preserves the correct parts.

Only the final-generation step changes; draft, question and sanitizer are inherited.
"""

from __future__ import annotations

from typing import Any

from clarify.algorithms.zarutskiy_clarifier import ZarutskiyClarifier, plain_question
from clarify.env import ClarificationEnvironment, TooManyQuestionException

PATCH_PROMPT = """
Here is your draft implementation for `{entry_point}`:

```python
{draft}
```

You asked the user a clarifying question and got an answer:
Question: {question}
Answer: {answer}

The answer states the user's intent: where it conflicts with the task text or your
draft, follow the answer; everything the answer does not mention keeps the draft as written.

Revise the draft minimally: change only the parts needed to satisfy the answer and the
task, and keep every other part exactly as it is. Keep the function name exactly
`{entry_point}`. Enclose your solution in ```python and ```.
""".strip()


class DraftPatchClarifier(ZarutskiyClarifier):
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
        content = (
            PATCH_PROMPT.replace("{entry_point}", problem["entry_point"])
            .replace("{draft}", candidate)
            .replace("{question}", question or "")
            .replace("{answer}", answer or "")
        )
        code, raw = self.generate_code(env, content)
        return self.clean(code, problem) if code else candidate
