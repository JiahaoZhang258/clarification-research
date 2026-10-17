"""Training prototype: one behavioral question with a distinguishing example.

The final prompt is frozen to the original CoT experiment. Only the question
prompt differs between FrozenCoTClarifier and ContrastQuestionClarifier.
"""

from __future__ import annotations

import re

from clarify.algorithms.zarutskiy_clarifier import (
    ZarutskiyClarifier,
    after_tag,
    as_code,
    call_llm,
    strip_fences,
)


FROZEN_COT_FINAL_PROMPT = """
Implement `{entry_point}` for the task below.

<problem>
{prompt}
</problem>

You asked the user a clarifying question and got an answer:
Question: {question}
Answer: {answer}

The answer states the user's intent: where it conflicts with the task text or its
examples, follow the answer; everything the answer does not mention follows the task text.

Reason through the problem step by step before writing any code:

1. What the answer settles: in one sentence, state the fact or rule the answer
   establishes, and name any part of the original text it overrides or refines.
2. Complete requirement: restate the full, de-conflicted requirement in one or two
   sentences.
3. Edge cases: list every boundary or special case (empty input, zero, negative,
   ties, duplicates, boundary values) and state the exact correct behavior for each,
   consulting the answer wherever it applies.
4. Implementation: write the code that satisfies the requirement and all edge cases.

Keep the function name exactly `{entry_point}`. Enclose your solution in ```python and ```.
""".strip()


CONTRAST_QUESTION_PROMPT = """
A user asked you to implement {entry_point}:

<problem>
{prompt}
</problem>

Your first draft (a fallible guess, not a specification):
<draft>
{candidate}
</draft>

You may ask exactly ONE short clarification about ONE behavioral requirement.
Choose the unresolved fact whose answer would change the implementation most.

First locate the actual missing rule or contradiction. Check the basic operation,
return meaning/structure, and any explicitly missing parameter or membership rule
before considering peripheral cases. Do not spend the question on a detail that
leaves the main operation unspecified. A fact already stated consistently in the
description is not missing. When two statements or an example conflict, ask which
behavior is intended rather than silently selecting one. Ignore implementation
restrictions and unusual invalid inputs when an ordinary-input rule is unresolved.

Construct two plausible interpretations of that ONE fact and ONE small valid
input for which they give different outputs. Work out both outputs carefully.
Use the example to ask about the general rule, not to request a hidden test.
If a rule is completely omitted (such as what one output field means or which
values belong to a named category), the two options are hypotheses; explicitly
allow the user to supply the intended rule instead. Do not fabricate a complete
table, ask for several independent facts, or ask for code or an algorithm.
If the ambiguity concerns the accepted call format, illustrate the competing
call forms instead of making up an input outside the domain.

Return exactly:
ANALYSIS: <2-4 sentences identifying the unresolved fact, interpretations A and B,
and why resolving this fact changes the draft.>
WITNESS: <one input/call; output under A; different output under B>
QUESTION: <one concise question about that rule, illustrated by the witness.
Do not imply the options are exhaustive if the rule is absent.>

Only the QUESTION is sent to the user. Plain text only, no Markdown or backticks.
Do not bundle questions or ask the user to provide a full specification.
""".strip()


def render(template, values):
    """Substitute once so task text containing placeholders is preserved."""
    return re.sub(r"\{(entry_point|prompt|candidate|question|answer)\}",
                  lambda match: values[match[1]], template)


class FrozenCoTClarifier(ZarutskiyClarifier):
    def final_code(self, env, problem, question, answer):
        content = render(FROZEN_COT_FINAL_PROMPT, {
            **problem, "question": question or "", "answer": answer or "",
        })
        code, raw = self.generate_code(env, content)
        return [code or as_code(raw)]


class ContrastQuestionClarifier(FrozenCoTClarifier):
    def question_targeted(self, env, problem, candidate):
        response = call_llm(env, render(CONTRAST_QUESTION_PROMPT, {
            **problem, "candidate": candidate,
        }))
        self.question_analysis = response
        question = strip_fences(after_tag(response, "QUESTION:"))
        if not question:
            # Use the same deterministic malformed-response fallback as the baseline.
            question = response.strip().splitlines()[-1] if response.strip() else ""
        return question or None
