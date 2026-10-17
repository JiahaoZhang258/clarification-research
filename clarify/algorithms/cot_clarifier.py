"""Chain-of-thought final generation layered on the pinned first-place algorithm.

Only the final code-generation prompt changes: instead of asking for code
directly, it asks the model to first restate the de-conflicted requirement and
enumerate edge cases, then write the implementation. The draft and question
steps are inherited unchanged.
"""

from __future__ import annotations

from typing import Any

from clarify.algorithms.zarutskiy_clarifier import ZarutskiyClarifier, as_code
from clarify.env import ClarificationEnvironment

COT_FINAL_PROMPT = """
Implement `{entry_point}` for the task below. First study three worked examples of
merging a clarification answer into a correct implementation, then do the same.

Example 1 - the answer overrides the original wording:
<problem>Write a function next_mirror(n) that returns the next mirror number smaller than n.</problem>
Question: I will assume the function must return a mirror number smaller than n. Is that right, or should it be larger?
Answer: It must return the smallest mirror number strictly larger than n.
Reasoning:
1. What the answer settles: the result is larger than n, overriding "smaller" in the text.
2. Complete requirement: return the smallest mirror number strictly greater than n.
3. Edge cases: if n is already a mirror number, return the next larger one; single digits advance to the next mirror number.
4. Implementation:
```python
def next_mirror(n):
    while True:
        n += 1
        s = str(n)
        if s == s[::-1]:
            return n
```

Example 2 - the answer resolves an edge-case ambiguity:
<problem>Write a function long_words(words, n) that returns the words in a sentence longer than n.</problem>
Question: I will assume "longer than" means strictly greater, and words equal to n are excluded. Is that right?
Answer: Yes, strictly greater; exclude words exactly equal to n.
Reasoning:
1. What the answer settles: only words with length strictly greater than n count.
2. Complete requirement: return words whose length is greater than n.
3. Edge cases: the input is one string of space-separated words; a word equal to n is excluded.
4. Implementation:
```python
def long_words(words, n):
    return [w for w in words.split() if len(w) > n]
```

Example 3 - the answer settles the return type:
<problem>Write a function smallest_even(nums) that finds the smallest even number in a list.</problem>
Question: I will assume the function returns the number itself, and None if there is no even number. Is that right?
Answer: Yes, return the number itself, or None when there is no even number.
Reasoning:
1. What the answer settles: return the value itself, and None for "no even number".
2. Complete requirement: return the smallest even integer, or None if none exists.
3. Edge cases: an empty or all-odd list returns None; negative numbers are allowed.
4. Implementation:
```python
def smallest_even(nums):
    evens = [x for x in nums if x % 2 == 0]
    return min(evens) if evens else None
```

Now do the same for the task below.

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


class CoTClarifier(ZarutskiyClarifier):
    def final_code(
        self,
        env: ClarificationEnvironment,
        problem: dict[str, Any],
        question: str | None,
        answer: str | None,
    ) -> list[str]:
        entry = problem["entry_point"]
        n = max(1, int(self.cfg("num_final")))
        content = (
            COT_FINAL_PROMPT.replace("{prompt}", problem["prompt"])
            .replace("{entry_point}", entry)
            .replace("{question}", question or "")
            .replace("{answer}", answer or "")
        )
        codes = []
        last_raw = ""
        for _ in range(n):
            code, raw = self.generate_code(env, content)
            last_raw = raw
            if code:
                codes.append(code)
        return codes or [as_code(last_raw)]
