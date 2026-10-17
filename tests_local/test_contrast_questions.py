import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import litellm

from clarify.algorithms.contrast_question_clarifier import (
    FROZEN_COT_FINAL_PROMPT,
    ContrastQuestionClarifier,
    FrozenCoTClarifier,
    render,
)
from experiments import compare_contrast_questions as E


class ScriptedEnv:
    def __init__(self):
        self.prompts = []
        self.questions = []

    def llm(self, messages):
        text = messages if isinstance(messages, str) else messages[-1]['content']
        self.prompts.append(text)
        if len(self.prompts) == 1:
            return '```python\ndef f(x): return x\n```'
        if len(self.prompts) == 2:
            return 'ANALYSIS: one rule\nWITNESS: f(2): 2 versus 4\nQUESTION: Should f(2) be 2 or 4?'
        return '```python\ndef f(x): return x * x\n```'

    def can_ask(self):
        return True

    def ask_human(self, question):
        self.questions.append(question)
        return 'Return the square.'


class ContrastQuestionTests(unittest.TestCase):
    def test_only_question_step_changes_and_answer_reaches_shared_final(self):
        envs = []
        for cls in (FrozenCoTClarifier, ContrastQuestionClarifier):
            env = ScriptedEnv()
            code = cls().run(env, {'prompt': 'Transform x.', 'entry_point': 'f'})
            self.assertIn('return x * x', code)
            self.assertEqual(len(env.prompts), 3)
            self.assertEqual(env.questions, ['Should f(2) be 2 or 4?'])
            self.assertIn('Answer: Return the square.', env.prompts[-1])
            envs.append(env)
        self.assertEqual(envs[0].prompts[0], envs[1].prompts[0])
        self.assertEqual(envs[0].prompts[-1], envs[1].prompts[-1])
        self.assertNotEqual(envs[0].prompts[1], envs[1].prompts[1])

    def test_final_matches_frozen_original_and_substitution_is_single_pass(self):
        frozen = (E.R.ROOT / 'reproducibility/frozen_prompts/original_cot.txt').read_text().strip()
        self.assertEqual(FROZEN_COT_FINAL_PROMPT, frozen)
        self.assertEqual(render('{prompt} {question}', {'prompt': '{question}', 'question': 'Q'}), '{question} Q')

    def test_resume_replays_paid_response_and_separates_experimental_arms(self):
        calls = []

        def paid(gateway, *args, **kwargs):
            calls.append(kwargs)
            response = litellm.ModelResponse(choices=[{'message': {'role': 'assistant', 'content': 'saved answer'}}],
                                            usage={'prompt_tokens': 10, 'completion_tokens': 3, 'total_tokens': 13})
            gateway.response_costs[id(response)] = 0.0001
            return response

        with tempfile.TemporaryDirectory() as tmp, patch.object(E, 'OUT', Path(tmp)), patch.object(E.R.ConcurrentGateway, 'complete', paid):
            gateway = E.CachedGateway('unused', {}, None)
            gateway.context = {'round': 1, 'arm': 'baseline', 'instance_id': 'one', 'phase': 'assistant'}
            request = {'model': E.MODEL, 'messages': [{'role': 'user', 'content': 'same request'}], 'temperature': 0.7}
            first = gateway.complete(**request)
            resumed = E.CachedGateway('unused', {}, None)
            resumed.context = gateway.context.copy()
            replay = resumed.complete(**request)
            self.assertEqual(len(calls), 1)
            self.assertEqual(first.choices[0].message.content, replay.choices[0].message.content)
            self.assertEqual(resumed.response_costs[id(replay)], 0.0001)
            resumed.context = {**resumed.context, 'arm': 'contrast'}
            resumed.complete(**request)
            self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
