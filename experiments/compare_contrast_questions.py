"""Frozen 100-original-task, two-arm, three-round train-only experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import random
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import litellm

from clarify.algorithms.contrast_question_clarifier import (
    CONTRAST_QUESTION_PROMPT,
    FROZEN_COT_FINAL_PROMPT,
    ContrastQuestionClarifier,
    FrozenCoTClarifier,
)
from clarify.env import ClarificationConfiguration, ClarificationEnvironment, test_context
from clarify.runtime import init_evalplus_evaluator
from evaluate_responses import _normalized_discounted_cumulative_gain, _turn_discounted_sucess
from experiments import reproduce_zarutskiy as R
from experiments.compare_question_selection import load_dev_tasks
from experiments.zenmux import MODEL, RunStopped, fetch_prices, read_key

R.BUDGET = 14.0
OUT = R.ROOT / 'runs/contrast_questions_train_v1'
EXPERIMENT = 'contrast_questions_train_v1'
SOURCE = R.ROOT / 'clarify/algorithms/contrast_question_clarifier.py'
ARMS = {'baseline': FrozenCoTClarifier, 'contrast': ContrastQuestionClarifier}


def prepare():
    tasks = load_dev_tasks()
    excluded = {'HumanEval/14', 'HumanEval/25', 'HumanEval/81'}
    prior = {}
    for directory in ('devset100', 'methods34'):
        path = R.ROOT / 'runs' / directory / 'manifest.json'
        data = json.loads(path.read_text())
        excluded.update(t['task_id'] for t in data['tasks'])
        prior[directory] = R.digest(path.read_bytes())
    grouped = {}
    for key, task in sorted(tasks.items()):
        if task['task_id'] not in excluded:
            grouped.setdefault(task['task_id'], []).append((key, task))
    rng = random.Random(1017)
    selected = []
    # Match the earlier development set's family proportions, but group by original ID.
    for family, count in (('HumanEval', 45), ('Mbpp', 55)):
        ids = sorted(i for i in grouped if i.startswith(family + '/'))
        for task_id in rng.sample(ids, count):
            selected.append(rng.choice(grouped[task_id]))
    rng.shuffle(selected)
    assert len(selected) == len({t['task_id'] for _, t in selected}) == 100
    assert not ({t['task_id'] for _, t in selected} & excluded)
    frozen = (R.ROOT / 'runs/prompt_distillation_v1/cot_prompt.txt').read_text().strip()
    assert FROZEN_COT_FINAL_PROMPT == frozen
    manifest = {
        'experiment': EXPERIMENT, 'split': 'train', 'seed': 1017,
        'count': 100, 'rounds': 3, 'arms': list(ARMS), 'model': MODEL,
        'temperature': 0.7, 'oracle_temperature': 0.0,
        'max_clarification_turns': 1, 'cumulative_budget_usd': R.BUDGET,
        'sampling': '45 HumanEval + 55 Mbpp original IDs; one random variant per ID',
        'excluded_original_task_ids': sorted(excluded), 'prior_manifest_sha256': prior,
        'algorithm_sha256': R.digest(SOURCE.read_bytes()),
        'runner_sha256': R.digest(__import__('pathlib').Path(__file__).read_bytes()),
        'base_source_sha256': R.digest((R.ROOT / 'clarify/algorithms/zarutskiy_clarifier.py').read_bytes()),
        'final_prompt_sha256': R.digest(frozen.encode()),
        'contrast_prompt_sha256': R.digest(CONTRAST_QUESTION_PROMPT.encode()),
        'selected_tasks_sha256': R.digest(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()),
        'development_review_sha256': R.digest((R.ROOT / 'experiments/contrast_question_v1/development_review.json').read_bytes()),
        'tasks': [{'instance_id': k, 'task_id': t['task_id'], 'prompt': t['prompt'],
                   'entry_point': t['entry_point']} for k, t in selected],
        'evaluation': 'Unmodified SDK Docker evaluator; hidden tests never enter solver prompts',
        'freeze_policy': 'No revisions based on these 100 tasks; no validation calls',
        'oracle': 'SDK simulated user, reference_prompt fallback on train; not validation curated QA',
        'ordering': 'seed 1017 shuffled jobs; both arms freshly run in every round',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        assert json.loads((OUT / 'manifest.json').read_text()) == manifest, 'Frozen inputs changed'
    else:
        R.save(OUT / 'manifest.json', manifest)
        (OUT / 'algorithm.py').write_bytes(SOURCE.read_bytes())
        (OUT / 'final_prompt.txt').write_text(frozen + '\n')
        (OUT / 'question_prompt.txt').write_text(CONTRAST_QUESTION_PROMPT + '\n')
    return selected


class CachedGateway(R.ConcurrentGateway):
    """Replay already-paid exact calls when an interrupted job is resumed."""

    def __init__(self, *args):
        super().__init__(*args)
        self.cache_lock = threading.RLock()
        self.cache = {r['cache_key']: r for r in R.rows(OUT / 'call_cache.jsonl')}

    def complete(self, *args, **kwargs):
        context = self.context.copy()
        key = R.digest(json.dumps({'context': context, 'request': kwargs},
                                  sort_keys=True, ensure_ascii=False).encode())
        with self.cache_lock:
            cached = self.cache.get(key)
        if cached is not None:
            response = litellm.ModelResponse(**cached['response'])
            self.response_costs[id(response)] = cached['cost_usd']
            return response
        response = super().complete(*args, **kwargs)
        record = {'cache_key': key, 'context': context,
                  'request_sha256': R.digest(json.dumps(kwargs, sort_keys=True, ensure_ascii=False).encode()),
                  'response': response.model_dump(mode='json'),
                  'cost_usd': self.response_costs[id(response)]}
        with self.cache_lock:
            R.append(OUT / 'call_cache.jsonl', record)
            self.cache[key] = record
        return response


def solve(job, gateway):
    number, arm, key, task = job
    gateway.context = {'experiment': EXPERIMENT, 'round': number, 'arm': arm,
                       'instance_id': key, 'phase': 'assistant'}
    trace = []

    def hook(phase):
        def record(message):
            gateway.context = {**gateway.context, 'phase': phase}
            trace.append({'phase': phase, **message})
        return record

    env = ClarificationEnvironment(
        ClarificationConfiguration(language_model=MODEL, clarification_model=MODEL,
                                   temperature=0.7, max_clarification_turns=1,
                                   max_prompt_budget=1.0, max_clarification_budget=1.0),
        task, llm_api_hook=hook('assistant'), clarification_api_hook=hook('human'))
    algorithm = ARMS[arm]()
    with test_context():
        code = algorithm.run(env, {'prompt': task['prompt'], 'entry_point': task['entry_point']})
    return {'round': number, 'arm': arm, 'instance_id': key, 'task_id': task['task_id'],
            'prompt': task['prompt'], 'prompt_result': code,
            'clarification_history': env.history,
            'prompt_cost': env.prompt_cost, 'clarification_cost': env.clarification_cost,
            'question_analysis': getattr(algorithm, 'question_analysis', None), 'trace': trace,
            'protocol_issue': len(env.history) != 1 or not env.history[0][3].strip()}


def run():
    selected = prepare()
    ledger = R.cumulative_ledger()
    R.append(OUT / 'invocations.jsonl', {'starting_cost': ledger.charged, 'time': time.time()})
    with (R.ROOT / 'runs/zenmux_cumulative.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for attempt in range(1, 11):
            done = {(r['round'], r['arm'], r['instance_id']) for r in R.rows(OUT / 'responses.jsonl')}
            if len(done) == 600:
                R.save(OUT / 'generation_status.json', {'state': 'complete', 'generated': 600})
                return
            R.save(OUT / 'generation_status.json', {'state': 'running', 'attempt': attempt})
            try:
                prices = fetch_prices()
                R.append(OUT / 'prices.jsonl', prices)
                jobs = [(n, a, k, t) for n in (1, 2, 3) for a in ARMS for k, t in selected
                        if (n, a, k) not in done]
                random.Random(1017).shuffle(jobs)
                failure = None
                with CachedGateway(read_key(R.ROOT / '.env'), prices, ledger) as gateway:
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        iterator = iter(jobs)
                        active = {}

                        def submit():
                            job = next(iterator, None)
                            if job and not gateway.stopped.is_set():
                                active[pool.submit(solve, job, gateway)] = job[:3]

                        for _ in range(8):
                            submit()
                        while active:
                            finished, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
                            for future in finished:
                                job_id = active.pop(future)
                                try:
                                    result = future.result()
                                    R.append(OUT / 'responses.jsonl', result)
                                    done.add(job_id)
                                    print(f'GENERATED {len(done)}/600 r{result["round"]} {result["arm"]} cumulative=${ledger.charged:.6f}', flush=True)
                                except BaseException as exc:
                                    gateway.stopped.set()
                                    # The triggering API error is more informative than workers
                                    # observing the gateway's subsequently set stop flag.
                                    if failure is None or (isinstance(exc, RunStopped) and
                                            str(exc).startswith('API request failed:')):
                                        failure = exc
                                    reason = str(exc) if isinstance(exc, RunStopped) else type(exc).__name__
                                    R.append(OUT / 'interruptions.jsonl', {'job': job_id, 'reason': reason})
                                if failure is None:
                                    submit()
                if failure:
                    raise failure
            except RunStopped as exc:
                reason = str(exc)
                print(f'STOP attempt {attempt}: {reason}', flush=True)
                transient = reason.startswith('API request failed:') or reason.startswith('Could not retrieve ZenMux prices')
                if transient and attempt < 10:
                    time.sleep(5)
                    continue
                R.save(OUT / 'generation_status.json', {'state': 'stopped', 'reason': reason})
                raise
            except BaseException as exc:
                R.save(OUT / 'generation_status.json', {'state': 'stopped', 'reason': type(exc).__name__})
                raise
        assert len(R.rows(OUT / 'responses.jsonl')) == 600
        R.save(OUT / 'generation_status.json', {'state': 'complete', 'generated': 600})


def metrics(records):
    if not records:
        return {'evaluated': 0}
    return {'evaluated': len(records), 'passed': sum(bool(r['success']) for r in records),
            'pass_at_1': statistics.mean(bool(r['success']) for r in records),
            'tds': _turn_discounted_sucess(records),
            'ndcg': _normalized_discounted_cumulative_gain(records),
            'clarification_rate': statistics.mean(bool(r['clarification_history']) for r in records),
            'quality_counts': dict(Counter(h[2] for r in records for h in r['clarification_history'])),
            'protocol_issue_count': sum(r['protocol_issue'] for r in records),
            'average_total_model_cost_usd': statistics.mean(r['prompt_cost'] + r['clarification_cost'] for r in records),
            'overasking_rate': None}


def report():
    records = R.rows(OUT / 'results.jsonl')
    assert len(records) == len({(r['round'], r['arm'], r['instance_id']) for r in records})
    output = {'complete': len(records) == 600, 'rounds': []}
    for number in (1, 2, 3):
        groups = {a: {r['instance_id']: r for r in records if r['round'] == number and r['arm'] == a} for a in ARMS}
        paired = groups['baseline'].keys() & groups['contrast'].keys()
        rescued = sum(groups['contrast'][k]['success'] and not groups['baseline'][k]['success'] for k in paired)
        lost = sum(groups['baseline'][k]['success'] and not groups['contrast'][k]['success'] for k in paired)
        n = rescued + lost
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(rescued, lost) + 1)) / 2**n) if n else 1.0
        output['rounds'].append({'round': number, **{a: metrics(list(g.values())) for a, g in groups.items()},
                                 'paired': len(paired), 'rescued': rescued, 'regressed': lost,
                                 'net_gain': rescued - lost, 'exact_mcnemar_p': p})
    if output['complete']:
        keys = sorted({r['instance_id'] for r in records})
        index = {(r['round'], r['arm'], r['instance_id']): r for r in records}
        differences = [statistics.mean(int(index[n, 'contrast', k]['success']) - int(index[n, 'baseline', k]['success']) for n in (1, 2, 3)) for k in keys]
        rng = random.Random(1017)
        boots = sorted(statistics.mean(rng.choices(differences, k=100)) for _ in range(5000))
        output['mean_pass_at_1_difference'] = statistics.mean(differences)
        output['task_grouped_bootstrap_95_interval'] = [boots[124], boots[4874]]
        output['average_metrics'] = {a: metrics([r for r in records if r['arm'] == a]) for a in ARMS}
    ledger = R.cumulative_ledger()
    calls = [r for r in ledger.latest.values() if r.get('experiment') == EXPERIMENT]
    output.update(cumulative_cost_usd=ledger.charged, cumulative_budget_usd=R.BUDGET,
                  experiment_conservative_cost_usd=sum(r['reserved_or_estimated_usd'] for r in calls),
                  call_statuses=dict(Counter(r['status'] for r in calls)),
                  limitations=['Train development experiment, not official leaderboard results.',
                               'Simulated user uses reference_prompt fallback; question ratings can disagree with test behavior.',
                               'Original task IDs are independent sampling units; repeated rounds are not 300 independent tasks per arm.',
                               'Overasking unavailable: training tasks do not have reliable curated requirement labels.',
                               'Bootstrap interval resamples original tasks and is exploratory; three repeats cannot establish small effects.'])
    R.save(OUT / 'summary.json', output)
    return output


def evaluate_watch():
    selected = prepare()
    evaluator = init_evalplus_evaluator([t for _, t in selected])
    while True:
        done = {(r['round'], r['arm'], r['instance_id']) for r in R.rows(OUT / 'results.jsonl')}
        for row in R.rows(OUT / 'responses.jsonl'):
            identity = (row['round'], row['arm'], row['instance_id'])
            if identity in done:
                continue
            success, status = evaluator.partial(row).evaluate(None, '```python\n' + row['prompt_result'].strip() + '\n```')
            R.append(OUT / 'results.jsonl', row | {'success': success, 'test_result': status})
            done.add(identity)
            print(f'EVALUATED {len(done)}/600 r{row["round"]} {row["arm"]} {success}', flush=True)
        summary = report()
        if summary['complete']:
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            return
        status_path = OUT / 'generation_status.json'
        if status_path.exists() and json.loads(status_path.read_text())['state'] == 'stopped':
            print('Generation stopped; available outputs evaluated.', flush=True)
            return
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'evaluate_watch', 'report'])
    args = parser.parse_args()
    try:
        result = globals()[args.action]()
        if args.action == 'prepare':
            print('Prepared 100 original tasks, two arms, three rounds; frozen CoT final prompt; $14 cumulative cap.')
        elif args.action == 'report':
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except RunStopped as exc:
        print(str(exc), flush=True)
        raise SystemExit(1)
