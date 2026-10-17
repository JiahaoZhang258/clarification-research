"""Paired final-generation comparison on frozen training clarifications."""
from __future__ import annotations

import argparse
import ast
import fcntl
import json
import math
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from clarify.algorithms.zarutskiy_clarifier import ZarutskiyClarifier, as_code
from clarify.env import ClarificationConfiguration, ClarificationEnvironment, test_context
from clarify.runtime import init_evalplus_evaluator
from evaluate_responses import _normalized_discounted_cumulative_gain, _turn_discounted_sucess
from experiments import reproduce_zarutskiy as R
from experiments.compare_question_selection import load_dev_tasks
from experiments.zenmux import MODEL, RunStopped, fetch_prices, read_key

OUT = R.ROOT / 'runs/prompt_distillation_v1'
SOURCE = R.ROOT / 'runs/devset100/targeted_responses.jsonl'
PROMPT = R.ROOT / 'experiments/prompt_distillation/evidence_checked_final_v1.txt'
PROVENANCE = PROMPT.with_suffix('.provenance.json')
BASELINE = R.ROOT / 'runs/cot_val/sources/zarutskiy_clarifier.py'
EXPERIMENT = 'prompt_distillation_v1'


def prepare():
    provenance = json.loads(PROVENANCE.read_text())
    excluded = set(provenance['excluded_task_ids_for_future_comparison'])
    selected = [r for r in R.rows(SOURCE) if r['task_id'] not in excluded]
    assert len(selected) == 97 and len({r['instance_id'] for r in selected}) == 97
    tasks = load_dev_tasks()
    train_ids = set((R.ROOT / 'data/splits/train.txt').read_text().splitlines())
    val_ids = set((R.ROOT / 'data/splits/validation.txt').read_text().splitlines())
    for r in selected:
        assert r['task_id'] in train_ids and r['task_id'] not in val_ids
        assert tasks[r['instance_id']]['prompt'] == r['prompt']
        assert len(r['clarification_history']) == 1
    tree = ast.parse(BASELINE.read_text())
    baseline = next(ast.literal_eval(n.value.func.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'COT_FINAL_PROMPT' for t in n.targets))
    prompts = {'cot': baseline.strip(), 'distilled': PROMPT.read_text().strip()}
    assert R.digest(PROMPT.read_bytes()) == provenance['prompt_sha256']
    for p in prompts.values():
        assert set(re.findall(r'\{(\w+)\}', p)) == {'entry_point', 'prompt', 'question', 'answer'}
    manifest = {'experiment': EXPERIMENT, 'count': len(selected), 'model': MODEL,
                'temperature': 0.7, 'budget_usd': 8.0, 'arms': list(prompts),
                'source_sha256': R.digest(SOURCE.read_bytes()),
                'baseline_source_sha256': R.digest(BASELINE.read_bytes()),
                'prompt_sha256': {k: R.digest(v.encode()) for k, v in prompts.items()},
                'excluded_task_ids': sorted(excluded),
                'instance_ids': [r['instance_id'] for r in selected],
                'scope': 'Final generation only; fixed historical targeted questions and answers; train development, not validation score',
                'order': 'seed 100 shuffled instances, alternating first arm; two workers',
                'evaluation': 'SDK Docker evaluator default mode, same as existing devset100 experiments'}
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        assert json.loads((OUT / 'manifest.json').read_text()) == manifest
    else:
        R.save(OUT / 'manifest.json', manifest)
        R.save(OUT / 'selected.json', [{k: r[k] for k in ('instance_id', 'task_id', 'prompt', 'clarification_history', 'need_clarification')} for r in selected])
        for name, prompt in prompts.items():
            (OUT / f'{name}_prompt.txt').write_text(prompt + '\n')
    return selected, tasks, prompts


def render(template, task, row):
    h = row['clarification_history'][0]
    values = {'prompt': task['prompt'], 'entry_point': task['entry_point'], 'question': h[0], 'answer': h[3]}
    return re.sub(r'\{(entry_point|prompt|question|answer)\}', lambda m: values[m[1]], template)


def run():
    selected, tasks, prompts = prepare()
    ledger = R.cumulative_ledger()
    prices = fetch_prices()
    R.append(OUT / 'invocations.jsonl', {'prices': prices, 'starting_cost': ledger.charged})
    done = {(r['instance_id'], r['arm']) for r in R.rows(OUT / 'responses.jsonl')}
    random.Random(100).shuffle(selected)
    jobs = [(r, arm) for i, r in enumerate(selected) for arm in (list(prompts) if i % 2 == 0 else list(reversed(prompts))) if (r['instance_id'], arm) not in done]
    output_lock = threading.Lock()
    with R.ConcurrentGateway(read_key(R.ROOT / '.env'), prices, ledger) as gateway:
        def generate(job):
            row, arm = job
            gateway.context = {'experiment': EXPERIMENT, 'instance_id': row['instance_id'], 'arm': arm, 'phase': 'assistant'}
            trace = []
            task = tasks[row['instance_id']]
            env = ClarificationEnvironment(ClarificationConfiguration(language_model=MODEL, temperature=0.7, max_prompt_budget=1.0), task, llm_api_hook=lambda m: trace.append(m))
            algorithm = ZarutskiyClarifier()
            with test_context():
                code, raw = algorithm.generate_code(env, render(prompts[arm], task, row))
                code = algorithm.clean(code or as_code(raw), task)
            result = {k: row[k] for k in ('instance_id', 'task_id', 'prompt', 'clarification_history', 'need_clarification')}
            result.update(arm=arm, prompt_result=code, trace=trace, final_generation_cost_usd=env.prompt_cost)
            with output_lock:
                R.append(OUT / 'responses.jsonl', result)
                done.add((row['instance_id'], arm))
                print(f'GENERATED {len(done)}/194 {arm} cumulative=${ledger.charged:.6f}', flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(generate, jobs):
                pass


def report():
    selected, _, _ = prepare()
    results = R.rows(OUT / 'results.jsonl')
    groups = {arm: {r['instance_id']: r for r in results if r['arm'] == arm} for arm in ('cot', 'distilled')}
    assert len(results) == sum(len(g) for g in groups.values()), 'Duplicate evaluation result'
    matched = set(groups['cot']) & set(groups['distilled'])
    better = [i for i in matched if groups['distilled'][i]['success'] and not groups['cot'][i]['success']]
    worse = [i for i in matched if groups['cot'][i]['success'] and not groups['distilled'][i]['success']]
    n = len(better) + len(worse)
    pvalue = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(len(better), len(worse)) + 1)) / 2**n) if n else 1.0
    metrics = {}
    for arm, rows in groups.items():
        a = list(rows.values())
        if not a:
            continue
        metrics[arm] = {'evaluated': len(a), 'passed': sum(bool(r['success']) for r in a),
                        'pass_at_1': sum(bool(r['success']) for r in a)/len(a),
                        'tds': _turn_discounted_sucess(a), 'ndcg': _normalized_discounted_cumulative_gain(a),
                        'clarification_rate': sum(bool(r['clarification_history']) for r in a)/len(a),
                        'avg_final_generation_cost_usd': sum(r['final_generation_cost_usd'] for r in a)/len(a)}
    ledger = R.cumulative_ledger()
    calls = [r for r in ledger.latest.values() if r.get('experiment') == EXPERIMENT]
    summary = {'complete': len(matched) == len(selected), 'paired_records': len(matched), 'metrics': metrics,
               'rescued': len(better), 'regressed': len(worse), 'net_gain': len(better)-len(worse),
               'exact_mcnemar_p': pvalue, 'rescued_ids': sorted(better), 'regressed_ids': sorted(worse),
               'conservative_new_cost_usd': sum(r['reserved_or_estimated_usd'] for r in calls),
               'cumulative_cost_usd': ledger.charged,
               'call_statuses': dict(Counter(r['status'] for r in calls)),
               'limitations': ['Train development set; not official validation score.', 'Fixed QA means question-quality metrics cannot measure a prompt improvement.', 'Cost metric is final generation only, not leaderboard total solver cost.', 'Some records share original task IDs; exact McNemar p is a record-level exploratory calculation, not independent-task confirmation.']}
    R.save(OUT / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def evaluate():
    selected, tasks, _ = prepare()
    evaluator = init_evalplus_evaluator([tasks[r['instance_id']] for r in selected])
    done = {(r['instance_id'], r['arm']) for r in R.rows(OUT / 'results.jsonl')}
    for r in R.rows(OUT / 'responses.jsonl'):
        if (r['instance_id'], r['arm']) in done:
            continue
        success, status = evaluator.partial(r).evaluate(None, '```python\n' + r['prompt_result'].strip() + '\n```')
        R.append(OUT / 'results.jsonl', r | {'success': success, 'test_result': status})
        done.add((r['instance_id'], r['arm']))
        print(f'EVALUATED {len(done)}/194 {r["arm"]} {success}', flush=True)
    report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'evaluate', 'report'])
    args = parser.parse_args()
    with (R.ROOT / 'runs/zenmux_cumulative.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            globals()[args.action]()
        except RunStopped as exc:
            print(str(exc), flush=True)
            raise SystemExit(1)
