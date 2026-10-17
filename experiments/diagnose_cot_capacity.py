"""Repeat final generation on 50 seeded failures; hidden tests are diagnostic only."""
import argparse
import fcntl
import random
from concurrent.futures import ThreadPoolExecutor

from clarify.algorithms.zarutskiy_clarifier import ZarutskiyClarifier, as_code
from clarify.env import ClarificationConfiguration, ClarificationEnvironment, test_context
from clarify.runtime import init_evalplus_evaluator
from experiments import reproduce_zarutskiy as R
from experiments.zenmux import MODEL, RunStopped, fetch_prices, read_key

OUT = R.ROOT / 'runs/cot_capacity_50x5'
BASE = R.ROOT / 'runs/cot_val/results.jsonl'


def prepare():
    baseline = R.rows(BASE)
    assert len(baseline) == 770 and sum(bool(r['success']) for r in baseline) == 547
    failures = sorted((r for r in baseline if not r['success']), key=lambda r: r['instance_id'])
    selected = random.Random(100).sample(failures, 50)
    tasks = dict(R.load_records())
    manifest = {'seed': 100, 'repetitions': 5, 'model': MODEL, 'temperature': 0.7,
                'budget_usd': 8.0, 'baseline_sha256': R.digest(BASE.read_bytes()),
                'selection': [r['instance_id'] for r in selected],
                'scope': 'Conditional recovery on original failures with frozen clarification; not full-set pass@1'}
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        import json
        assert json.loads((OUT / 'manifest.json').read_text()) == manifest
    else:
        R.save(OUT / 'manifest.json', manifest)
    for row in selected:
        users = [t['content'] for t in row['trace'] if t['phase'] == 'assistant' and t['role'] == 'user']
        assert len(users) == 3 and users[-1].startswith('Implement `')
        row['frozen_prompt'] = users[-1]
        assert tasks[row['instance_id']]['prompt'] == row['prompt']
    R.save(OUT / 'selected.json', selected)
    return selected, tasks


def run():
    selected, tasks = prepare()
    ledger = R.cumulative_ledger()
    prices = fetch_prices()
    R.append(OUT / 'invocations.jsonl', {'prices': prices, 'starting_cost': ledger.charged})
    done = {(r['instance_id'], r['repeat']) for r in R.rows(OUT / 'responses.jsonl')}
    jobs = [(r, k) for k in range(5) for r in selected if (r['instance_id'], k) not in done]
    with R.ConcurrentGateway(read_key(R.ROOT / '.env'), prices, ledger) as gateway:
        def generate(job):
            row, k = job
            gateway.context = {'experiment': 'cot_capacity_50x5', 'instance_id': row['instance_id'], 'repeat': k, 'phase': 'assistant'}
            trace = []
            env = ClarificationEnvironment(ClarificationConfiguration(language_model=MODEL, temperature=0.7, max_prompt_budget=1.0), tasks[row['instance_id']], llm_api_hook=lambda m: trace.append(m))
            algorithm = ZarutskiyClarifier()
            with test_context():
                code, raw = algorithm.generate_code(env, row['frozen_prompt'])
                code = algorithm.clean(code or as_code(raw), tasks[row['instance_id']])
            return {key: row[key] for key in ('instance_id', 'task_id', 'prompt', 'clarification_history')} | {'repeat': k, 'prompt_result': code, 'trace': trace}
        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(generate, jobs):
                R.append(OUT / 'responses.jsonl', result)
                done.add((result['instance_id'], result['repeat']))
                print(f'GENERATED {len(done)}/250 cumulative=${ledger.charged:.6f}', flush=True)


def evaluate():
    selected, tasks = prepare()
    evaluator = init_evalplus_evaluator([tasks[r['instance_id']] for r in selected])
    done = {(r['instance_id'], r['repeat']) for r in R.rows(OUT / 'results.jsonl')}
    for r in R.rows(OUT / 'responses.jsonl'):
        if (r['instance_id'], r['repeat']) in done:
            continue
        success, status = evaluator.partial(r).evaluate(None, '```python\n' + r['prompt_result'].strip() + '\n```')
        R.append(OUT / 'results.jsonl', r | {'success': success, 'test_result': status})
        done.add((r['instance_id'], r['repeat']))
        print(f'EVALUATED {len(done)}/250 {success}', flush=True)
    results = R.rows(OUT / 'results.jsonl')
    counts = {r['instance_id']: sum(bool(x['success']) for x in results if x['instance_id'] == r['instance_id']) for r in selected}
    summary = {'evaluated': len(results), 'successful_generations': sum(counts.values()), 'recovered_instances': sum(v > 0 for v in counts.values()), 'success_counts': counts, 'complete': len(done) == 250}
    R.save(OUT / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'evaluate'])
    args = parser.parse_args()
    with (R.ROOT / 'runs/zenmux_cumulative.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            globals()[args.action]()
        except RunStopped as exc:
            print(str(exc), flush=True)
            raise SystemExit(1)
