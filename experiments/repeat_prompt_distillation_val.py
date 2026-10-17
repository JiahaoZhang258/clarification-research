"""Three additional frozen validation runs; no baseline regeneration."""
from __future__ import annotations

import argparse
import fcntl
import json
import statistics
import time

from experiments import compare_prompt_distillation_val as V
from experiments.zenmux import RunStopped

BASE = V.R.ROOT / 'runs/prompt_distillation_val_repeats_v1'
INITIAL = V.R.ROOT / 'runs/prompt_distillation_val_reuse_v1'
V.R.BUDGET = 12.0


def configure(round_number):
    assert round_number in (1, 2, 3)
    V.OUT = BASE / f'repeat_{round_number}'
    V.EXPERIMENT = f'prompt_distillation_val_repeat_{round_number}_v1'
    V.R.BUDGET = 12.0


def prepare():
    BASE.mkdir(exist_ok=True)
    first = json.loads((INITIAL / 'manifest.json').read_text())
    for number in (1, 2, 3):
        configure(number)
        selected, tasks, prompts = V.prepare()
        manifest = json.loads((V.OUT / 'manifest.json').read_text())
        for key in ('prompt_sha256', 'source_sha256', 'baseline_results_sha256', 'instance_ids', 'model', 'temperature'):
            assert manifest[key] == first[key], key
        assert len(selected) == 770
        for row in selected:
            text = V.render(prompts['distilled'], tasks[row['instance_id']], row)
            assert row['prompt'] in text and row['clarification_history'][0][3] in text
    print('Prepared three independent 770-record runs; identical prompts, QA and settings; $12 cumulative cap.', flush=True)


def run():
    prepare()
    with (V.R.ROOT / 'runs/zenmux_cumulative.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for number in (1, 2, 3):
            configure(number)
            for attempt in range(1, 6):
                if len(V.R.rows(V.OUT / 'responses.jsonl')) == 770:
                    break
                V.R.save(BASE / 'generation_status.json', {'state': 'running', 'repeat': number, 'attempt': attempt})
                print(f'REPEAT {number}/3 ATTEMPT {attempt}', flush=True)
                try:
                    V.run()
                    break
                except RunStopped as exc:
                    reason = str(exc)
                    print(f'REPEAT {number} STOP: {reason}', flush=True)
                    if not reason.startswith('API request failed:') or attempt == 5:
                        V.R.save(BASE / 'generation_status.json', {'state': 'stopped', 'repeat': number, 'reason': reason})
                        raise
                    time.sleep(5)
            assert len(V.R.rows(V.OUT / 'responses.jsonl')) == 770
            print(f'REPEAT {number}/3 GENERATION COMPLETE', flush=True)
        V.R.save(BASE / 'generation_status.json', {'state': 'complete', 'generated': 2310})


def evaluate_watch():
    prepare()
    while True:
        completed = 0
        for number in (1, 2, 3):
            configure(number)
            generated = len(V.R.rows(V.OUT / 'responses.jsonl'))
            evaluated = len(V.R.rows(V.OUT / 'results.jsonl'))
            if generated > evaluated:
                print(f'EVALUATING REPEAT {number}/3', flush=True)
                V.evaluate()
                evaluated = len(V.R.rows(V.OUT / 'results.jsonl'))
            completed += evaluated == 770
        if completed == 3:
            summarize()
            return
        status_path = BASE / 'generation_status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'stopped':
            print('Generation stopped; all currently available outputs evaluated.', flush=True)
            return
        time.sleep(10)


def summarize():
    summaries = [json.loads((directory / 'summary.json').read_text()) for directory in [INITIAL] + [BASE / f'repeat_{i}' for i in (1, 2, 3)]]
    assert all(s['complete'] and s['paired_records'] == 770 for s in summaries)
    rows = []
    for i, s in enumerate(summaries):
        m = s['metrics']['distilled']
        rows.append({'round': i, 'label': 'initial' if i == 0 else f'repeat_{i}',
                     'passed': m['passed'], 'pass_at_1': m['pass_at_1'], 'tds': m['tds'],
                     'rescued': s['rescued'], 'regressed': s['regressed'], 'net_gain': s['net_gain'],
                     'record_level_mcnemar_p': s['exact_mcnemar_p'],
                     'avg_final_generation_cost_usd': m['avg_final_generation_cost_usd']})
    counts = [r['passed'] for r in rows]
    ledger = V.R.cumulative_ledger()
    calls = [r for r in ledger.latest.values() if r.get('experiment', '').startswith('prompt_distillation_val_repeat_')]
    summary = {'complete': True, 'baseline_passed': 547, 'baseline_reused': True,
               'rounds': rows, 'mean_passed': statistics.mean(counts),
               'min_passed': min(counts), 'max_passed': max(counts),
               'sample_stdev_passed': statistics.stdev(counts),
               'mean_pass_at_1': statistics.mean(r['pass_at_1'] for r in rows),
               'mean_tds': statistics.mean(r['tds'] for r in rows),
               'mean_net_gain_vs_single_baseline': statistics.mean(counts) - 547,
               'additional_three_runs_conservative_cost_usd': sum(r['reserved_or_estimated_usd'] for r in calls),
               'cumulative_cost_usd': ledger.charged,
               'limitations': ['Four runs describe sampling variability on this fixed validation set and frozen historical QA, not end-to-end clarification variability.', 'Only one historical baseline run; baseline variability is not estimated.', 'Do not pool 4x770 correlated observations as independent tasks.', 'No prompt revisions from validation outcomes.']}
    V.R.save(BASE / 'stability_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run', 'evaluate_watch', 'summarize'])
    args = parser.parse_args()
    try:
        globals()[args.action]()
    except RunStopped as exc:
        print(str(exc), flush=True)
        raise SystemExit(1)
