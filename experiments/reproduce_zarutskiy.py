"""Reproduce the leaderboard-pinned algorithm on all official validation records."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from clarify.data import load_split
from clarify.env import ClarificationConfiguration, ClarificationEnvironment, test_context
from clarify.runtime import init_evalplus_evaluator
from evaluate_responses import _compute_output_row
from experiments.zenmux import MODEL, Ledger, RunStopped, ZenMuxGateway, fetch_prices, read_key
from generate_responses import _load_clarification_algorithm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/zarutskiy_val_ae88217"
SOURCE = ROOT / "clarify/algorithms/zarutskiy_clarifier.py"
SOURCE_SHA = "8295ce0d907664506b9e7795aadbc71b78586516f139de75f9b9aef2db065612"
COMMIT = "ae882179a9b33e9439be6fe3524a84d6c701606f"
PRIOR_LEDGER = ROOT / "runs/methods34/usage.jsonl"
SHARED_LEDGER = ROOT / "runs/zenmux_cumulative_usage.jsonl"
BUDGET = 8.0
EXPERIMENT = "zarutskiy_val_ae88217"


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def instance_records(benchmark):
    """Keep duplicates as independent instances; stabilize order across SDK set iteration."""
    groups = {"HumanEval": [], "Mbpp": []}
    for task in benchmark:
        fingerprint = digest(json.dumps(task, sort_keys=True, ensure_ascii=False).encode())
        groups[task["task_id"].split("/")[0]].append((fingerprint, task))
    for group in groups.values():
        group.sort(key=lambda item: (item[1]["task_id"], item[1]["prompt"], item[0]))
    occurrences = Counter()
    records = []
    for index in range(max(map(len, groups.values()))):
        for family in ("HumanEval", "Mbpp"):
            if index >= len(groups[family]):
                continue
            fingerprint, task = groups[family][index]
            occurrences[fingerprint] += 1
            records.append((f"{fingerprint}:{occurrences[fingerprint]}", task))
    return records


def load_records():
    benchmark = load_split(str(ROOT / "data/splits/validation.txt"))
    clarifications = {
        (row["task_id"], row["prompt"]): row["clarifications"]
        for row in rows(ROOT / "data/clarifications.jsonl")
    }
    for task in benchmark:
        key = (task["task_id"], task["prompt"])
        if key in clarifications:
            task["clarifications"] = clarifications[key]
    return instance_records(benchmark)


def prepare():
    if digest(SOURCE.read_bytes()) != SOURCE_SHA:
        raise ValueError("Pinned algorithm source does not match the upstream bytes.")
    records = load_records()
    if len(records) != 770:
        raise ValueError("Expected exactly 770 official validation records.")
    cls = _load_clarification_algorithm(str(SOURCE))
    manifest = {
        "algorithm_commit": COMMIT,
        "algorithm_sha256": SOURCE_SHA,
        "algorithm_default_config": cls.DEFAULT_CONFIG,
        "sdk_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "split": "val",
        "num_samples": 1,
        "model": MODEL,
        "clarification_model": MODEL,
        "temperature": 0.7,
        "oracle_temperature": 0.0,
        "max_clarification_turns": 1,
        "max_prompt_budget": 1.0,
        "max_clarification_budget": 1.0,
        "max_output_tokens": 8192,
        "request_timeout_seconds": 120,
        "transport_retries": 0,
        "ordering": "stable task/prompt/content order interleaved by dataset; duplicates retained",
        "cumulative_budget_usd": BUDGET,
        "prior_ledger_sha256": digest(PRIOR_LEDGER.read_bytes()),
        "data_sha256": {
            str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in sorted((ROOT / "data").rglob("*"))
            if path.is_file() and (path.suffix == ".jsonl" or path.name == "validation.txt")
        },
        "tasks": [
            {
                "instance_id": key,
                "task_id": task["task_id"],
                "prompt": task["prompt"],
                "entry_point": task["entry_point"],
            }
            for key, task in records
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "manifest.json"
    if target.exists():
        if json.loads(target.read_text()) != manifest:
            raise ValueError("Run inputs or prior spending changed; refusing to mix experiments.")
    else:
        save(target, manifest)
    snapshot = OUTPUT / "sources/zarutskiy_clarifier.py"
    snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists() and digest(snapshot.read_bytes()) != SOURCE_SHA:
        raise ValueError("Stored algorithm snapshot differs from the pinned version.")
    snapshot.write_bytes(SOURCE.read_bytes())
    return records, cls


class LockedLedger(Ledger):
    def __init__(self, path, budget):
        self.mutex = threading.RLock()
        super().__init__(path, budget)

    @property
    def charged(self):
        with self.mutex:
            return super().charged

    def append(self, record):
        with self.mutex:
            super().append(record)

    def reserve(self, amount, context):
        with self.mutex:
            return super().reserve(amount, context)


def cumulative_ledger():
    if not SHARED_LEDGER.exists():
        temporary = SHARED_LEDGER.with_suffix(".tmp")
        temporary.write_bytes(PRIOR_LEDGER.read_bytes())
        temporary.replace(SHARED_LEDGER)
    ledger = LockedLedger(SHARED_LEDGER, BUDGET)
    prior = Ledger(PRIOR_LEDGER, BUDGET)
    if any(ledger.latest.get(key) != value for key, value in prior.latest.items()):
        raise ValueError("Cumulative ledger no longer includes the exact prior spending.")
    return ledger


class ConcurrentGateway(ZenMuxGateway):
    """Keep task metadata isolated while serializing every budget reservation."""

    def __init__(self, key, prices, ledger):
        self.local = threading.local()
        self.stopped = threading.Event()
        super().__init__(key, prices, ledger)

    @property
    def context(self):
        return getattr(self.local, "context", {})

    @context.setter
    def context(self, value):
        self.local.context = value

    def complete(self, *args, **kwargs):
        if self.stopped.is_set():
            raise RunStopped("A request stopped the experiment; no new calls allowed.")
        try:
            return super().complete(*args, **kwargs)
        except BaseException:
            self.stopped.set()
            raise


def solve(item, cls, gateway):
    key, task = item
    gateway.context = {
        "experiment": EXPERIMENT,
        "instance_id": key,
        "method": "ZarutskiyClarifier",
        "phase": "assistant",
    }
    trace = []

    def hook(phase):
        def record(message):
            gateway.context["phase"] = phase
            trace.append({"phase": phase, **message})

        return record

    env = ClarificationEnvironment(
        ClarificationConfiguration(
            language_model=MODEL,
            clarification_model=MODEL,
            temperature=0.7,
            max_clarification_turns=1,
            max_prompt_budget=1.0,
            max_clarification_budget=1.0,
        ),
        task,
        llm_api_hook=hook("assistant"),
        clarification_api_hook=hook("human"),
    )
    started = time.monotonic()
    try:
        with test_context():
            code = cls().run(env, {"prompt": task["prompt"], "entry_point": task["entry_point"]})
    except BaseException as exc:
        return {
            "instance_id": key,
            "task_id": task["task_id"],
            "trace": trace,
            "interrupted": True,
            "error_type": type(exc).__name__,
            "reason": str(exc) if isinstance(exc, RunStopped) else "Local execution error",
        }
    return {
        "instance_id": key,
        "task_id": task["task_id"],
        "prompt": task["prompt"],
        "algorithm": cls.__name__,
        "algorithm_path": str(SOURCE.relative_to(ROOT)),
        "model": MODEL,
        "prompt_result": code,
        "need_clarification": len(task.get("clarifications", [None])) > 0,
        "clarification_history": env.history,
        "prompt_cost": env.prompt_cost,
        "clarification_cost": env.clarification_cost,
        "elapsed_seconds": time.monotonic() - started,
        "trace": trace,
    }


def execute(limit, workers):
    records, cls = prepare()
    ledger = cumulative_ledger()
    completed = {row["instance_id"] for row in rows(OUTPUT / "responses.jsonl")}
    pending = iter([item for item in records[:limit] if item[0] not in completed])
    prices = fetch_prices()
    append(
        OUTPUT / "invocations.jsonl",
        {"time": time.time(), "limit": limit, "workers": workers, "prices": prices},
    )
    print(
        f"Starting: {len(completed)} completed; cumulative estimate ${ledger.charged:.6f}",
        flush=True,
    )
    failures = []
    with ConcurrentGateway(read_key(ROOT / ".env"), prices, ledger) as gateway:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}

            def submit_one():
                item = next(pending, None)
                if item is not None and not gateway.stopped.is_set():
                    active[pool.submit(solve, item, cls, gateway)] = item[0]

            for _ in range(workers):
                submit_one()
            while active:
                finished, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
                if not finished:
                    print(
                        f"Waiting: {len(completed)}/{limit}; reserved/estimated ${ledger.charged:.6f}",
                        flush=True,
                    )
                for future in finished:
                    active.pop(future)
                    result = future.result()
                    if result.get("interrupted"):
                        gateway.stopped.set()
                        failures.append(result["reason"])
                        append(OUTPUT / "interrupted.jsonl", result)
                        print("STOP:", result["reason"], flush=True)
                    else:
                        append(OUTPUT / "responses.jsonl", result)
                        completed.add(result["instance_id"])
                        print(
                            f"[{len(completed)}/{limit}] {result['task_id']} "
                            f"questions={len(result['clarification_history'])} "
                            f"cumulative=${ledger.charged:.6f}",
                            flush=True,
                        )
                    submit_one()
    report()
    if failures:
        raise RunStopped(failures[0])


def evaluate():
    records, _ = prepare()
    evaluator = init_evalplus_evaluator([task for _, task in records])
    completed = {r["instance_id"] for r in rows(OUTPUT / "results.jsonl")}
    results = rows(OUTPUT / "responses.jsonl")
    for result in results:
        if result["instance_id"] in completed:
            continue
        code = result["prompt_result"]
        if "```python" not in code:
            code = f"```python\n{code.strip()}\n```"
        success, status = evaluator.partial(result).evaluate(None, code)
        append(OUTPUT / "results.jsonl", {**result, "success": success, "test_result": status})
        completed.add(result["instance_id"])
        print(
            f"EVAL [{len(completed)}/{len(results)}] {result['task_id']} "
            f"{'PASS' if success else 'FAIL'}",
            flush=True,
        )
    report()


def report():
    generated = rows(OUTPUT / "responses.jsonl")
    evaluated = rows(OUTPUT / "results.jsonl")
    usage = (
        Ledger(SHARED_LEDGER, BUDGET) if SHARED_LEDGER.exists() else Ledger(PRIOR_LEDGER, BUDGET)
    )
    calls = [r for r in usage.latest.values() if r.get("experiment") == EXPERIMENT]
    completed = [r for r in calls if r["status"] == "completed"]
    prior = json.loads((ROOT / "runs/methods34/billing_final.json").read_text())
    reported = sum(r.get("provider_reported_cost_usd") or 0 for r in completed)
    summary = {
        "expected": 770,
        "generated": len(generated),
        "evaluated": len(evaluated),
        "complete": len(generated) == len(evaluated) == 770,
        "api_calls": len(calls),
        "completed_api_calls": len(completed),
        "unknown_cost_calls": len(calls) - len(completed),
        "calls_without_reported_cost": sum(
            r.get("provider_reported_cost_usd") is None for r in completed
        ),
        "new_provider_reported_usd": reported,
        "cumulative_provider_reported_usd": prior["total_provider_reported_usd"] + reported,
        "new_conservative_usd": sum(r["reserved_or_estimated_usd"] for r in calls),
        "cumulative_conservative_usd": usage.charged,
        "budget_usd": BUDGET,
        "truncated_calls": sum(r.get("finish_reason") == "length" for r in completed),
        "question_counts": dict(Counter(len(r["clarification_history"]) for r in generated)),
    }
    if evaluated:
        summary["metrics"] = _compute_output_row(evaluated)
        summary["passed"] = sum(r["success"] for r in evaluated)
        summary["by_dataset"] = {}
        for family in ("HumanEval", "Mbpp"):
            subset = [r for r in evaluated if r["task_id"].startswith(family + "/")]
            summary["by_dataset"][family] = {
                "n": len(subset),
                "passed": sum(r["success"] for r in subset),
            }
    save(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "evaluate", "report"])
    parser.add_argument("--limit", type=int, default=770)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.limit <= 770 or not 1 <= args.workers <= 4:
        parser.error("limit must be 1..770 and workers 1..4")
    if args.command == "run":
        with (ROOT / "runs/zenmux_cumulative.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                execute(args.limit, args.workers)
            except RunStopped as exc:
                print("Experiment stopped:", str(exc), flush=True)
                raise SystemExit(2) from None
    elif args.command == "prepare":
        records, _ = prepare()
        print(f"Prepared {len(records)} instances; source SHA256 {SOURCE_SHA}")
    elif args.command == "evaluate":
        evaluate()
    else:
        report()


if __name__ == "__main__":
    main()
