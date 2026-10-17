"""Prepare, run, and evaluate a paired local experiment using the official SDK."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

from clarify.algorithms.assumption_audit import AssumptionAudit
from clarify.algorithms.example_confirmation import ExampleConfirmation
from clarify.data import load_split
from clarify.env import ClarificationConfiguration, ClarificationEnvironment, test_context
from clarify.runtime import init_evalplus_evaluator
from evaluate_responses import (
    _normalized_discounted_cumulative_gain,
    _turn_discounted_sucess,
)
from experiments.zenmux import MODEL, Ledger, RunStopped, ZenMuxGateway, fetch_prices, read_key

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"example": ExampleConfirmation, "audit": AssumptionAudit}


def identity(problem: dict) -> str:
    value = problem["task_id"] + "\0" + problem["prompt"]
    return hashlib.sha256(value.encode()).hexdigest()


def append_json(path: Path, value: dict) -> None:
    with path.open("a") as out:
        out.write(json.dumps(value, ensure_ascii=False) + "\n")
        out.flush()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def load_tasks() -> dict[str, dict]:
    # Preserve the official preprocessing and oracle clarifications. The algorithm
    # receives only prompt + entry_point, as in generate_responses.py.
    tasks = {identity(p): p for p in load_split(str(ROOT / "data/splits/train.txt"))}
    clarification_file = ROOT / "data/clarifications.jsonl"
    if clarification_file.exists():
        clarifications = {
            (p["task_id"], p["prompt"]): p["clarifications"] for p in rows(clarification_file)
        }
        for p in tasks.values():
            key = (p["task_id"], p["prompt"])
            if key in clarifications:
                p["clarifications"] = clarifications[key]
    return tasks


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "manifest.json"
    if target.exists():
        print("Reusing fixed manifest:", target)
        return
    tasks = load_tasks()
    rng = random.Random(42)
    chosen = []
    for family in ("Mbpp", "HumanEval"):
        pool = sorted(k for k, p in tasks.items() if p["task_id"].startswith(family + "/"))
        chosen.append(rng.sample(pool, 20))
    # Interleave families so the first five tasks exercise both dataset formats.
    selected = [k for pair in zip(*chosen, strict=True) for k in pair]
    manifest = {
        "split": "train",
        "seed": 42,
        "sampling": "20 uniformly sampled unique task/prompt pairs per family; interleaved",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "model": MODEL,
        "clarification_model": MODEL,
        "temperature": 0.7,
        "oracle_temperature": 0.0,
        "max_clarification_turns": 1,
        "num_samples": 1,
        "max_prompt_budget": 1.0,
        "max_clarification_budget": 1.0,
        "max_output_tokens": 8192,
        "request_timeout_seconds": 120,
        "automatic_retries": 0,
        "tasks": [
            {
                "key": k,
                "task_id": tasks[k]["task_id"],
                "prompt": tasks[k]["prompt"],
                "entry_point": tasks[k]["entry_point"],
            }
            for k in selected
        ],
    }
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Prepared 40 fixed training examples:", target)


def execute(output: Path, budget: float, limit: int) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    sources = output / "sources"
    sources.mkdir(exist_ok=True)
    fingerprints = {}
    for method, cls in METHODS.items():
        source = ROOT / (cls.__module__.replace(".", "/") + ".py")
        snapshot = sources / source.name
        content = source.read_bytes()
        if snapshot.exists() and snapshot.read_bytes() != content:
            raise ValueError("Algorithm changed; use a new experiment directory for new results.")
        snapshot.write_bytes(content)
        fingerprints[method] = hashlib.sha256(content).hexdigest()
    manifest["algorithm_sha256"] = fingerprints
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tasks = load_tasks()
    selected = manifest["tasks"][:limit]
    for item in selected:
        if item["key"] not in tasks:
            raise ValueError("Dataset changed after preparation; refusing to change the sample.")
    prices = fetch_prices()
    (output / "prices.json").write_text(json.dumps(prices, indent=2) + "\n")
    ledger = Ledger(output / "usage.jsonl", budget)
    (output / "budget.json").write_text(json.dumps({"total_limit_usd": budget}) + "\n")
    completed = {
        method: {r["key"] for r in rows(output / f"{method}_responses.jsonl")} for method in METHODS
    }
    environment_config = ClarificationConfiguration(
        language_model=MODEL,
        clarification_model=MODEL,
        temperature=0.7,
        max_clarification_turns=1,
        max_prompt_budget=1.0,
        max_clarification_budget=1.0,
    )
    with ZenMuxGateway(read_key(ROOT / ".env"), prices, ledger) as gateway:
        if not any(
            r["status"] == "completed" and r.get("phase") == "connectivity"
            for r in ledger.latest.values()
        ):
            response = gateway.complete(
                model=MODEL,
                messages=[{"role": "user", "content": "Reply with OK only."}],
                temperature=0,
                max_tokens=16,
            )
            if not response.choices[0].message.content:
                raise RunStopped("Connectivity check returned no text.")
            print("ZenMux connectivity confirmed through LiteLLM.", flush=True)

        for index, item in enumerate(selected):
            task = tasks[item["key"]]
            # Alternate method order; both methods see the identical public prompt.
            method_order = list(METHODS) if index % 2 == 0 else list(reversed(METHODS))
            for method in method_order:
                if item["key"] in completed[method]:
                    continue
                gateway.context = {"method": method, "key": item["key"], "phase": "assistant"}
                trace = []

                def hook(phase, trace=trace):
                    def record(message):
                        gateway.context["phase"] = phase
                        trace.append({"phase": phase, **message})

                    return record

                env = ClarificationEnvironment(
                    environment_config,
                    task,
                    llm_api_hook=hook("assistant"),
                    clarification_api_hook=hook("human"),
                )
                started = time.monotonic()
                try:
                    with test_context():
                        code = METHODS[method]().run(
                            env,
                            {
                                "prompt": task["prompt"],
                                "entry_point": task["entry_point"],
                            },
                        )
                except RunStopped:
                    append_json(
                        output / "interrupted.jsonl",
                        {
                            "key": item["key"],
                            "method": method,
                            "trace": trace,
                        },
                    )
                    raise
                except Exception as exc:
                    append_json(
                        output / "interrupted.jsonl",
                        {
                            "key": item["key"],
                            "method": method,
                            "error_type": type(exc).__name__,
                            "trace": trace,
                        },
                    )
                    raise RunStopped("Algorithm failed: " + type(exc).__name__) from None
                result = {
                    "key": item["key"],
                    "task_id": task["task_id"],
                    "prompt": task["prompt"],
                    "algorithm": METHODS[method].__name__,
                    "model": MODEL,
                    "prompt_result": code,
                    "need_clarification": len(task.get("clarifications", [None])) > 0,
                    "clarification_history": env.history,
                    "prompt_cost": env.prompt_cost,
                    "clarification_cost": env.clarification_cost,
                    "elapsed_seconds": time.monotonic() - started,
                    "trace": trace,
                }
                append_json(output / f"{method}_responses.jsonl", result)
                print(
                    f"[{index + 1}/{limit}] {method}: {task['task_id']} | "
                    f"questions={len(env.history)} | "
                    f"total conservative estimate=${ledger.charged:.6f}",
                    flush=True,
                )


def evaluate(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text())
    tasks = load_tasks()
    benchmark = [tasks[item["key"]] for item in manifest["tasks"]]
    # Use the official test constructor, Docker execution, and success criterion.
    evaluator = init_evalplus_evaluator(benchmark)
    for method in METHODS:
        target = output / f"{method}_results.jsonl"
        completed = {r["key"] for r in rows(target)}
        for result in rows(output / f"{method}_responses.jsonl"):
            if result["key"] in completed:
                continue
            code = result["prompt_result"]
            if "```python" not in code:
                code = f"```python\n{code.strip()}\n```"
            success, status = evaluator.partial(result).evaluate(None, code)
            append_json(target, {**result, "success": success, "test_result": status})
            print(f"{method}: {result['task_id']} {'PASS' if success else 'FAIL'}", flush=True)
    report(output)


def report(output: Path) -> None:
    all_results = {method: rows(output / f"{method}_results.jsonl") for method in METHODS}
    paired = set.intersection(*({r["key"] for r in values} for values in all_results.values()))
    summary = {"paired_tasks": len(paired), "methods": {}}
    for method, values in all_results.items():
        values = [r for r in values if r["key"] in paired]
        if not values:
            continue
        questions = [h for r in values for h in r["clarification_history"]]
        summary["methods"][method] = {
            "count": len(values),
            "passed": sum(r["success"] for r in values),
            "pass_at_1": sum(r["success"] for r in values) / len(values),
            "tds": _turn_discounted_sucess(values),
            "ndcg": _normalized_discounted_cumulative_gain(values),
            "asked_tasks": sum(bool(r["clarification_history"]) for r in values),
            "quality_3_questions": sum(h[2] == "3" for h in questions),
            "conservative_estimated_cost_usd": sum(
                r["prompt_cost"] + r["clarification_cost"] for r in values
            ),
        }
    budget_file = output / "budget.json"
    if budget_file.exists():
        budget = json.loads(budget_file.read_text())["total_limit_usd"]
        summary["budget_usd"] = budget
        summary["all_calls_reserved_or_estimated_usd"] = Ledger(
            output / "usage.jsonl", budget
        ).charged
    summary["notes"] = (
        "Exploratory training subset, one sample per task; not a validation leaderboard result. "
        "TDS/nDCG use the official functions. Costs use public ZenMux prices without cache "
        "discounts; pending or failed calls retain conservative reservations."
    )
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "evaluate", "report"])
    parser.add_argument("--output", type=Path, default=ROOT / "runs/methods34")
    parser.add_argument(
        "--budget", type=float, help="Required total USD cap for API calls, including resumed calls"
    )
    parser.add_argument("--limit", type=int, default=40, choices=range(1, 41), metavar="1..40")
    args = parser.parse_args()
    if args.action == "run" and args.budget is None:
        parser.error("run requires an explicitly authorized --budget")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.action == "prepare":
                prepare(args.output)
            elif args.action == "run":
                execute(args.output, args.budget, args.limit)
            elif args.action == "evaluate":
                evaluate(args.output)
            else:
                report(args.output)
        except RunStopped as exc:
            print(str(exc), flush=True)
            raise SystemExit(2) from None


if __name__ == "__main__":
    main()
