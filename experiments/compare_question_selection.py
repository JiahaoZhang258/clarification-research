"""Compare question-selection strategies on the fixed 100-task dev set.

Three arms run on the identical 100-task manifest under ``runs/devset100``:

    targeted      - the pinned first-place algorithm at its default config
                    (single targeted question, assume-framing).
    multi_first   - JevQuestionSelector with ``selection="first"``: generate up
                    to three candidate questions, ask the model's own top pick.
    multi_jev     - JevQuestionSelector with ``selection="jev"``: generate up to
                    three candidate questions, let the decision model pick one.

All arms share the same draft, single-turn clarification, final generation and
sanitizer; only the question-generation/selection step differs. Spending goes
through the shared cumulative ledger (the 2 USD cap), so a new run never resets
the budget guard.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from clarify.data import load_split
from clarify.env import ClarificationConfiguration, ClarificationEnvironment, test_context
from clarify.runtime import init_evalplus_evaluator
from evaluate_responses import (
    _normalized_discounted_cumulative_gain,
    _turn_discounted_sucess,
)
from experiments.reproduce_zarutskiy import (
    BUDGET,
    PRIOR_LEDGER,
    SHARED_LEDGER,
    SOURCE_SHA,
    ConcurrentGateway,
    append,
    cumulative_ledger,
    digest,
    rows,
    save,
)
from experiments.zenmux import MODEL, Ledger, RunStopped, fetch_prices, read_key
from generate_responses import _load_clarification_algorithm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs" / "devset100"
EXPERIMENT = "devset100_question_selection"
DECISION_MODEL = "typesafe/jev-latest"
ORIGINAL_SOURCE = ROOT / "clarify/algorithms/zarutskiy_clarifier.py"
JEV_SOURCE = ROOT / "clarify/algorithms/jev_question_selector.py"
TARGETED_JEV_SOURCE = ROOT / "clarify/algorithms/jev_targeted_selector.py"
SELF_REPAIR_SOURCE = ROOT / "clarify/algorithms/self_repair_clarifier.py"
COT_SOURCE = ROOT / "clarify/algorithms/cot_clarifier.py"

ARMS = {
    "targeted": {
        "label": "targeted",
        "source": ORIGINAL_SOURCE,
        "config": None,
        "decision_model": None,
    },
    "interface": {
        "label": "interface",
        "source": ORIGINAL_SOURCE,
        "config": {"interface_question": True},
        "decision_model": None,
    },
    "final3": {
        "label": "final3",
        "source": ORIGINAL_SOURCE,
        "config": {"num_final": 3},
        "decision_model": None,
    },
    "repair": {
        "label": "repair",
        "source": SELF_REPAIR_SOURCE,
        "config": {"repair_attempts": 2},
        "decision_model": None,
    },
    "cot": {
        "label": "cot",
        "source": COT_SOURCE,
        "config": None,
        "decision_model": None,
    },
    "multi_first": {
        "label": "multi_first",
        "source": JEV_SOURCE,
        "config": {"selection": "first"},
        "decision_model": None,
    },
    "multi_jev": {
        "label": "multi_jev",
        "source": JEV_SOURCE,
        "config": {"selection": "jev"},
        "decision_model": DECISION_MODEL,
    },
    "targeted_jev": {
        "label": "targeted_jev",
        "source": TARGETED_JEV_SOURCE,
        "config": {"selection": "jev"},
        "decision_model": DECISION_MODEL,
    },
}


def identity(problem: dict) -> str:
    value = problem["task_id"] + "\0" + problem["prompt"]
    return hashlib.sha256(value.encode()).hexdigest()


def load_dev_tasks() -> dict[str, dict]:
    # Preserve the official preprocessing and any oracle clarifications. The train
    # split carries no curated clarifications.jsonl entries, so the simulator falls
    # back to each task's reference_prompt as the hidden ground truth.
    tasks = {identity(p): p for p in load_split(str(ROOT / "data/splits/train.txt"))}
    clarification_file = ROOT / "data/clarifications.jsonl"
    if clarification_file.exists():
        clarifications = {
            (r["task_id"], r["prompt"]): r["clarifications"] for r in rows(clarification_file)
        }
        for p in tasks.values():
            key = (p["task_id"], p["prompt"])
            if key in clarifications:
                p["clarifications"] = clarifications[key]
    return tasks


def load_env_keys() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")


def prepare() -> tuple[list[tuple[str, dict]], dict]:
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    tasks = load_dev_tasks()
    records: list[tuple[str, dict]] = []
    for item in manifest["tasks"]:
        key = item["key"]
        if key not in tasks:
            raise ValueError("Dev-set task missing from the current training split.")
        records.append((key, tasks[key]))
    if digest(ORIGINAL_SOURCE.read_bytes()) != SOURCE_SHA:
        raise ValueError("Pinned first-place source changed; refusing to run arm 'targeted'.")
    for arm in ARMS.values():
        arm["class"] = _load_clarification_algorithm(str(arm["source"]))
        arm["sha256"] = digest(arm["source"].read_bytes())
    sources = OUTPUT / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    for arm in ARMS.values():
        (sources / arm["source"].name).write_bytes(arm["source"].read_bytes())
    meta = {
        "experiment": EXPERIMENT,
        "manifest_sha256": digest((OUTPUT / "manifest.json").read_bytes()),
        "decision_model": DECISION_MODEL,
        "arms": {
            name: {
                "algorithm": arm["class"].__name__,
                "source_sha256": arm["sha256"],
                "config": arm["config"],
                "decision_model": arm["decision_model"],
            }
            for name, arm in ARMS.items()
        },
    }
    save(OUTPUT / "arms.json", meta)
    return records, ARMS


def solve(item: tuple[str, dict], arm: dict, gateway) -> dict:
    key, task = item
    gateway.context = {
        "experiment": EXPERIMENT,
        "instance_id": key,
        "method": arm["label"],
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
            decision_model=arm["decision_model"],
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
            algorithm = arm["class"](arm["config"] or {})
            code = algorithm.run(env, {"prompt": task["prompt"], "entry_point": task["entry_point"]})
            selection = getattr(algorithm, "last_selection", None)
            repair_info = getattr(algorithm, "last_repair", None)
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
        "algorithm": arm["label"],
        "algorithm_path": str(arm["source"].relative_to(ROOT)),
        "model": MODEL,
        "prompt_result": code,
        "need_clarification": len(task.get("clarifications", [])) > 0,
        "clarification_history": env.history,
        "question_selection": selection,
        "repair_info": repair_info,
        "prompt_cost": env.prompt_cost,
        "clarification_cost": env.clarification_cost,
        "elapsed_seconds": time.monotonic() - started,
        "trace": trace,
    }


def run_arm(arm_name, arm, records, limit, workers, ledger, prices, failures) -> None:
    completed = {r["instance_id"] for r in rows(OUTPUT / f"{arm_name}_responses.jsonl")}
    pending = [item for item in records[:limit] if item[0] not in completed]
    if not pending:
        print(f"[{arm_name}] all {len(completed)} complete", flush=True)
        return
    print(
        f"[{arm_name}] running {len(pending)} tasks; cumulative ${ledger.charged:.6f}",
        flush=True,
    )
    with ConcurrentGateway(read_key(ROOT / ".env"), prices, ledger) as gateway:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            iterator = iter(pending)

            def submit_one():
                item = next(iterator, None)
                if item is not None and not gateway.stopped.is_set():
                    active[pool.submit(solve, item, arm, gateway)] = item[0]

            for _ in range(workers):
                submit_one()
            while active:
                finished, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
                if not finished:
                    print(
                        f"[{arm_name}] waiting {len(completed)}/{limit}; "
                        f"cumulative ${ledger.charged:.6f}",
                        flush=True,
                    )
                for future in finished:
                    active.pop(future)
                    result = future.result()
                    if result.get("interrupted"):
                        gateway.stopped.set()
                        failures.append(result["reason"])
                        append(OUTPUT / f"{arm_name}_interrupted.jsonl", result)
                        print("STOP:", result["reason"], flush=True)
                    else:
                        append(OUTPUT / f"{arm_name}_responses.jsonl", result)
                        completed.add(result["instance_id"])
                        print(
                            f"[{arm_name}] [{len(completed)}/{limit}] {result['task_id']} "
                            f"questions={len(result['clarification_history'])} "
                            f"cumulative=${ledger.charged:.6f}",
                            flush=True,
                        )
                    submit_one()


def execute(limit: int, workers: int) -> None:
    records, arms = prepare()
    ledger = cumulative_ledger()
    prices = fetch_prices()
    append(
        OUTPUT / "invocations.jsonl",
        {"time": time.time(), "limit": limit, "workers": workers, "prices": prices},
    )
    failures = []
    for arm_name, arm in arms.items():
        run_arm(arm_name, arm, records, limit, workers, ledger, prices, failures)
    report()
    if failures:
        raise RunStopped(failures[0])


def evaluate() -> None:
    records, arms = prepare()
    evaluator = init_evalplus_evaluator([task for _, task in records])
    for arm_name in ARMS:
        completed = {r["instance_id"] for r in rows(OUTPUT / f"{arm_name}_results.jsonl")}
        results = rows(OUTPUT / f"{arm_name}_responses.jsonl")
        for result in results:
            if result["instance_id"] in completed:
                continue
            code = result["prompt_result"]
            if "```python" not in code:
                code = f"```python\n{code.strip()}\n```"
            success, status = evaluator.partial(result).evaluate(None, code)
            append(
                OUTPUT / f"{arm_name}_results.jsonl",
                {**result, "success": success, "test_result": status},
            )
            completed.add(result["instance_id"])
            print(
                f"[{arm_name}] [{len(completed)}/{len(results)}] {result['task_id']} "
                f"{'PASS' if success else 'FAIL'}",
                flush=True,
            )
    report()


def report() -> None:
    summary: dict = {"experiment": EXPERIMENT}
    for arm_name in ARMS:
        generated = rows(OUTPUT / f"{arm_name}_responses.jsonl")
        evaluated = rows(OUTPUT / f"{arm_name}_results.jsonl")
        arm_summary = {"generated": len(generated), "evaluated": len(evaluated)}
        if evaluated:
            questions = [h for r in evaluated for h in r["clarification_history"]]
            arm_summary.update(
                {
                    "count": len(evaluated),
                    "passed": sum(r["success"] for r in evaluated),
                    "pass_at_1": sum(r["success"] for r in evaluated) / len(evaluated),
                    "tds": _turn_discounted_sucess(evaluated),
                    "ndcg": _normalized_discounted_cumulative_gain(evaluated),
                    "asked_tasks": sum(bool(r["clarification_history"]) for r in evaluated),
                    "quality_3_questions": sum(h[2] == "3" for h in questions),
                    "prompt_cost_usd": sum(r.get("prompt_cost", 0.0) for r in evaluated),
                }
            )
            arm_summary["by_dataset"] = {}
            for family in ("HumanEval", "Mbpp"):
                subset = [r for r in evaluated if r["task_id"].startswith(family + "/")]
                arm_summary["by_dataset"][family] = {
                    "n": len(subset),
                    "passed": sum(r["success"] for r in subset),
                }
        summary[arm_name] = arm_summary
    ledger = Ledger(SHARED_LEDGER, BUDGET) if SHARED_LEDGER.exists() else Ledger(PRIOR_LEDGER, BUDGET)
    summary["cumulative_conservative_usd"] = ledger.charged
    summary["budget_usd"] = BUDGET
    summary["notes"] = (
        "Dev set drawn from the train split; hidden requirements come from reference_prompt, "
        "not the curated clarifications.jsonl used by validation. Directional probe, one sample "
        "per task; not a validation leaderboard result."
    )
    save(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    load_env_keys()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "evaluate", "report"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100 or not 1 <= args.workers <= 4:
        parser.error("limit must be 1..100 and workers 1..4")
    if args.command == "run":
        with (ROOT / "runs/zenmux_cumulative.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                execute(args.limit, args.workers)
            except RunStopped as exc:
                print("Experiment stopped:", str(exc), flush=True)
                raise SystemExit(2) from None
    elif args.command == "prepare":
        records, arms = prepare()
        print(f"Prepared {len(records)} dev tasks across {len(arms)} arms.")
        for name, arm in arms.items():
            print(f"  {name}: {arm['class'].__name__} (sha256 {arm['sha256'][:16]}…)")
    elif args.command == "evaluate":
        evaluate()
    else:
        report()


if __name__ == "__main__":
    main()
