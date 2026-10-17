"""Export compact, auditable research results from private local runs/ files.

No API calls. No responses, generated code, hidden tests, keys or billing ledger
are exported. Run this only in the original workspace containing raw artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def read(path):
    return json.loads((ROOT / path).read_text())


def rows(path):
    return [json.loads(line) for line in (ROOT / path).read_text().splitlines() if line]


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    OUT.mkdir(exist_ok=True)
    specs = [
        ("baseline", "runs/zarutskiy_val_ae88217", "full_pipeline"),
        ("original_cot", "runs/cot_val", "full_pipeline"),
        ("fewshot_cot", "runs/cot_fewshot_val", "full_pipeline"),
        ("draft_patch", "runs/patch_val", "full_pipeline"),
        ("distilled_initial", "runs/prompt_distillation_val_reuse_v1", "fixed_historical_qa"),
        *[(f"distilled_repeat_{n}", f"runs/prompt_distillation_val_repeats_v1/repeat_{n}",
           "fixed_historical_qa") for n in (1, 2, 3)],
    ]
    tables, groups, provenance = [], {}, []
    for name, directory, scope in specs:
        path = f"{directory}/results.jsonl"
        records = rows(path)
        assert len(records) == len({r["instance_id"] for r in records}) == 770
        groups[name] = {r["instance_id"]: r for r in records}
        summary = read(f"{directory}/summary.json")
        metrics = summary["metrics"]
        if scope == "fixed_historical_qa":
            metrics = metrics["distilled"]
        table = {"id": name, "scope": scope, "count": 770,
                 "passed": sum(bool(r["success"]) for r in records),
                 "pass_at_1": metrics["pass_at_1"], "tds": metrics["tds"],
                 "ndcg": metrics["ndcg"], "clarification_rate": metrics["clarification_rate"],
                 "overasking_rate": sum(bool(r["clarification_history"]) for r in records
                                        if not r["need_clarification"]) /
                                    sum(not r["need_clarification"] for r in records)}
        if scope == "full_pipeline":
            table["average_solver_cost_usd"] = metrics["avg_cost_usd"]
        else:
            table["average_final_generation_cost_usd"] = metrics["avg_final_generation_cost_usd"]
        tables.append(table)
        provenance.append({"id": name, "local_source": path,
                           "raw_results_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()})
    ids = sorted(groups["original_cot"])
    assert all(set(group) == set(ids) for group in groups.values())
    with (OUT / "validation_outcomes.jsonl").open("w") as handle:
        for key in ids:
            compact = {"instance_id": key,
                       "need_clarification": groups["original_cot"][key]["need_clarification"],
                       "passed": {}, "asked": {}, "quality_3": {}}
            for name, group in groups.items():
                row = group[key]
                assert len(row["clarification_history"]) == 1
                compact["passed"][name] = bool(row["success"])
                compact["asked"][name] = True
                compact["quality_3"][name] = row["clarification_history"][0][2] == "3"
            handle.write(json.dumps(compact, separators=(",", ":")) + "\n")
    write("validation.json", {"model": "openai/gpt-4.1-mini", "split": "validation",
                              "organizer_confirmed": False, "experiments": tables,
                              "limitations": ["Local results, not official confirmed scores.",
                                               "Distilled runs reuse original CoT QA and one historical baseline.",
                                               "Do not pool repeats as independent tasks.",
                                               "Solver cost and final-generation-only cost are different scopes."]})
    stability = read("runs/prompt_distillation_val_repeats_v1/stability_summary.json")
    # Publication results omit private billing and API/account operational state.
    write("distillation_stability.json", {k: v for k, v in stability.items()
                                          if "cost" not in k})
    training = read("runs/contrast_questions_train_v1/summary.json")
    generated = len(rows("runs/contrast_questions_train_v1/responses.jsonl"))
    evaluated = sum(r[a]["evaluated"] for r in training["rounds"] for a in ("baseline", "contrast"))
    write("contrast_questions_partial.json", {
        "complete": training["complete"], "generated": generated, "evaluated": evaluated,
        "target": 600, "status": "pending_analysis" if training["complete"] else "incomplete_provider_access_unavailable",
        "rounds": training["rounds"], "limitations": training["limitations"],
        "conclusion": "Pending analysis." if training["complete"] else "No efficacy conclusion: groups are incomplete and unbalanced.",
    })
    train_distilled = read("runs/prompt_distillation_v1/summary.json")
    write("training_distillation.json", {k: train_distilled[k] for k in (
        "complete", "paired_records", "metrics", "rescued", "regressed", "net_gain",
        "exact_mcnemar_p", "limitations")})
    dev = read("runs/devset100/summary.json")
    write("training_variants.json", {k: v for k, v in dev.items()
                                      if k not in ("cumulative_conservative_usd", "budget_usd")})
    capacity = read("runs/cot_capacity_50x5/summary.json")
    write("capacity_probe.json", {"complete": capacity["complete"],
                                 "evaluated": capacity["evaluated"],
                                 "successful_generations": capacity["successful_generations"],
                                 "recovered_instances": capacity["recovered_instances"],
                                 "original_instances": 50, "samples_per_instance": 5,
                                 "scope": "Previously failed validation instances with frozen QA; not a general pass-rate estimate."})
    write("provenance.json", {"exported_on": "2026-10-17", "sources": provenance,
                              "outcomes_sha256": hashlib.sha256((OUT / "validation_outcomes.jsonl").read_bytes()).hexdigest(),
                              "excluded": ["raw responses", "generated code", "hidden tests",
                                           "API request/response logs", "keys", "billing ledgers"]})
    print(f"Exported {len(tables)} validation rows and {len(ids)} compact per-instance outcomes.")


if __name__ == "__main__":
    main()
