"""Check public research artifacts without API calls or private runs/ files."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results"
    summary = json.loads((directory / "validation.json").read_text())
    outcomes = [json.loads(line) for line in (directory / "validation_outcomes.jsonl").read_text().splitlines()]
    assert len(outcomes) == len({r["instance_id"] for r in outcomes}) == 770
    expected_ids = {e["id"] for e in summary["experiments"]}
    for row in outcomes:
        assert set(row) == {"instance_id", "need_clarification", "passed", "asked", "quality_3"}
        assert all(set(row[field]) == expected_ids for field in ("passed", "asked", "quality_3"))
        assert all(type(v) is bool for field in ("passed", "asked", "quality_3") for v in row[field].values())
    for experiment in summary["experiments"]:
        key = experiment["id"]
        passed = sum(r["passed"][key] for r in outcomes)
        assert passed == experiment["passed"]
        rate = passed / 770
        assert math.isclose(rate, experiment["pass_at_1"])
        tds = sum(r["passed"][key] * math.log(10) / math.log(10 + r["asked"][key]) for r in outcomes) / 770
        assert math.isclose(tds, experiment["tds"])
        assert math.isclose(sum(r["quality_3"][key] for r in outcomes) / 770, experiment["ndcg"])
        assert math.isclose(sum(r["asked"][key] for r in outcomes) / 770, experiment["clarification_rate"])
        unnecessary = [r for r in outcomes if not r["need_clarification"]]
        assert math.isclose(sum(r["asked"][key] for r in unnecessary) / len(unnecessary), experiment["overasking_rate"])
    provenance = json.loads((directory / "provenance.json").read_text())
    assert hashlib.sha256((directory / "validation_outcomes.jsonl").read_bytes()).hexdigest() == provenance["outcomes_sha256"]
    stability = json.loads((directory / "distillation_stability.json").read_text())
    counts = [sum(r["passed"][key] for r in outcomes) for key in (
        "distilled_initial", "distilled_repeat_1", "distilled_repeat_2", "distilled_repeat_3")]
    assert counts == [r["passed"] for r in stability["rounds"]]
    assert statistics.mean(counts) == stability["mean_passed"]
    for name, digest in (
        ("original_cot", "aa2fe014bd54c58f4e7862519a166dc50215e054bc710493ebb1aca281637e63"),
        ("distilled_v1", "b28cee805aade6b0fba37a4bd5b00427aed07454b67430528dd84d5a6b6b270a"),
    ):
        content = (ROOT / f"reproducibility/frozen_prompts/{name}.txt").read_text().strip()
        assert hashlib.sha256(content.encode()).hexdigest() == digest
    partial = json.loads((directory / "contrast_questions_partial.json").read_text())
    assert partial["evaluated"] == sum(r[a]["evaluated"] for r in partial["rounds"] for a in ("baseline", "contrast"))
    assert partial["complete"] == (partial["evaluated"] == partial["target"])
    print(f"Verified {len(expected_ids)} validation experiments, 770 outcomes, stability statistics, frozen prompts and partial-experiment counts.")


if __name__ == "__main__":
    main()
