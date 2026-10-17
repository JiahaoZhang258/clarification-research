import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import litellm

from experiments.reproduce_zarutskiy import ConcurrentGateway, LockedLedger, instance_records
from experiments.zenmux import MODEL, RunStopped


class ReproductionTests(unittest.TestCase):
    def test_duplicate_instances_survive_stable_ordering(self):
        a = {"task_id": "Mbpp/1", "prompt": "A"}
        b = {"task_id": "HumanEval/1", "prompt": "B"}
        first = instance_records([a, b, a.copy()])
        second = instance_records([a.copy(), a, b])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({key for key, _ in first}), 3)
        self.assertEqual(sum(task == a for _, task in first), 2)

    def test_concurrent_reservations_cannot_overspend(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = LockedLedger(Path(folder) / "usage.jsonl", 2)
            ledger.reserve(1.9, {"phase": "prior"})
            barrier = threading.Barrier(4)

            def reserve(_):
                barrier.wait()
                try:
                    ledger.reserve(0.06, {"phase": "new"})
                    return True
                except RunStopped:
                    return False

            with ThreadPoolExecutor(max_workers=4) as pool:
                self.assertEqual(sum(pool.map(reserve, range(4))), 1)
            self.assertAlmostEqual(ledger.charged, 1.96)
            self.assertAlmostEqual(LockedLedger(ledger.path, 2).charged, 1.96)

    def test_parallel_calls_keep_task_and_phase_attribution(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = LockedLedger(Path(folder) / "usage.jsonl", 2)
            barrier = threading.Barrier(4)

            def transport(**request):
                barrier.wait()
                return SimpleNamespace(
                    id=request["messages"][0]["content"],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10, cost=0.000056),
                    choices=[SimpleNamespace(finish_reason="stop")],
                )

            prices = {"prompt_per_token": 0.4e-6, "completion_per_token": 1.6e-6}
            with patch.object(litellm, "completion", side_effect=transport):
                with ConcurrentGateway("TEST-SECRET", prices, ledger) as gateway:

                    def call(index):
                        gateway.context = {"instance_id": str(index), "phase": f"phase-{index}"}
                        gateway.complete(
                            model=MODEL, messages=[{"role": "user", "content": str(index)}]
                        )

                    with ThreadPoolExecutor(max_workers=4) as pool:
                        list(pool.map(call, range(4)))
            self.assertEqual(len(ledger.latest), 4)
            for record in ledger.latest.values():
                self.assertEqual(record["generation_id"], record["instance_id"])
                self.assertEqual(record["phase"], "phase-" + record["instance_id"])
            self.assertNotIn("TEST-SECRET", json.dumps(ledger.latest))


if __name__ == "__main__":
    unittest.main()
