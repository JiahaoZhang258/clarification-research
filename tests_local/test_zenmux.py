import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import litellm

from experiments.zenmux import API_BASE, MODEL, Ledger, RunStopped, ZenMuxGateway

PRICES = {"prompt_per_token": 0.4 / 1_000_000, "completion_per_token": 1.6 / 1_000_000}


def response(prompt=100, completion=10):
    return SimpleNamespace(
        id="test-generation",
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="OK"))],
    )


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "usage.jsonl"

    def call(self, gateway):
        return gateway.complete(model=MODEL, messages=[{"role": "user", "content": "Hello"}])

    def test_routing_cost_and_secret_redaction(self):
        ledger = Ledger(self.path, 2)
        with patch.object(litellm, "completion", return_value=response()) as transport:
            with ZenMuxGateway("TEST-SECRET", PRICES, ledger) as gateway:
                result = self.call(gateway)
                self.assertAlmostEqual(
                    litellm.completion_cost(completion_response=result), 0.000056
                )
            request = transport.call_args.kwargs
            self.assertEqual(request["model"], "openai/" + MODEL)
            self.assertEqual(request["api_base"], API_BASE)
            self.assertEqual(request["api_key"], "TEST-SECRET")
            self.assertEqual(request["num_retries"], 0)
            self.assertEqual(request["max_retries"], 0)
        self.assertNotIn("TEST-SECRET", self.path.read_text())
        self.assertAlmostEqual(Ledger(self.path, 2).charged, 0.000056)

    def test_budget_stops_before_network(self):
        ledger = Ledger(self.path, 0.000001)
        with patch.object(litellm, "completion") as transport:
            with ZenMuxGateway("unused", PRICES, ledger) as gateway:
                with self.assertRaises(RunStopped):
                    self.call(gateway)
            transport.assert_not_called()
        self.assertEqual(ledger.charged, 0)

    def test_unknown_failed_cost_retains_reservation_on_resume(self):
        ledger = Ledger(self.path, 2)
        with patch.object(litellm, "completion", side_effect=RuntimeError("TEST-SECRET")):
            with ZenMuxGateway("TEST-SECRET", PRICES, ledger) as gateway:
                with self.assertRaisesRegex(RunStopped, "API request failed: RuntimeError"):
                    self.call(gateway)
        self.assertGreater(ledger.charged, 0)
        self.assertEqual(Ledger(self.path, 2).charged, ledger.charged)
        self.assertNotIn("TEST-SECRET", self.path.read_text())

    def test_missing_usage_is_not_free(self):
        result = response()
        result.usage = None
        ledger = Ledger(self.path, 2)
        with patch.object(litellm, "completion", return_value=result):
            with ZenMuxGateway("unused", PRICES, ledger) as gateway:
                with self.assertRaisesRegex(RunStopped, "Missing token usage"):
                    self.call(gateway)
        self.assertGreater(ledger.charged, 0)
        self.assertEqual(next(iter(ledger.latest.values()))["status"], "missing_usage")

    def test_reported_price_is_recorded_and_not_underestimated(self):
        result = response()
        result.usage.cost = "0.0001"
        ledger = Ledger(self.path, 2)
        with patch.object(litellm, "completion", return_value=result):
            with ZenMuxGateway("unused", PRICES, ledger) as gateway:
                self.call(gateway)
        self.assertEqual(ledger.charged, 0.0001)
        self.assertEqual(next(iter(ledger.latest.values()))["provider_reported_cost_usd"], 0.0001)

    def test_resumed_calls_share_total_cap(self):
        ledger = Ledger(self.path, 2)
        ledger.reserve(1.99, {"phase": "interrupted"})
        resumed = Ledger(self.path, 2)
        with patch.object(litellm, "completion") as transport:
            with ZenMuxGateway("unused", PRICES, resumed) as gateway:
                with self.assertRaises(RunStopped):
                    self.call(gateway)
            transport.assert_not_called()

    def test_stop_bypasses_sdk_exception_fallback(self):
        # An oracle request must stop the experiment on transport/budget failure;
        # the SDK catches ordinary Exceptions and otherwise invents a refusal.
        self.assertFalse(issubclass(RunStopped, Exception))


if __name__ == "__main__":
    unittest.main()
