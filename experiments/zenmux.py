"""Local LiteLLM routing and a durable, conservative spending ledger."""

from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import litellm

API_BASE = "https://zenmux.ai/api/v1"
MODEL = "openai/gpt-4.1-mini"
MAX_OUTPUT_TOKENS = 8192


class RunStopped(BaseException):
    """Stop the whole run, including inside the SDK's broad oracle exception handler."""


def read_key(path: Path) -> str:
    key = os.environ.get("ZENMUX_API_KEY", "")
    if not key and path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("ZENMUX_API_KEY="):
                key = line.partition("=")[2].strip().strip("\"'")
                break
    if not key:
        raise ValueError("Set ZENMUX_API_KEY or add it to the ignored .env file.")
    return key


def fetch_prices() -> dict:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(API_BASE + "/models", timeout=20) as response:
                models = json.load(response)["data"]
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise RunStopped(
                    "Could not retrieve ZenMux prices after three read-only attempts."
                ) from None
            time.sleep(attempt + 1)
    model = next(m for m in models if m["id"] == MODEL)
    result = {"model": MODEL, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    for name in ("prompt", "completion"):
        entries = model["pricings"][name]
        if (
            len(entries) != 1
            or entries[0]["unit"] != "perMTokens"
            or entries[0]["currency"] != "USD"
        ):
            raise ValueError("Unsupported ZenMux pricing; refusing to guess a price.")
        result[name + "_per_token"] = float(entries[0]["value"]) / 1_000_000
    result["source"] = API_BASE + "/models"
    return result


class Ledger:
    def __init__(self, path: Path, budget: float):
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("Budget must be a positive finite USD amount.")
        self.path = path
        self.budget = budget
        self.latest = {}
        if path.exists():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                self.latest[record["call_id"]] = record

    @property
    def charged(self) -> float:
        return sum(record["reserved_or_estimated_usd"] for record in self.latest.values())

    def append(self, record: dict) -> None:
        record = {**record, "recorded_at": datetime.now(timezone.utc).isoformat()}
        with self.path.open("a") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        self.latest[record["call_id"]] = record

    def reserve(self, amount: float, context: dict) -> dict:
        if self.charged + amount > self.budget:
            raise RunStopped("Budget guard stopped before another API request.")
        record = {
            "call_id": uuid.uuid4().hex,
            "status": "reserved",
            "reserved_or_estimated_usd": amount,
            **context,
        }
        self.append(record)
        return record


class ZenMuxGateway:
    """Redirect SDK calls locally; algorithm files remain independent of ZenMux."""

    def __init__(self, key: str, prices: dict, ledger: Ledger):
        self.key = key
        self.prices = prices
        self.ledger = ledger
        self.context = {"phase": "connectivity"}
        self.response_costs = {}

    def __enter__(self):
        self.original_completion = litellm.completion
        self.original_cost = litellm.completion_cost
        self.original_suppress = litellm.suppress_debug_info
        litellm.suppress_debug_info = True
        self.log_levels = {}
        for name in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "httpx", "openai"):
            logger = logging.getLogger(name)
            self.log_levels[name] = logger.level
            logger.setLevel(logging.CRITICAL)
        litellm.completion = self.complete
        litellm.completion_cost = self.cost
        return self

    def __exit__(self, *exc):
        litellm.completion = self.original_completion
        litellm.completion_cost = self.original_cost
        litellm.suppress_debug_info = self.original_suppress
        for name, level in self.log_levels.items():
            logging.getLogger(name).setLevel(level)

    def cost(self, *args, **kwargs):
        response = kwargs.get("completion_response")
        if response is not None and id(response) in self.response_costs:
            return self.response_costs[id(response)]
        return self.original_cost(*args, **kwargs)

    def complete(self, *args, **kwargs):
        if args or kwargs.get("model") != MODEL:
            raise RunStopped("Unexpected model or invocation in this fixed-model experiment.")
        messages = kwargs["messages"]
        # Text-only requests: UTF-8 byte length bounds byte-level BPE tokens.
        # Add a conservative allowance for protocol framing; reserve full output.
        encoded = json.dumps(messages, ensure_ascii=False).encode("utf-8")
        if any(not isinstance(m.get("content"), str) for m in messages):
            raise RunStopped("The budget guard supports text messages only.")
        prompt_bound = len(encoded) + 4096 + 32 * len(messages)
        output_bound = min(kwargs.get("max_tokens") or MAX_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
        reserve = (
            prompt_bound * self.prices["prompt_per_token"]
            + output_bound * self.prices["completion_per_token"]
        )
        record = self.ledger.reserve(
            reserve,
            {
                **self.context,
                "model": MODEL,
                "prompt_token_bound": prompt_bound,
                "output_token_bound": output_bound,
            },
        )
        request = {
            **kwargs,
            # The first prefix selects LiteLLM's compatible provider; the second
            # belongs to the actual ZenMux model slug sent over the wire.
            "model": "openai/" + MODEL,
            "api_base": API_BASE,
            "api_key": self.key,
            "max_tokens": output_bound,
            "num_retries": 0,
            "max_retries": 0,
            "timeout": 120,
        }
        try:
            response = self.original_completion(**request)
        except Exception as exc:
            self.ledger.append(
                {**record, "status": "request_error", "error_type": type(exc).__name__}
            )
            # No provider exception text: it can contain request credentials.
            raise RunStopped("API request failed: " + type(exc).__name__) from None
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            self.ledger.append(
                {
                    **record,
                    "status": "missing_usage",
                    "generation_id": getattr(response, "id", None),
                }
            )
            raise RunStopped("Missing token usage; retained reservation and stopped.")
        estimate = (
            prompt_tokens * self.prices["prompt_per_token"]
            + completion_tokens * self.prices["completion_per_token"]
        )
        reported_cost = getattr(usage, "cost", None)
        try:
            reported_cost = float(reported_cost)
        except (TypeError, ValueError):
            reported_cost = None
        if reported_cost is not None and (not math.isfinite(reported_cost) or reported_cost < 0):
            reported_cost = None
        conservative_cost = max(estimate, reported_cost or 0.0)
        self.ledger.append(
            {
                **record,
                "status": "completed",
                "reserved_or_estimated_usd": conservative_cost,
                "provider_reported_cost_usd": reported_cost,
                "public_price_estimate_usd": estimate,
                "generation_id": getattr(response, "id", None),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "cost_basis": "ZenMux public prices; cache discounts not subtracted",
            }
        )
        if (
            prompt_tokens > prompt_bound
            or completion_tokens > output_bound
            or conservative_cost > reserve
        ):
            raise RunStopped("Provider usage exceeded the reservation bound; stopped.")
        self.response_costs[id(response)] = conservative_cost
        return response
