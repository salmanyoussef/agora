"""Cumulative spend tracker with a hard ceiling (public-demo circuit breaker).

Builds on the existing per-pipeline cost estimation (see dspy_setup.format_pipeline_cost_usd):
after each pipeline run, the estimated USD cost is added to a persistent counter.
Once the counter reaches AGORA_COST_CEILING_USD, new search requests are rejected
gracefully until the ceiling is raised or the counter file is reset.

State is a small JSON file (AGORA_BUDGET_FILE) so the total survives restarts when
the file lives on a persistent disk. Thread-safe within one process; the demo runs
a single uvicorn worker, so this is sufficient.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Charged when a run's cost cannot be estimated (unknown model pricing), so
# unpriced runs still consume budget instead of being free.
FALLBACK_COST_PER_RUN_USD = 0.02


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


class BudgetTracker:
    def __init__(self, ceiling_usd: float | None = None, path: str | None = None):
        self.ceiling_usd = ceiling_usd if ceiling_usd is not None else _env_float(
            "AGORA_COST_CEILING_USD", 25.0
        )
        self.path = Path(path or os.environ.get("AGORA_BUDGET_FILE", "agora_budget.json"))
        self._lock = threading.Lock()
        self._spent_usd = 0.0
        self._runs = 0
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._spent_usd = float(data.get("spent_usd", 0.0))
                self._runs = int(data.get("runs", 0))
                logger.info(
                    "Budget tracker loaded: spent=%.4f USD over %d runs (ceiling %.2f USD)",
                    self._spent_usd, self._runs, self.ceiling_usd,
                )
        except Exception:
            logger.exception("Budget tracker: failed to load %s; starting at 0", self.path)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "spent_usd": self._spent_usd,
                        "runs": self._runs,
                        "ceiling_usd": self.ceiling_usd,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                ),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except Exception:
            logger.exception("Budget tracker: failed to persist %s", self.path)

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def exhausted(self) -> bool:
        return self._spent_usd >= self.ceiling_usd

    def record_run(self, cost_usd: float | str | None) -> None:
        """Add one pipeline run's estimated cost (falls back to a flat charge if unknown)."""
        try:
            cost = float(cost_usd) if cost_usd is not None else FALLBACK_COST_PER_RUN_USD
        except (TypeError, ValueError):
            cost = FALLBACK_COST_PER_RUN_USD
        if cost <= 0:
            cost = FALLBACK_COST_PER_RUN_USD
        with self._lock:
            self._spent_usd += cost
            self._runs += 1
            self._save()
        logger.info(
            "Budget: +%.6f USD (run %d) → total %.4f / %.2f USD",
            cost, self._runs, self._spent_usd, self.ceiling_usd,
        )
        if self.exhausted:
            logger.warning(
                "Budget CEILING REACHED: %.4f / %.2f USD — new requests will be rejected",
                self._spent_usd, self.ceiling_usd,
            )

    def status(self) -> dict:
        return {
            "spent_usd": round(self._spent_usd, 6),
            "ceiling_usd": self.ceiling_usd,
            "runs": self._runs,
            "exhausted": self.exhausted,
        }


budget_tracker = BudgetTracker()

EXHAUSTED_MESSAGE = (
    "This public demo has reached its usage budget for the review period. "
    "Please contact the authors to have it restored. / "
    "Cette démo publique a atteint son budget d'utilisation. "
    "Merci de contacter les auteurs pour la réactiver."
)
