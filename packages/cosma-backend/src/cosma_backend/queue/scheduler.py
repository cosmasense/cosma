"""
Scheduler

Rules engine that evaluates system conditions (battery, CPU, time, etc.)
and pauses/resumes the indexing queue accordingly.
"""

from __future__ import annotations

import asyncio
import datetime
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

from cosma_backend.logging import get_logger
from cosma_backend.queue.metrics import SystemMetricsCollector

if TYPE_CHECKING:
    from cosma_backend.queue.indexing_queue import IndexingQueue
    from cosma_backend.settings import SchedulerConfig, SchedulerRuleConfig
    from cosma_backend.utils.pubsub import Hub

logger = get_logger(__name__)


class Scheduler:
    """
    Periodically collects system metrics, evaluates rules, and
    pauses/resumes the indexing queue when conditions change.
    """

    def __init__(
        self,
        queue: IndexingQueue,
        updates_hub: Optional[Hub] = None,
        config: Optional[SchedulerConfig] = None,
    ):
        from cosma_backend.settings import SchedulerConfig as _SC

        self._queue = queue
        self._updates_hub = updates_hub
        self._config = config or _SC()
        self._collector = SystemMetricsCollector()
        self._task: Optional[asyncio.Task] = None
        self._conditions_met = True
        self._last_metrics: dict[str, Any] = {}

    @property
    def config(self) -> "SchedulerConfig":
        return self._config

    @property
    def conditions_met(self) -> bool:
        return self._conditions_met

    @property
    def last_metrics(self) -> dict[str, Any]:
        return self._last_metrics

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._monitor_loop())
            logger.info("Scheduler started",
                        enabled=self._config.enabled,
                        interval=self._config.check_interval_seconds)

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("Scheduler stopped")

    # ------------------------------------------------------------------
    # Config updates
    # ------------------------------------------------------------------

    def update_config(self, data: dict[str, Any]) -> None:
        """Update scheduler config from a dict (e.g. from the API)."""
        from cosma_backend.settings import SchedulerRuleConfig

        if "enabled" in data:
            self._config.enabled = bool(data["enabled"])
        if "combine_mode" in data:
            self._config.combine_mode = str(data["combine_mode"]).upper()
        if "check_interval_seconds" in data:
            self._config.check_interval_seconds = int(data["check_interval_seconds"])
        if "rules" in data and isinstance(data["rules"], list):
            self._config.rules = []
            for r in data["rules"]:
                self._config.rules.append(SchedulerRuleConfig(
                    rule=r.get("rule", ""),
                    operator=r.get("operator", ""),
                    value=r.get("value"),
                    enabled=r.get("enabled", True),
                ))

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        while True:
            try:
                if self._config.enabled and self._config.rules:
                    metrics = await self._collector.collect()
                    self._last_metrics = metrics

                    met = self._evaluate_rules(metrics)

                    if met != self._conditions_met:
                        self._conditions_met = met
                        if met:
                            self._queue.scheduler_resume()
                        else:
                            self._queue.scheduler_pause()
                else:
                    # Scheduler disabled — make sure queue isn't scheduler-paused
                    if not self._conditions_met:
                        self._conditions_met = True
                        self._queue.scheduler_resume()

                await asyncio.sleep(self._config.check_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in scheduler monitor loop")
                await asyncio.sleep(self._config.check_interval_seconds)

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rules(self, metrics: dict[str, Any]) -> bool:
        """Evaluate all enabled rules. Returns True if conditions allow processing."""
        enabled_rules = [r for r in self._config.rules if r.enabled]
        if not enabled_rules:
            return True

        results = [self._evaluate_rule(r, metrics) for r in enabled_rules]

        if self._config.combine_mode == "ANY":
            return any(results)
        # Default: ALL
        return all(results)

    def _evaluate_rule(self, rule: "SchedulerRuleConfig", metrics: dict[str, Any]) -> bool:
        """Evaluate a single rule against collected metrics. Returns True if the rule passes."""
        metric_key = rule.rule
        operator = rule.operator
        value = rule.value

        if metric_key == "time_window":
            return self._evaluate_time_window(value)

        if metric_key == "power_source":
            actual = metrics.get("power_source_plugged")
            if actual is None:
                logger.debug("power_source metric unavailable, rule passes by default")
                return True
            expected = _to_bool(value)
            return actual == expected

        # Numeric comparisons
        metric_map = {
            "battery_level": "battery_level",
            "cpu_idle": "cpu_idle_seconds",
            "cpu_temperature": "cpu_temperature",
            "fan_speed": "fan_speed",
        }

        actual = metrics.get(metric_map.get(metric_key, metric_key))
        if actual is None:
            logger.debug("Metric unavailable, rule passes by default", metric=metric_key)
            return True

        try:
            threshold = float(value)
        except (TypeError, ValueError):
            logger.warning("Invalid rule value", rule=metric_key, value=value)
            return True

        if operator == "gt":
            return actual > threshold
        if operator == "lt":
            return actual < threshold
        if operator == "eq":
            return actual == threshold
        if operator == "gte":
            return actual >= threshold
        if operator == "lte":
            return actual <= threshold

        logger.warning("Unknown operator", operator=operator, rule=metric_key)
        return True

    @staticmethod
    def _evaluate_time_window(value: Any) -> bool:
        """Check if current time is within [start, end] window.

        ``value`` should be a list of two strings like ``["02:00", "04:00"]``.
        """
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return True

        try:
            now = datetime.datetime.now().time()
            start = datetime.time.fromisoformat(value[0])
            end = datetime.time.fromisoformat(value[1])

            if start <= end:
                return start <= now <= end
            # Wraps midnight, e.g. 23:00 → 02:00
            return now >= start or now <= end
        except (ValueError, TypeError):
            return True


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)
