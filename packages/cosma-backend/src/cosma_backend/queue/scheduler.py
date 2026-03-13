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
        self._last_rule_results: list[dict[str, Any]] = []
        self._warnings: list[str] = []
        self._config_lock = asyncio.Lock()

    @property
    def config(self) -> "SchedulerConfig":
        return self._config

    @property
    def conditions_met(self) -> bool:
        return self._conditions_met

    @property
    def last_metrics(self) -> dict[str, Any]:
        return self._last_metrics

    @property
    def last_rule_results(self) -> list[dict[str, Any]]:
        return self._last_rule_results

    @property
    def warnings(self) -> list[str]:
        return self._warnings

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

    async def update_config(self, data: dict[str, Any]) -> None:
        """Update scheduler config from a dict (e.g. from the API)."""
        from cosma_backend.settings import SchedulerRuleConfig

        async with self._config_lock:
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

            self.validate_rules()

    def validate_rules(self) -> list[str]:
        """Detect contradictory or tautological rule configurations.

        Returns a list of human-readable warning strings.
        """
        warnings: list[str] = []
        enabled_rules = [
            r for r in self._config.rules
            if (r.enabled if hasattr(r, "enabled") else r.get("enabled", True))
        ]
        if not enabled_rules:
            return warnings

        # Group numeric rules by metric key
        numeric_groups: dict[str, list[tuple[str, float]]] = {}
        for r in enabled_rules:
            key = r.rule if hasattr(r, "rule") else r.get("rule", "")
            op = r.operator if hasattr(r, "operator") else r.get("operator", "")
            val = r.value if hasattr(r, "value") else r.get("value")
            if key in ("time_window", "power_source", "cpu_idle", "low_power_mode"):
                continue
            try:
                numeric_groups.setdefault(key, []).append((op, float(val)))
            except (TypeError, ValueError):
                continue

        mode = self._config.combine_mode

        for key, ops in numeric_groups.items():
            if len(ops) < 2:
                continue

            # In ALL mode, check for impossible combinations
            if mode == "ALL":
                lower_bounds = []  # gte/gt thresholds
                upper_bounds = []  # lte/lt thresholds
                for op, val in ops:
                    if op in ("gte", "gt"):
                        lower_bounds.append((op, val))
                    elif op in ("lte", "lt"):
                        upper_bounds.append((op, val))

                if lower_bounds and upper_bounds:
                    max_lower = max(v for _, v in lower_bounds)
                    min_upper = min(v for _, v in upper_bounds)
                    if max_lower > min_upper:
                        warnings.append(
                            f"Contradictory rules on '{key}': requires "
                            f">= {max_lower} AND <= {min_upper} (impossible, queue will never start)"
                        )
                    elif max_lower == min_upper:
                        lower_strict = any(op == "gt" for op, v in lower_bounds if v == max_lower)
                        upper_strict = any(op == "lt" for op, v in upper_bounds if v == min_upper)
                        if lower_strict or upper_strict:
                            warnings.append(
                                f"Contradictory rules on '{key}': strict inequality at "
                                f"{max_lower} makes the range empty (queue will never start)"
                            )

            # In ANY mode, check for tautological rules
            if mode == "ANY":
                for op, val in ops:
                    if op == "gte" and val <= 0 and key in (
                        "battery_level", "gpu_usage", "memory_usage", "queue_size",
                    ):
                        warnings.append(
                            f"Tautological rule on '{key}': >= {val} is always true "
                            f"(queue will never be paused by this rule)"
                        )
                    if op == "lte" and val >= 100 and key in (
                        "battery_level", "gpu_usage", "memory_usage",
                    ):
                        warnings.append(
                            f"Tautological rule on '{key}': <= {val} is always true "
                            f"(queue will never be paused by this rule)"
                        )

        # Duplicate rule types
        seen_keys: dict[str, int] = {}
        for r in enabled_rules:
            key = r.rule if hasattr(r, "rule") else r.get("rule", "")
            seen_keys[key] = seen_keys.get(key, 0) + 1
        for key, count in seen_keys.items():
            if count > 1:
                warnings.append(
                    f"Duplicate rule type '{key}' ({count} instances) - "
                    f"this may cause unexpected behaviour"
                )

        self._warnings = warnings
        return warnings

    async def test_rules(self) -> dict[str, Any]:
        """Evaluate rules against live metrics (dry-run).

        Returns conditions_met, rule_results, and the collected metrics.
        Note: this *does* update last_rule_results as a side-effect, matching
        the behaviour of the normal monitor loop.
        """
        metrics = await self._collector.collect()

        async with self._config_lock:
            conditions_met = self._evaluate_rules(metrics)
            rule_results = self._last_rule_results.copy()

        return {
            "conditions_met": conditions_met,
            "rule_results": rule_results,
            "metrics": {k: v for k, v in metrics.items() if k != "collected_at"},
        }

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        while True:
            try:
                async with self._config_lock:
                    enabled = self._config.enabled
                    has_rules = bool(self._config.rules)
                    interval = self._config.check_interval_seconds

                if enabled and has_rules:
                    metrics = await self._collector.collect()
                    self._last_metrics = metrics

                    async with self._config_lock:
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

                await asyncio.sleep(interval)
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
        # Handle both dataclass and dict rules (for backwards compatibility)
        enabled_rules = []
        for r in self._config.rules:
            is_enabled = r.enabled if hasattr(r, "enabled") else r.get("enabled", True)
            if is_enabled:
                enabled_rules.append(r)
        if not enabled_rules:
            self._last_rule_results = []
            return True

        # Inject queue_size into metrics so rules can reference it
        metrics["queue_size"] = self._queue.queue_size

        rule_results: list[dict[str, Any]] = []
        results: list[bool] = []
        for r in enabled_rules:
            key = r.rule if hasattr(r, "rule") else r.get("rule", "")
            passed = self._evaluate_rule(r, metrics)
            results.append(passed)
            # Check if the metric for this rule is available
            metric_available = self._is_metric_available(key, metrics)
            rule_results.append({
                "rule": key,
                "passed": passed,
                "metric_available": metric_available,
            })

        self._last_rule_results = rule_results

        if self._config.combine_mode == "ANY":
            return any(results)
        # Default: ALL
        return all(results)

    def _evaluate_rule(self, rule: "SchedulerRuleConfig", metrics: dict[str, Any]) -> bool:
        """Evaluate a single rule against collected metrics. Returns True if the rule passes."""
        from cosma_backend.settings import SCHEDULER_RULE_TYPES

        # Handle both dataclass and dict rules (for backwards compatibility)
        if hasattr(rule, "rule"):
            metric_key = rule.rule
            operator = rule.operator
            value = rule.value
        else:
            metric_key = rule.get("rule", "")
            operator = rule.get("operator", "")
            value = rule.get("value")

        # Use the default operator from the rule type registry when none is specified
        if not operator:
            rule_meta = SCHEDULER_RULE_TYPES.get(metric_key, {})
            operator = rule_meta.get("default_operator", "")

        if metric_key == "time_window":
            return self._evaluate_time_window(value)

        if metric_key == "power_source":
            actual = metrics.get("power_source_plugged")
            if actual is None:
                logger.debug("power_source metric unavailable, rule passes by default")
                return True
            expected = _to_bool(value)
            return actual == expected

        if metric_key == "low_power_mode":
            actual = metrics.get("low_power_mode")
            if actual is None:
                logger.debug("low_power_mode metric unavailable, rule passes by default")
                return True
            expected = _to_bool(value)
            return actual == expected

        # Numeric comparisons
        metric_map = {
            "battery_level": "battery_level",
            "cpu_idle": "cpu_idle_seconds",
            "cpu_temperature": "cpu_temperature",
            "fan_speed": "fan_speed",
            "memory_usage": "memory_usage",
            "gpu_usage": "gpu_usage",
            "queue_size": "queue_size",
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
        Also accepts a hyphen-separated string like ``"02:00-04:00"`` for
        backwards compatibility.
        """
        # Normalise legacy "HH:MM-HH:MM" string into a list
        if isinstance(value, str) and "-" in value:
            parts = value.split("-", 1)
            if len(parts) == 2:
                value = parts

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

    @staticmethod
    def _is_metric_available(rule_key: str, metrics: dict[str, Any]) -> bool:
        """Check if the metric for a rule is currently available."""
        if rule_key == "time_window":
            return True  # time is always available
        metric_map = {
            "battery_level": "battery_level",
            "power_source": "power_source_plugged",
            "cpu_idle": "cpu_idle_seconds",
            "cpu_temperature": "cpu_temperature",
            "fan_speed": "fan_speed",
            "memory_usage": "memory_usage",
            "memory_pressure": "memory_pressure",
            "gpu_usage": "gpu_usage",
            "queue_size": "queue_size",
            "low_power_mode": "low_power_mode",
        }
        metric_key = metric_map.get(rule_key, rule_key)
        return metrics.get(metric_key) is not None


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    return bool(v)
