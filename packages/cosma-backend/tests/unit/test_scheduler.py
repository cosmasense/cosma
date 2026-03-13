"""Unit tests for Scheduler and SystemMetricsCollector."""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cosma_backend.queue.scheduler import Scheduler, _to_bool
from cosma_backend.queue.metrics import SystemMetricsCollector
from cosma_backend.settings import SchedulerConfig, SchedulerRuleConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_queue():
    q = MagicMock()
    q.scheduler_pause = MagicMock()
    q.scheduler_resume = MagicMock()
    q.queue_size = 0
    return q


@pytest.fixture
def scheduler(mock_queue):
    config = SchedulerConfig(enabled=True, combine_mode="ALL", check_interval_seconds=5)
    return Scheduler(queue=mock_queue, config=config)


# ---------------------------------------------------------------------------
# Rule evaluation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSchedulerRuleEvaluation:
    def test_battery_level_gt(self, scheduler):
        rule = SchedulerRuleConfig(rule="battery_level", operator="gt", value=20)
        metrics = {"battery_level": 50}
        assert scheduler._evaluate_rule(rule, metrics) is True

    def test_battery_level_lt(self, scheduler):
        rule = SchedulerRuleConfig(rule="battery_level", operator="lt", value=20)
        metrics = {"battery_level": 50}
        assert scheduler._evaluate_rule(rule, metrics) is False

    def test_power_source_eq_true(self, scheduler):
        rule = SchedulerRuleConfig(rule="power_source", operator="eq", value=True)
        metrics = {"power_source_plugged": True}
        assert scheduler._evaluate_rule(rule, metrics) is True

    def test_power_source_eq_false(self, scheduler):
        rule = SchedulerRuleConfig(rule="power_source", operator="eq", value=True)
        metrics = {"power_source_plugged": False}
        assert scheduler._evaluate_rule(rule, metrics) is False

    def test_cpu_idle_gt(self, scheduler):
        rule = SchedulerRuleConfig(rule="cpu_idle", operator="gt", value=0)
        metrics = {"cpu_idle_seconds": 300.0}
        assert scheduler._evaluate_rule(rule, metrics) is True

    def test_time_window_within(self, scheduler):
        now = datetime.datetime.now().time()
        # Create a window that includes the current time
        start = (datetime.datetime.now() - datetime.timedelta(hours=1)).strftime("%H:%M")
        end = (datetime.datetime.now() + datetime.timedelta(hours=1)).strftime("%H:%M")
        rule = SchedulerRuleConfig(rule="time_window", operator="within", value=[start, end])
        assert scheduler._evaluate_rule(rule, {}) is True

    def test_time_window_outside(self, scheduler):
        # Create a window that excludes the current time
        now = datetime.datetime.now()
        start = (now + datetime.timedelta(hours=2)).strftime("%H:%M")
        end = (now + datetime.timedelta(hours=3)).strftime("%H:%M")
        rule = SchedulerRuleConfig(rule="time_window", operator="within", value=[start, end])
        assert scheduler._evaluate_rule(rule, {}) is False

    def test_time_window_crossing_midnight(self, scheduler):
        rule = SchedulerRuleConfig(rule="time_window", operator="within", value=["23:00", "05:00"])
        # This should pass if current time is between 23:00 and 05:00, or fail otherwise
        # We just verify it doesn't crash
        result = scheduler._evaluate_rule(rule, {})
        assert isinstance(result, bool)

    def test_cpu_temperature_lt(self, scheduler):
        rule = SchedulerRuleConfig(rule="cpu_temperature", operator="lt", value=80)
        metrics = {"cpu_temperature": 65.0}
        assert scheduler._evaluate_rule(rule, metrics) is True

    def test_fan_speed_lt(self, scheduler):
        rule = SchedulerRuleConfig(rule="fan_speed", operator="lt", value=3000)
        metrics = {"fan_speed": 1200.0}
        assert scheduler._evaluate_rule(rule, metrics) is True

    def test_missing_metric_passes(self, scheduler):
        """Missing metric → rule passes by default."""
        rule = SchedulerRuleConfig(rule="battery_level", operator="gt", value=20)
        metrics = {}  # no battery_level
        assert scheduler._evaluate_rule(rule, metrics) is True

    def test_unknown_operator_passes(self, scheduler):
        rule = SchedulerRuleConfig(rule="battery_level", operator="unknown_op", value=20)
        metrics = {"battery_level": 50}
        assert scheduler._evaluate_rule(rule, metrics) is True


# ---------------------------------------------------------------------------
# Combine mode
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSchedulerCombineMode:
    def test_all_with_all_pass(self, scheduler):
        scheduler._config.combine_mode = "ALL"
        scheduler._config.rules = [
            SchedulerRuleConfig(rule="battery_level", operator="gt", value=20, enabled=True),
            SchedulerRuleConfig(rule="cpu_temperature", operator="lt", value=80, enabled=True),
        ]
        metrics = {"battery_level": 50, "cpu_temperature": 65.0}
        assert scheduler._evaluate_rules(metrics) is True

    def test_all_with_one_fail(self, scheduler):
        scheduler._config.combine_mode = "ALL"
        scheduler._config.rules = [
            SchedulerRuleConfig(rule="battery_level", operator="gt", value=60, enabled=True),
            SchedulerRuleConfig(rule="cpu_temperature", operator="lt", value=80, enabled=True),
        ]
        metrics = {"battery_level": 50, "cpu_temperature": 65.0}
        assert scheduler._evaluate_rules(metrics) is False

    def test_any_with_one_pass(self, scheduler):
        scheduler._config.combine_mode = "ANY"
        scheduler._config.rules = [
            SchedulerRuleConfig(rule="battery_level", operator="gt", value=60, enabled=True),
            SchedulerRuleConfig(rule="cpu_temperature", operator="lt", value=80, enabled=True),
        ]
        metrics = {"battery_level": 50, "cpu_temperature": 65.0}
        assert scheduler._evaluate_rules(metrics) is True

    def test_any_with_none_pass(self, scheduler):
        scheduler._config.combine_mode = "ANY"
        scheduler._config.rules = [
            SchedulerRuleConfig(rule="battery_level", operator="gt", value=60, enabled=True),
            SchedulerRuleConfig(rule="cpu_temperature", operator="lt", value=50, enabled=True),
        ]
        metrics = {"battery_level": 50, "cpu_temperature": 65.0}
        assert scheduler._evaluate_rules(metrics) is False

    def test_no_enabled_rules_returns_true(self, scheduler):
        scheduler._config.rules = [
            SchedulerRuleConfig(rule="battery_level", operator="gt", value=20, enabled=False),
        ]
        assert scheduler._evaluate_rules({}) is True


# ---------------------------------------------------------------------------
# Config updates
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSchedulerUpdateConfig:
    async def test_update_enabled(self, scheduler):
        await scheduler.update_config({"enabled": False})
        assert scheduler._config.enabled is False

    async def test_update_combine_mode(self, scheduler):
        await scheduler.update_config({"combine_mode": "any"})
        assert scheduler._config.combine_mode == "ANY"

    async def test_update_rules_from_dict(self, scheduler):
        await scheduler.update_config({
            "rules": [
                {"rule": "battery_level", "operator": "gt", "value": 30, "enabled": True},
                {"rule": "cpu_temperature", "operator": "lt", "value": 80, "enabled": False},
            ]
        })
        assert len(scheduler._config.rules) == 2
        assert scheduler._config.rules[0].rule == "battery_level"
        assert scheduler._config.rules[1].enabled is False


# ---------------------------------------------------------------------------
# SystemMetricsCollector
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSystemMetricsCollector:
    async def test_collect_returns_expected_keys(self):
        collector = SystemMetricsCollector()
        metrics = await collector.collect()
        expected_keys = {
            "battery_level",
            "power_source_plugged",
            "cpu_percent",
            "cpu_idle_seconds",
            "cpu_temperature",
            "fan_speed",
            "memory_usage",
            "memory_pressure",
            "gpu_usage",
            "low_power_mode",
            "collected_at",
        }
        assert set(metrics.keys()) == expected_keys

    async def test_correct_value_types(self):
        collector = SystemMetricsCollector()
        metrics = await collector.collect()
        # collected_at is always a float
        assert isinstance(metrics["collected_at"], float)
        # Numeric values are float or None
        for key in ("battery_level", "cpu_percent", "cpu_idle_seconds", "cpu_temperature", "fan_speed"):
            val = metrics[key]
            assert val is None or isinstance(val, (int, float))
        # power_source is bool or None
        assert metrics["power_source_plugged"] is None or isinstance(metrics["power_source_plugged"], bool)
