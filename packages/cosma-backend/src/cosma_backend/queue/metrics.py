"""
System Metrics Collector

Collects battery, CPU, temperature, and fan speed information using psutil
and macOS CLI tools (osx-cpu-temp, istats) when available.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import Any, Optional

from cosma_backend.logging import get_logger

logger = get_logger(__name__)


class SystemMetricsCollector:
    """Collects system metrics for use by the scheduler rule engine."""

    async def collect(self) -> dict[str, Any]:
        """Return a snapshot of current system metrics."""
        metrics: dict[str, Any] = {}

        metrics["battery_level"] = self._get_battery_level()
        metrics["power_source_plugged"] = self._get_power_source()
        metrics["cpu_percent"] = self._get_cpu_percent()
        metrics["cpu_idle_seconds"] = await self._get_cpu_idle_seconds()
        metrics["cpu_temperature"] = await self._get_cpu_temperature()
        metrics["fan_speed"] = await self._get_fan_speed()
        metrics["memory_usage"] = self._get_memory_usage()
        metrics["gpu_usage"] = await self._get_gpu_usage()
        metrics["collected_at"] = time.time()

        return metrics

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------

    def _get_battery_level(self) -> Optional[float]:
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery is not None:
                return battery.percent
        except Exception:
            pass
        return None

    def _get_power_source(self) -> Optional[bool]:
        """True if plugged in, False if on battery, None if unknown."""
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery is not None:
                return battery.power_plugged
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    def _get_cpu_percent(self) -> Optional[float]:
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return None

    async def _get_cpu_idle_seconds(self) -> Optional[float]:
        """Approximate idle time based on low CPU usage.

        This is a rough heuristic — we don't track a rolling window, so
        we return 0 if CPU is busy and a large value if idle.  The
        scheduler rule should use a threshold like ``cpu_idle gt 0`` to
        mean "CPU is currently idle".
        """
        try:
            import psutil
            percent = psutil.cpu_percent(interval=0.1)
            # Consider "idle" if usage is below 15%
            if percent < 15:
                return 300.0  # report 5 min idle as a proxy
            return 0.0
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Temperature (macOS: osx-cpu-temp)
    # ------------------------------------------------------------------

    async def _get_cpu_temperature(self) -> Optional[float]:
        # First try psutil (Linux, some Windows)
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ("coretemp", "cpu_thermal", "cpu-thermal"):
                    if name in temps and temps[name]:
                        return temps[name][0].current
                # Fallback to first available
                first = next(iter(temps.values()))
                if first:
                    return first[0].current
        except (AttributeError, Exception):
            pass

        # macOS fallback: osx-cpu-temp
        if shutil.which("osx-cpu-temp"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "osx-cpu-temp",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                # Output like "65.0°C"
                text = stdout.decode().strip()
                temp_str = text.replace("°C", "").strip()
                return float(temp_str)
            except Exception as e:
                logger.debug("osx-cpu-temp failed", error=str(e))

        return None

    # ------------------------------------------------------------------
    # Fan speed (macOS: istats)
    # ------------------------------------------------------------------

    async def _get_fan_speed(self) -> Optional[float]:
        if shutil.which("istats"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "istats", "fan", "speed",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                # Parse lines like "Fan 0 speed:  1200 RPM"
                for line in stdout.decode().splitlines():
                    if "RPM" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "RPM" and i > 0:
                                return float(parts[i - 1])
            except Exception as e:
                logger.debug("istats failed", error=str(e))

        return None

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _get_memory_usage(self) -> Optional[float]:
        """Return system memory usage as a percentage (0-100)."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except Exception:
            return None

    # ------------------------------------------------------------------
    # GPU (Apple Silicon)
    # ------------------------------------------------------------------

    async def _get_gpu_usage(self) -> Optional[float]:
        """Return GPU utilization percentage on Apple Silicon via ioreg."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "ioreg", "-r", "-d", "1", "-c", "IOAccelerator",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            # Look for "Device Utilization %" or "GPU Activity(%)"
            import re
            text = stdout.decode()
            # Try "Device Utilization %" first (common on Apple Silicon)
            match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', text)
            if match:
                return float(match.group(1))
            # Fallback: "GPU Activity(%)"
            match = re.search(r'"GPU Activity\(%\)"\s*=\s*(\d+)', text)
            if match:
                return float(match.group(1))
        except Exception as e:
            logger.debug("GPU usage metric unavailable", error=str(e))

        return None
