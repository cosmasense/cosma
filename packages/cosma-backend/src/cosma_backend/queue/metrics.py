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

    # CPU idle threshold (percentage below which CPU is considered idle)
    _CPU_IDLE_THRESHOLD = 15.0

    def __init__(self) -> None:
        self._idle_since: Optional[float] = None

    async def collect(self) -> dict[str, Any]:
        """Return a snapshot of current system metrics.

        All synchronous psutil calls go through ``asyncio.to_thread`` so the
        event loop stays responsive. ``psutil.cpu_percent(interval=0.1)``
        in particular blocks for 100 ms — on the event loop it stalls every
        HTTP request. Under processing load, this collector was the top
        source of slow-request warnings.
        """
        metrics: dict[str, Any] = {}

        metrics["battery_level"] = await asyncio.to_thread(self._get_battery_level)
        metrics["power_source_plugged"] = await asyncio.to_thread(self._get_power_source)
        metrics["cpu_percent"] = await asyncio.to_thread(self._get_cpu_percent)
        metrics["cpu_idle_seconds"] = await self._get_cpu_idle_seconds()
        metrics["cpu_temperature"] = await self._get_cpu_temperature()
        metrics["fan_speed"] = await self._get_fan_speed()
        metrics["memory_usage"] = await asyncio.to_thread(self._get_memory_usage)
        metrics["memory_pressure"] = await asyncio.to_thread(self._get_memory_pressure)
        metrics["gpu_usage"] = await self._get_gpu_usage()
        metrics["low_power_mode"] = await self._get_low_power_mode()
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
        """Track actual idle duration using a rolling timestamp.

        Records when CPU first dropped below the idle threshold and
        returns elapsed seconds since then.  Resets when CPU goes above
        the threshold.
        """
        try:
            import psutil
            percent = await asyncio.to_thread(psutil.cpu_percent, 0.1)
            now = time.time()
            if percent < self._CPU_IDLE_THRESHOLD:
                if self._idle_since is None:
                    self._idle_since = now
                return now - self._idle_since
            else:
                self._idle_since = None
                return 0.0
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Temperature (Apple Silicon: smctemp, Intel: osx-cpu-temp)
    # ------------------------------------------------------------------

    async def _get_cpu_temperature(self) -> Optional[float]:
        # Apple Silicon + Intel: smctemp -c (outputs a single float like "83.4")
        if shutil.which("smctemp"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "smctemp", "-c",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                text = stdout.decode().strip()
                if text:
                    return float(text)
            except Exception as e:
                logger.debug("smctemp failed", error=str(e))

        # psutil (Linux, some Windows)
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                for name in ("coretemp", "cpu_thermal", "cpu-thermal"):
                    if name in temps and temps[name]:
                        return temps[name][0].current
                first = next(iter(temps.values()))
                if first:
                    return first[0].current
        except (AttributeError, Exception):
            pass

        # Intel fallback: osx-cpu-temp
        if shutil.which("osx-cpu-temp"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "osx-cpu-temp",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                text = stdout.decode().strip()
                temp_str = text.replace("\u00b0C", "").strip()
                return float(temp_str)
            except Exception as e:
                logger.debug("osx-cpu-temp failed", error=str(e))

        return None

    # ------------------------------------------------------------------
    # Fan speed (Apple Silicon: smctemp -l, Intel: istats)
    # ------------------------------------------------------------------

    async def _get_fan_speed(self) -> Optional[float]:
        import re as _re

        # Apple Silicon + Intel: smctemp -l, parse F0Ac key
        if shutil.which("smctemp"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "smctemp", "-l",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
                for line in stdout.decode().splitlines():
                    # Match "  F0Ac  [flt ]  1203.4 (bytes: ...)"
                    if line.strip().startswith("F0Ac"):
                        match = _re.search(r'\]\s+([\d.]+)', line)
                        if match:
                            return float(match.group(1))
            except Exception as e:
                logger.debug("smctemp fan speed failed", error=str(e))

        # Intel fallback: istats
        if shutil.which("istats"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    "istats", "fan", "speed",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
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

    def _get_memory_pressure(self) -> Optional[float]:
        """Return memory pressure as a percentage (0-100).

        Calculated as (active + wired) / total * 100, which represents
        non-reclaimable memory. This is different from memory_usage which
        includes inactive/cached pages that macOS can reclaim.
        """
        try:
            import psutil
            vm = psutil.virtual_memory()
            if hasattr(vm, 'active') and hasattr(vm, 'wired'):
                return round((vm.active + vm.wired) / vm.total * 100, 1)
            return vm.percent
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

    # ------------------------------------------------------------------
    # Low Power Mode (macOS)
    # ------------------------------------------------------------------

    async def _get_low_power_mode(self) -> Optional[bool]:
        """Return True if macOS Low Power Mode is active."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pmset", "-g",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            text = stdout.decode()
            # Look for "lowpowermode 1" or "lowpowermode  1"
            for line in text.splitlines():
                if "lowpowermode" in line.lower():
                    return "1" in line.split()[-1:]
        except Exception as e:
            logger.debug("Low power mode metric unavailable", error=str(e))

        return None
