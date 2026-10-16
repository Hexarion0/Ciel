"""Process-tree CPU/RAM and low-overhead NVML GPU/power/fan sampling."""

from __future__ import annotations

import statistics
import threading
import time

import psutil


class Sampler:
    """GPU power, utilization, temperature, and fan speed describe the whole device.

    Only compute VRAM can be attributed to the test process tree. Fan speed is a
    controller percentage, not RPM or an acoustic/fan-comfort measurement.
    """

    def __init__(self, interval: float = 0.1):
        self.process = psutil.Process()
        self.interval = interval
        self.stop = threading.Event()
        self.rows = []
        self.cpu_times = {}
        self.nvml = None
        self.gpu = None
        self.gpu_errors = {}
        try:
            import pynvml

            pynvml.nvmlInit()
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml = pynvml
        except Exception as error:
            self.gpu_errors["initialization"] = f"{type(error).__name__}: {error}"
        self.thread = threading.Thread(target=self._run, name="shootout-telemetry", daemon=True)

    def _tree(self):
        return [self.process, *self.process.children(recursive=True)]

    def cpu_seconds(self) -> float:
        for process in self._tree():
            try:
                times = process.cpu_times()
                self.cpu_times[process.pid] = times.user + times.system
            except psutil.Error:
                pass
        return sum(self.cpu_times.values())

    def snapshot(self) -> dict:
        pids = set()
        rss = 0
        for process in self._tree():
            try:
                rss += process.memory_info().rss
                pids.add(process.pid)
            except psutil.Error:
                pass
        result = {"elapsed_seconds": time.perf_counter() - self.started,
                  "rss_mib": rss / 1024**2}
        if self.nvml is None:
            return result
        api, handle = self.nvml, self.gpu
        queries = {
            "gpu_util_percent": lambda: api.nvmlDeviceGetUtilizationRates(handle).gpu,
            "gpu_total_vram_mib": lambda: api.nvmlDeviceGetMemoryInfo(handle).used / 1024**2,
            "gpu_power_w": lambda: api.nvmlDeviceGetPowerUsage(handle) / 1000,
            "gpu_temp_c": lambda: api.nvmlDeviceGetTemperature(handle, api.NVML_TEMPERATURE_GPU),
            "gpu_fan_percent": lambda: api.nvmlDeviceGetFanSpeed(handle),
            "engine_vram_mib": lambda: sum(
                process.usedGpuMemory / 1024**2
                for process in api.nvmlDeviceGetComputeRunningProcesses(handle)
                if process.pid in pids and process.usedGpuMemory != api.NVML_VALUE_NOT_AVAILABLE
            ),
        }
        for key, query in queries.items():
            try:
                result[key] = float(query())
            except Exception as error:
                self.gpu_errors[key] = f"{type(error).__name__}: {error}"
        return result

    def _run(self):
        while not self.stop.is_set():
            self.rows.append(self.snapshot())
            self.stop.wait(self.interval)

    def __enter__(self):
        self.started = time.perf_counter()
        self.cpu_start = self.cpu_seconds()
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=5)
        self.rows.append(self.snapshot())
        elapsed = time.perf_counter() - self.started
        cpu = 100 * (self.cpu_seconds() - self.cpu_start) / elapsed
        self.metrics = {"cpu_average_percent_one_core_100": cpu,
                        "cpu_average_percent_machine": cpu / (psutil.cpu_count() or 1),
                        "sample_interval_seconds": self.interval, "samples": len(self.rows),
                        "fan_rpm": None, "gpu_telemetry_errors": self.gpu_errors}
        for key in ("rss_mib", "gpu_util_percent", "gpu_total_vram_mib", "engine_vram_mib",
                    "gpu_power_w", "gpu_temp_c", "gpu_fan_percent"):
            values = [row[key] for row in self.rows if key in row]
            for label, fn in (("peak", max), ("median", statistics.median), ("min", min)):
                self.metrics[f"{key}_{label}"] = fn(values) if values else None
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
