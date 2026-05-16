from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class PowerSample:
    timestamp: float
    watts: float


@dataclass
class PowerMonitor:
    interval_sec: float = 0.05
    samples: list[PowerSample] = field(default_factory=list)
    available: bool = False
    error: str | None = None
    _thread: threading.Thread | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _pynvml: object | None = None
    _handle: object | None = None

    def __enter__(self) -> "PowerMonitor":
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on host GPU/NVML
            self.error = str(exc)
            self.available = False
            return self

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def _loop(self) -> None:
        assert self._pynvml is not None
        assert self._handle is not None
        while not self._stop.is_set():
            try:
                milliwatts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append(PowerSample(time.perf_counter(), milliwatts / 1000.0))
            except Exception as exc:
                self.error = str(exc)
                self.available = False
                return
            time.sleep(self.interval_sec)

    def summary(self) -> dict:
        if not self.samples:
            return {
                "available": self.available,
                "error": self.error,
                "sample_count": 0,
                "avg_watts": None,
                "max_watts": None,
                "elapsed_sec": 0.0,
                "energy_joules": None,
            }
        elapsed = self.samples[-1].timestamp - self.samples[0].timestamp
        watts = [sample.watts for sample in self.samples]
        avg_watts = sum(watts) / len(watts)
        return {
            "available": self.available,
            "error": self.error,
            "sample_count": len(self.samples),
            "avg_watts": avg_watts,
            "max_watts": max(watts),
            "elapsed_sec": elapsed,
            "energy_joules": avg_watts * elapsed,
        }

