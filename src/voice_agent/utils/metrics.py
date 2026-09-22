"""Global metrics registry (stdlib only, Prometheus-compatible exposition)."""

from __future__ import annotations

import threading
import time


class _Counter:
    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value


class _Gauge:
    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class _Histogram:
    """Fixed-bucket histogram with sum/count (ms and seconds friendly)."""

    __slots__ = ("_buckets", "_counts", "_lock", "_sum")

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self._buckets = buckets
        self._counts = [0] * len(buckets)
        self._sum = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._counts[i] += 1

    def snapshot(self) -> tuple[tuple[float, ...], tuple[int, ...], float, int]:
        total = self._counts[-1] if self._counts else 0
        return self._buckets, tuple(self._counts), self._sum, total


LATENCY_BUCKETS = (25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 15000.0)


class MetricsRegistry:
    """Process-wide metrics: connections, turns, per-stage latency, errors."""

    def __init__(self) -> None:
        self._counters: dict[str, _Counter] = {}
        self._gauges: dict[str, _Gauge] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._meta_lock = threading.Lock()
        self._start_time = time.time()

    # -- primitives --
    def counter(self, name: str) -> _Counter:
        with self._meta_lock:
            return self._counters.setdefault(name, _Counter())

    def gauge(self, name: str) -> _Gauge:
        with self._meta_lock:
            return self._gauges.setdefault(name, _Gauge())

    def histogram(self, name: str) -> _Histogram:
        with self._meta_lock:
            return self._histograms.setdefault(name, _Histogram(LATENCY_BUCKETS))

    # -- convenience --
    def inc(self, name: str, amount: int = 1) -> None:
        self.counter(name).inc(amount)

    def observe(self, name: str, value_ms: float) -> None:
        self.histogram(name).observe(value_ms)

    def uptime_s(self) -> float:
        return time.time() - self._start_time

    def to_dict(self) -> dict[str, object]:
        return {
            "uptime_s": round(self.uptime_s(), 1),
            "counters": {k: c.value for k, c in self._counters.items()},
            "gauges": {k: round(g.value, 3) for k, g in self._gauges.items()},
            "histograms": {
                k: {
                    "buckets": list(b),
                    "counts": list(n),
                    "sum_ms": round(s, 2),
                    "count": t,
                    "avg_ms": round(s / t, 2) if t else 0.0,
                }
                for k, (b, n, s, t) in (
                    (k, h.snapshot()) for k, h in self._histograms.items()
                )
            },
        }

    def to_prometheus(self) -> str:
        """Render Prometheus text exposition format."""
        lines: list[str] = []
        for name, c in sorted(self._counters.items()):
            lines.append(f"# TYPE voice_{name} counter")
            lines.append(f"voice_{name} {c.value}")
        for name, g in sorted(self._gauges.items()):
            lines.append(f"# TYPE voice_{name} gauge")
            lines.append(f"voice_{name} {g.value}")
        for name, h in sorted(self._histograms.items()):
            buckets, counts, total, _ = h.snapshot()
            total_count = counts[-1] if counts else 0
            base = name[:-3] if name.endswith("_ms") else name
            lines.append(f"# TYPE voice_{base}_ms histogram")
            for bound, count in zip(buckets, counts, strict=True):
                lines.append(f'voice_{base}_ms_bucket{{le="{bound / 1000}"}} {count}')
            lines.append(f'voice_{base}_ms_bucket{{le="+Inf"}} {total_count}')
            lines.append(f"voice_{base}_ms_sum {total / 1000}")
            lines.append(f"voice_{base}_ms_count {total_count}")
        lines.append("")
        return "\n".join(lines)


_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Return the process-wide metrics registry."""
    return _registry
