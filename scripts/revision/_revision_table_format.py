"""Shared Markdown formatting only; no retrieval metric or interval computation."""

from __future__ import annotations

import math
from numbers import Real


class TableFormatter:
    def __init__(self, decimals=2):
        if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 12:
            raise ValueError("--decimals must be an integer in [0, 12]")
        self.decimals = decimals
        self.missing = []

    def reason(self, reason):
        if reason not in self.missing:
            self.missing.append(str(reason))

    def _absent(self, context, explanation="missing or null in input"):
        self.reason(f"{context}: {explanation}")
        return "n/a"

    def number(self, value, context, signed=False, percent=False):
        if value is None:
            return self._absent(context)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{context}: expected a finite recorded number, got {value!r}")
        value = float(value) * (100 if percent else 1)
        if not math.isfinite(value):
            raise ValueError(f"{context}: unit conversion overflowed; not emitting a non-finite value")
        result = f"{value:+.{self.decimals}f}" if signed else f"{value:.{self.decimals}f}"
        # Keep the actual sign through rounding, including a negative rounded zero.
        return result.replace("-", "−")

    def count(self, value, context):
        if value is None:
            return self._absent(context)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or int(value) != value:
            raise ValueError(f"{context}: expected an integer count, got {value!r}")
        return str(int(value))

    def text(self, value, context):
        if value is None:
            return self._absent(context)
        return str(value)

    def interval(self, value, context, signed=False, percent=False):
        if value is None:
            return self._absent(context)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(f"{context}: expected a two-endpoint recorded interval")
        if any(item is None for item in value):
            return self._absent(context, "recorded interval has a null endpoint")
        if value[0] > value[1]:
            raise ValueError(f"{context}: lower interval endpoint exceeds upper endpoint")
        return "[" + ", ".join(self.number(item, context, signed=signed, percent=percent) for item in value) + "]"

    def with_ci(self, value, ci, context, signed=False, percent=False):
        estimate = self.number(value, context + ".estimate", signed=signed, percent=percent)
        interval = self.interval(ci, context + ".CI", signed=signed, percent=percent)
        return f"{estimate} {interval}" if interval != "n/a" else f"{estimate} [n/a]"

    def mean_std(self, mean, std, context, signed=False, percent=False):
        return (self.number(mean, context + ".mean", signed=signed, percent=percent) + " ± "
                + self.number(std, context + ".std", percent=percent))

    @staticmethod
    def table(headers, rows):
        def cell(value):
            return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")
        result = ["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        for row in rows:
            if len(row) != len(headers):
                raise ValueError(f"Markdown row has {len(row)} cells; expected {len(headers)}")
            result.append("| " + " | ".join(map(cell, row)) + " |")
        return "\n".join(result) + "\n"
