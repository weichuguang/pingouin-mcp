"""
Markdown rendering of statistical results.

An agent reading a tool result needs three things, in this order:

1. **What the number means** — not its 15th decimal place.
2. **Whether the assumption was met** — the step that determines whether the
   number should be trusted at all.
3. **What to do next** — the decision the result enables.

These helpers standardize that rendering so every tool speaks the same visual
and logical language.
"""

from __future__ import annotations

import math
from typing import Any


def fmt(value: Any, decimals: int = 4) -> str:
    """Format a number for display the way a journal would."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    if f == 0:
        return "0"
    a = abs(f)
    if a < 1e-4:
        return f"{f:.2e}"
    if a >= 1e5:
        return f"{f:.3e}"
    return f"{f:.{decimals}f}".rstrip("0").rstrip(".")


def pfmt(p: Any) -> str:
    """Format a p-value the way it is reported in papers."""
    if p is None:
        return "—"
    try:
        f = float(p)
    except (TypeError, ValueError):
        return str(p)
    if math.isnan(f):
        return "nan"
    if f < 1e-4:
        return "< .0001"
    if f < 0.001:
        return f"{f:.5f}"
    return f"{f:.4f}".lstrip("0") if f < 1 else f"{f:.4f}"


def sig_marker(p: Any) -> str:
    """Conventional significance asterisks."""
    try:
        f = float(p)
    except (TypeError, ValueError):
        return ""
    if math.isnan(f):
        return ""
    if f < 0.001:
        return "***"
    if f < 0.01:
        return "**"
    if f < 0.05:
        return "*"
    return "ns"


def effect_magnitude(kind: str, value: Any) -> str:
    """
    Qualitative label for an effect size, using published conventions.

    These thresholds are conventional rather than mathematical — they are
    offered as orientation, and the report always shows the numeric value
    alongside so a reader is never forced to rely on the label.
    """
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return ""
    if kind in ("cohen_d", "d", "hedges_g"):
        if v < 0.2:
            return "negligible"
        if v < 0.5:
            return "small"
        if v < 0.8:
            return "medium"
        return "large"
    if kind in ("eta_squared", "eta2", "partial_eta_squared", "np2"):
        if v < 0.01:
            return "negligible"
        if v < 0.06:
            return "small"
        if v < 0.14:
            return "medium"
        return "large"
    if kind in ("r", "rho", "correlation"):
        if v < 0.1:
            return "negligible"
        if v < 0.3:
            return "small"
        if v < 0.5:
            return "medium"
        return "large"
    if kind in ("r2", "r_squared"):
        if v < 0.01:
            return "negligible"
        if v < 0.09:
            return "small"
        if v < 0.25:
            return "medium"
        return "large"
    return ""


def assumptions_block(items: list[tuple[str, bool | None, str]]) -> list[str]:
    """
    Render an assumption checklist.

    ``items`` is a list of ``(name, passed, note)``. ``passed`` may be ``None``
    when the assumption could not be evaluated — rendered as "not checked"
    rather than silently treated as passing.
    """
    lines = ["", "**Assumption checks**", "", "| Assumption | Result | Note |", "|---|---|---|"]
    for name, passed, note in items:
        if passed is None:
            mark = "not checked"
        elif passed:
            mark = "met"
        else:
            mark = "**violated**"
        lines.append(f"| {name} | {mark} | {note} |")
    return lines


def next_steps_block(options: list[str]) -> list[str]:
    """Render a 'what you can do next' list."""
    if not options:
        return []
    lines = ["", "**Possible next steps**", ""]
    lines += [f"- {o}" for o in options]
    return lines


def table(rows: list[dict], columns: list[str] | None = None, decimals: int = 4) -> str:
    """Render a list of dicts as a Markdown table with formatted numbers."""
    if not rows:
        return "_No rows._"
    columns = columns or list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c)
            if isinstance(v, float):
                cells.append(fmt(v, decimals))
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


def dataframe_table(df, max_rows: int = 40, decimals: int = 4) -> str:
    """Render a pandas DataFrame as a Markdown table."""
    try:
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            return f"```\n{df}\n```"
        if len(df) > max_rows:
            head = df.head(max_rows // 2)
            tail = df.tail(max_rows // 2)
            body = pd.concat([head, tail])
        else:
            body = df
        rows = []
        for _, r in body.iterrows():
            rows.append({str(k): v for k, v in r.items()})
        out = table(rows, [str(c) for c in df.columns], decimals=decimals)
        if len(df) > max_rows:
            out += f"\n\n_({len(df) - max_rows} more rows omitted)_"
        return out
    except Exception:
        return f"```\n{df}\n```"


def header(title: str, subtitle: str | None = None) -> list[str]:
    """A consistent result header."""
    lines = [f"### {title}"]
    if subtitle:
        lines.append(f"_{subtitle}_")
    return lines
