# -*- coding: utf-8 -*-
"""
Generate TOOLS_REFERENCE.md from the live server.

The reference is generated rather than hand-written so it cannot drift from the
registered tools. Run after changing any tool signature or docstring:

    python scripts/gen_tools_reference.py
"""
from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

from pingouin_mcp import operations as ops
from pingouin_mcp.server import mcp

GROUPS = [
    ("Data management", [
        "stat_load_data", "stat_make_dataframe", "stat_describe_data",
        "stat_data_quality_check"]),
    ("Decision support", ["stat_decision_assistant"]),
    ("Assumption testing", [
        "stat_assumption_report", "stat_normality_test", "stat_homogeneity_test",
        "stat_sphericity_test"]),
    ("Two-group comparisons", [
        "stat_ttest", "stat_nonparametric_2", "stat_bayes_ttest"]),
    ("Multi-group comparisons", [
        "stat_anova", "stat_welch_anova", "stat_rm_anova", "stat_mixed_anova",
        "stat_ancova", "stat_nonparametric_k"]),
    ("Post-hoc and multiple comparisons", [
        "stat_pairwise_tests", "stat_pairwise_tukey", "stat_multicomp_correction"]),
    ("Association", [
        "stat_correlation", "stat_partial_correlation", "stat_rm_correlation",
        "stat_correlation_matrix", "stat_distance_correlation"]),
    ("Regression and mediation", [
        "stat_linear_regression", "stat_logistic_regression", "stat_mediation"]),
    ("Effect size, power, reliability", [
        "stat_effect_size", "stat_power_analysis", "stat_cronbach_alpha",
        "stat_intraclass_correlation", "stat_equivalence_test"]),
    ("Categorical data", ["stat_chi2_independence", "stat_chi2_mcnemar"]),
    ("Visualization", [
        "stat_plot_paired", "stat_plot_blandaltman", "stat_qqplot", "stat_plot_rm_corr"]),
    ("Session", ["stat_session_info", "stat_reset_session", "stat_run_code"]),
]


def _params(fn) -> str:
    sig = inspect.signature(fn)
    rows = []
    for name, p in sig.parameters.items():
        if name in ("ctx",):
            continue
        ann = p.annotation
        ann = "" if ann is inspect.Parameter.empty else str(ann).replace("'", "")
        default = "" if p.default is inspect.Parameter.empty else repr(p.default)
        rows.append(f"| `{name}` | {ann or '—'} | {default or '—'} |")
    if not rows:
        return "_No parameters._"
    return "| Parameter | Type | Default |\n|---|---|---|\n" + "\n".join(rows)


async def build() -> str:
    tools = {t.name: t for t in await mcp.list_tools()}
    out = [
        "# Pingouin-MCP — Tool Reference",
        "",
        f"**{len(tools)} tools** across {len(GROUPS)} groups, organised by research task "
        "rather than by Pingouin function name.",
        "",
        "> Generated from the running server by `scripts/gen_tools_reference.py`. "
        "Do not edit by hand — the file is regenerated from the registered tools and their "
        "docstrings.",
        "",
        "Every tool returns Markdown for the agent to relay, the equivalent Pingouin code for "
        "reproduction, and (where applicable) the path to a saved figure.",
        "",
        "---",
        "",
    ]

    listed = set()
    for group, names in GROUPS:
        present = [n for n in names if n in tools]
        if not present:
            continue
        out += [f"## {group}", ""]
        for name in present:
            listed.add(name)
            t = tools[name]
            out += [f"### `{name}`", ""]
            desc = (t.description or "").strip()
            out += [desc, ""]
            fn = ops.OPERATIONS.get(name)
            if fn is not None:
                out += [_params(fn), ""]
        out += ["---", ""]

    # Anything registered but not yet grouped — surfaced rather than hidden.
    ungrouped = sorted(set(tools) - listed)
    if ungrouped:
        out += ["## Ungrouped tools", "",
                "_Registered but not yet placed in a group above._", ""]
        for name in ungrouped:
            out += [f"### `{name}`", "", (tools[name].description or "").strip(), ""]
        out += ["---", ""]

    out += [
        "## Conventions",
        "",
        "- **Session-scoped.** Datasets are referenced by name across calls. "
        "`stat_session_info` lists what is loaded.",
        "- **Reproducible.** Every analysis returns the equivalent Pingouin code.",
        "- **Assumptions are explicit.** Tools report the checks their test requires rather than "
        "silently assuming them.",
        "- **Errors are messages, not exceptions.** A bad column or dataset name comes back as "
        "readable text so the caller can correct it.",
        "",
        "## Version",
        "",
        "Verified against **Pingouin 0.6.1**, pandas 3.x, Python 3.13.",
        "Pingouin renamed its result columns from hyphenated to underscored in 0.6 "
        "(`p-val` → `p_val`); the server resolves both spellings.",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "TOOLS_REFERENCE.md"
    md = asyncio.run(build())
    target.write_text(md, encoding="utf-8")
    print(f"wrote {target}  ({len(md)} chars)")
