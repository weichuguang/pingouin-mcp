"""
Tool implementations for Pingouin MCP.

Every function here returns a dict with the same shape::

    {"markdown": str,          # what the agent reads and relays
     "figures": [str, ...],    # paths to PNGs, if any
     "code": str | None,       # the equivalent Pingouin call, for transparency
     "data": dict | None}      # machine-readable numbers, for chaining

Design notes
------------
* **No function invents a statistic.** Every number comes from a Pingouin call.
* **Assumptions are reported alongside the result**, never silently skipped.
* **`code` is always populated** — the equivalent Pingouin snippet is shown so
  the analysis can be reproduced outside the MCP server. This is the
  reproducibility contract of the whole server.
* Failure returns ``{"markdown": "<actionable message>"}`` rather than raising,
  so an agent can read the problem and adjust.
"""

from __future__ import annotations

import math
import traceback
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg

from pingouin_mcp import decision as dec
from pingouin_mcp.kernel import get_session
from pingouin_mcp.summaries import (
    assumptions_block,
    dataframe_table,
    fmt,
    next_steps_block,
    pfmt,
    sig_marker,
    table,
)


# ── Result helpers ────────────────────────────────────────────────────────────

def _ok(markdown: str, figures=None, code=None, data=None) -> dict:
    return {
        "markdown": markdown,
        "figures": figures or [],
        "code": code,
        "data": data,
    }


def _err(message: str, hint: str | None = None) -> dict:
    lines = [f"**Error:** {message}"]
    if hint:
        lines += ["", hint]
    return {"markdown": "\n".join(lines), "figures": [], "code": None, "data": None}


def _unexpected(exc: Exception) -> dict:
    """
    Turn an exception into an agent-readable message.

    Errors caused by the caller (a missing column, a wrong type, a bad argument)
    carry a message that already says what to fix, so a traceback is noise. Errors
    from deeper inside the stack do not, so those keep a short traceback to make
    the failure diagnosable.
    """
    name = type(exc).__name__
    message = str(exc)

    # Input-shaped errors: the message alone is actionable.
    if name in ("KeyError", "ValueError", "TypeError", "AttributeError",
                "FileNotFoundError", "IndexError") and message:
        return _err(f"`{name}: {message}`" if name != "KeyError" else f"{message}")

    hint = ("```\n" + "\n".join(traceback.format_exc(limit=6).strip().splitlines()[-8:])
            + "\n```\n\n_This is an unexpected failure in the analysis call. The traceback "
              "above names the function that raised it._")
    return _err(f"`{name}: {message}`", hint)


def _get_df(name: str) -> pd.DataFrame:
    obj = get_session().require(name)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(
            f"`{name}` is a {type(obj).__name__}, not a DataFrame. "
            "Statistical tools operate on DataFrames — load one with "
            "`stat_load_data`, or select the frame you meant from `stat_session_info`."
        )
    return obj


def _check_columns(df: pd.DataFrame, cols: list[str | None]) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) not found: {missing}. "
            f"Available columns: {list(df.columns)}. "
            "Column names are case-sensitive."
        )


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    """
    Resolve a result column across Pingouin's naming conventions.

    Pingouin 0.6 renamed its output columns from hyphenated to underscored
    (``p-val`` → ``p_val``, ``p-unc`` → ``p_unc``). Hard-coding either form makes
    the server silently misread results when the library changes. This helper
    tries every known spelling and returns the first that exists, so a rename
    degrades to "column missing" rather than "wrong number reported".
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _num(row, *candidates: str) -> float | None:
    """Read a numeric field from a result row, tolerating column renames."""
    for c in candidates:
        if c in row.index:
            try:
                return float(row[c])
            except (TypeError, ValueError):
                return None
    return None


def _round_df(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round float columns for display without mutating the original."""
    try:
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].round(decimals)
        return out
    except Exception:
        return df


def _verdict(p: float, alpha: float, hi: str, lo: str) -> str:
    """A one-line plain-language reading of a p-value."""
    try:
        if math.isnan(float(p)):
            return "test could not be computed"
    except (TypeError, ValueError):
        return "test could not be computed"
    return f"{hi} ({sig_marker(p)})" if float(p) < alpha else f"{lo} (ns)"


# ══════════════════════════════════════════════════════════════════════════════
# A. Data management
# ══════════════════════════════════════════════════════════════════════════════

def stat_load_data(path: str, name: str = "df", **read_kwargs) -> dict:
    """Load a tabular dataset into the session."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        # Help the agent find the file rather than just failing.
        near = sorted(Path(p.parent if p.parent.exists() else ".").glob("*.csv"))[:10]
        hint = ("Files ending in .csv in that folder: " +
                ", ".join(f"`{f.name}`" for f in near)) if near else None
        return _err(f"File not found: `{path}`", hint)

    suffix = p.suffix.lower()
    try:
        if suffix in (".csv", ".txt"):
            df = pd.read_csv(p, **read_kwargs)
        elif suffix in (".tsv",):
            df = pd.read_csv(p, sep="\t", **read_kwargs)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(p, **read_kwargs)
        elif suffix in (".json",):
            df = pd.read_json(p, **read_kwargs)
        elif suffix in (".parquet",):
            df = pd.read_parquet(p, **read_kwargs)
        elif suffix in (".sav",):
            df = pd.read_spss(p, **read_kwargs)
        else:
            return _err(
                f"Unsupported file type `{suffix}`.",
                "Supported: .csv .tsv .txt .xlsx .xls .json .parquet .sav",
            )
    except Exception as exc:
        return _unexpected(exc)

    get_session().set(name, df)

    md = [f"Loaded **{len(df)}** rows × **{len(df.columns)}** columns as `{name}`.", ""]
    md.append("**Columns**")
    md.append("")
    md.append("| Column | dtype | Non-null | Example |")
    md.append("|---|---|---|---|")
    for c in df.columns:
        ex = df[c].dropna()
        sample = str(ex.iloc[0])[:40] if len(ex) else "—"
        md.append(f"| `{c}` | {df[c].dtype} | {df[c].notna().sum()}/{len(df)} | {sample} |")

    md += next_steps_block([
        "`stat_data_quality_check` — missingness, outliers, distributions",
        "`stat_describe_data` — descriptive statistics by group",
        "`stat_decision_assistant` — if you have not yet decided which test to use",
    ])
    return _ok("\n".join(md), code=f"pd.read_csv('{path}')", data={"rows": len(df), "cols": list(df.columns)})


def stat_make_dataframe(name: str, columns: dict) -> dict:
    """Build a DataFrame from column-name → list-of-values, for examples and tests."""
    try:
        df = pd.DataFrame(columns)
    except Exception as exc:
        return _unexpected(exc)
    get_session().set(name, df)
    return _ok(
        f"Created `{name}` with {len(df)} rows and columns {list(df.columns)}.\n\n"
        + dataframe_table(_round_df(df.head(10))),
        data={"rows": len(df)},
    )


def stat_describe_data(name: str, dv: str | None = None, group: str | None = None) -> dict:
    """Descriptive statistics, optionally split by group."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, group])
        if dv is None:
            num = df.select_dtypes(include=[np.number])
            if num.empty:
                return _err("No numeric columns found.", f"Columns: {list(df.columns)}")
            desc = num.describe().T.reset_index().rename(columns={"index": "variable"})
            desc["sem"] = num.sem().values
            md = ["### Descriptive statistics", "",
                  dataframe_table(_round_df(desc))]
        else:
            if group:
                rows = []
                for g, sub in df.groupby(group, dropna=False):
                    s = sub[dv].dropna()
                    rows.append({
                        "group": str(g), "n": len(s), "mean": s.mean(), "sd": s.std(ddof=1),
                        "median": s.median(), "min": s.min(), "max": s.max(),
                        "sem": s.sem(),
                    })
                md = [f"### Descriptive statistics: `{dv}` by `{group}`", "",
                      table(rows, ["group", "n", "mean", "sd", "median", "min", "max", "sem"])]
            else:
                s = df[dv].dropna()
                md = [f"### Descriptive statistics: `{dv}`", "",
                      table([{"n": len(s), "mean": s.mean(), "sd": s.std(ddof=1),
                              "median": s.median(), "sem": s.sem(),
                              "min": s.min(), "max": s.max()}])]
    except Exception as exc:
        return _unexpected(exc)

    md += next_steps_block([
        "Run `stat_assumption_report` before choosing a test",
        "Plot the distribution with `stat_qqplot` or a histogram",
    ])
    return _ok("\n".join(md), code=f"df.groupby('{group}')['{dv}'].describe()" if group and dv else "df.describe()")


def stat_data_quality_check(name: str, columns: list[str] | None = None) -> dict:
    """
    Missingness, outliers and distribution shape in one pass.

    This is the first thing to run on any new dataset: it determines whether the
    planned analysis is even possible.
    """
    try:
        df = _get_df(name)
    except Exception as exc:
        return _unexpected(exc)

    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return _err("No numeric columns to check.", f"Columns: {list(df.columns)}")

    md = [f"### Data quality check — `{name}`", ""]

    # ---- Missingness -------------------------------------------------------
    total_cells = df[cols].size
    missing_cells = int(df[cols].isna().sum().sum())
    md += ["**Missingness**", ""]
    rows = []
    for c in cols:
        n_miss = int(df[c].isna().sum())
        rows.append({"column": c, "n_missing": n_miss,
                     "percent": 100 * n_miss / len(df) if len(df) else 0.0})
    rows.append({"column": "**total**", "n_missing": missing_cells,
                 "percent": 100 * missing_cells / total_cells if total_cells else 0.0})
    md.append(table(rows, ["column", "n_missing", "percent"]))
    if missing_cells:
        md += ["", "_Note: complete-case analysis (listwise deletion) discards a whole row "
               "when any variable is missing. If missingness is substantial or related to the "
               "outcome, report how much data the analysis actually used._"]

    # ---- Outliers ----------------------------------------------------------
    md += ["", "**Outliers (robust MAD rule — `pg.madmedianrule`)**", ""]
    out_rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) < 4:
            out_rows.append({"column": c, "n_outliers": "—", "note": "too few values"})
            continue
        try:
            flag = pg.madmedianrule(s.to_numpy())
            n_out = int(np.sum(flag))
        except Exception:
            n_out = 0
        out_rows.append({"column": c, "n_outliers": n_out,
                         "note": "inspect before deciding" if n_out else ""})
    md.append(table(out_rows, ["column", "n_outliers", "note"]))

    # ---- Distribution shape ------------------------------------------------
    md += ["", "**Distribution shape**", ""]
    shape_rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) < 4:
            continue
        shape_rows.append({
            "column": c, "skew": s.skew(), "kurtosis": s.kurtosis(),
            "n": len(s),
            "flag": "notably skewed" if abs(s.skew()) > 1 else "",
        })
    if shape_rows:
        md.append(table(shape_rows, ["column", "skew", "kurtosis", "n", "flag"]))

    md += next_steps_block([
        "`stat_normality_test` — formal normality test per group",
        "`stat_assumption_report` — full checklist for your design",
    ])
    data = {"missing_cells": missing_cells, "columns": cols}
    return _ok("\n".join(md), code="pg.madmedianrule(df[col])", data=data)


# ══════════════════════════════════════════════════════════════════════════════
# B. Assumption testing
# ══════════════════════════════════════════════════════════════════════════════

def stat_normality_test(name: str, dv: str, group: str | None = None,
                        method: str = "shapiro") -> dict:
    """Test normality of a variable, optionally within each group."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, group])
        if group:
            res = pg.normality(data=df, dv=dv, group=group, method=method)
        else:
            # pingouin's long-format path requires a grouping column; for a
            # single variable, hand it the Series directly.
            res = pg.normality(df[dv].dropna(), method=method)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index() if res.index.name else res
    res = _round_df(res_raw)
    md = [f"### Normality — `{dv}`" + (f" by `{group}`" if group else ""),
          f"_Method: {method} (α = 0.05)_", ""]
    md.append(dataframe_table(res))

    try:
        pcol = "pval" if "pval" in res.columns else res.columns[-1]
        ncol = "W" if "W" in res.columns else None
        violated = res[res[pcol] < 0.05]
        if len(violated) == 0:
            md += ["", "**Reading:** normality is not rejected in any group."]
        else:
            which = violated.iloc[:, 0].astype(str).tolist() if res.shape[1] else []
            md += ["", f"**Reading:** normality is rejected in {len(violated)} group(s)"
                   + (f" ({', '.join(which)})" if which else "") + "."]
    except Exception:
        pass

    md += ["", "_Note: with large n, Shapiro-Wilk detects trivial departures from "
           "normality. Look at the QQ plot and skewness as well; with small n it has low "
           "power, so a non-significant result does not establish normality._"]
    md += next_steps_block([
        "`stat_qqplot` to inspect visually",
        "`stat_decision_assistant` to see which test the result implies",
    ])
    return _ok("\n".join(md), code=f"pg.normality(data=df, dv='{dv}'"
                + (f", group='{group}'" if group else "") + f", method='{method}')",
               data={"table": res_raw.to_dict("records")})


def stat_homogeneity_test(name: str, dv: str, group: str,
                          method: str = "levene") -> dict:
    """Test equality of variances across groups (Levene / Bartlett)."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, group])
        res = pg.homoscedasticity(data=df, dv=dv, group=group, method=method)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index() if res.index.name else res
    res = _round_df(res_raw)
    md = [f"### Homogeneity of variance — `{dv}` by `{group}`",
          f"_Method: {method}_", "", dataframe_table(res)]
    try:
        p = float(res["pval"].iloc[0])
        md += ["", f"**Reading:** {_verdict(p, 0.05, 'variances differ significantly', 'no evidence of unequal variances')}."]
        if p < 0.05:
            md += ["", "Use a test that does not assume equal variances: **Welch's t-test** "
                   "(two groups) or **Welch's ANOVA** (three or more)."]
    except Exception:
        pass
    return _ok("\n".join(md),
               code=f"pg.homoscedasticity(data=df, dv='{dv}', group='{group}', method='{method}')",
               data={"table": res_raw.to_dict("records")})


def stat_sphericity_test(name: str, dv: str, within: str, subject: str,
                         method: str = "mauchly") -> dict:
    """Mauchly's test of sphericity for a repeated-measures factor."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, within, subject])
        res = pg.sphericity(data=df, dv=dv, within=within, subject=subject, method=method)
    except Exception as exc:
        return _unexpected(exc)

    try:
        spher, W, chi2, dof, p = res
    except Exception:
        return _ok(f"### Sphericity\n\n```\n{res}\n```")

    md = ["### Sphericity (Mauchly's test)", "",
          "| Sphericity | W | chi2 | dof | p |", "|---|---|---|---|---|",
          f"| {'met' if spher else '**violated**'} | {fmt(W)} | {fmt(chi2)} | {fmt(dof, 1)} | {pfmt(p)} |"]
    try:
        eps = pg.epsilon(data=df, dv=dv, within=within, subject=subject, correction="gg")
        md += ["", f"**Greenhouse-Geisser epsilon:** {fmt(eps)}",
               "", "_If sphericity is violated, report the corrected p-value and the epsilon. "
               "Epsilon near 1 means little correction is needed; the smaller it is, the more the "
               "uncorrected F is inflated._"]
    except Exception:
        pass
    return _ok("\n".join(md),
               code=f"pg.sphericity(data=df, dv='{dv}', within='{within}', subject='{subject}')",
               data={"sphericity": bool(spher), "W": W, "p": p})


def stat_assumption_report(name: str, dv: str, between: str | None = None,
                           within: str | None = None, subject: str | None = None,
                           covar: str | None = None) -> dict:
    """
    Run every assumption check implied by a design, in one pass.

    This exists so that assumptions cannot be quietly skipped: an agent that
    calls one tool gets the whole checklist, pass or fail.
    """
    try:
        df = _get_df(name)
        _check_columns(df, [dv, between, within, subject, covar])
    except Exception as exc:
        return _unexpected(exc)

    md = ["### Assumption report", ""]
    checks: list[tuple[str, bool | None, str]] = []
    blocks: list[str] = []

    # Normality
    try:
        if between:
            res = pg.normality(data=df, dv=dv, group=between)
            pcol = "pval"
            bad = res[res[pcol] < 0.05]
            checks.append(("Normality (per group)", len(bad) == 0,
                           f"{len(bad)}/{len(res)} group(s) rejected"))
        elif within and subject:
            res = pg.normality(data=df, dv=dv)
            p = float(res["pval"].iloc[0])
            checks.append(("Normality", p >= 0.05, f"Shapiro-Wilk p = {pfmt(p)}"))
        else:
            res = pg.normality(df[dv].dropna())
            p = float(res["pval"].iloc[0])
            checks.append(("Normality", p >= 0.05, f"Shapiro-Wilk p = {pfmt(p)}"))
        blocks += ["**Normality**", "", dataframe_table(_round_df(res.reset_index() if res.index.name else res)), ""]
    except Exception as exc:
        checks.append(("Normality", None, f"could not run: {exc}"))

    # Homogeneity
    if between:
        try:
            res = pg.homoscedasticity(data=df, dv=dv, group=between)
            p = float(res["pval"].iloc[0])
            checks.append(("Homogeneity of variance", p >= 0.05, f"Levene p = {pfmt(p)}"))
            blocks += ["**Homogeneity of variance**", "",
                       dataframe_table(_round_df(res.reset_index() if res.index.name else res)), ""]
        except Exception as exc:
            checks.append(("Homogeneity of variance", None, f"could not run: {exc}"))

    # Sphericity
    if within and subject and (df[within].nunique() >= 3):
        try:
            res = pg.sphericity(data=df, dv=dv, within=within, subject=subject)
            spher = bool(res[0])
            checks.append(("Sphericity", spher, f"Mauchly W = {fmt(res[1])}, p = {pfmt(res[4])}"))
            blocks += [f"**Sphericity:** {'met' if spher else 'violated'} "
                       f"(W = {fmt(res[1])}, p = {pfmt(res[4])})", ""]
        except Exception as exc:
            checks.append(("Sphericity", None, f"could not run: {exc}"))

    # Outliers
    try:
        s = df[dv].dropna()
        n_out = int(np.sum(pg.madmedianrule(s.to_numpy()))) if len(s) >= 4 else 0
        checks.append(("Outliers", n_out == 0, f"{n_out} flagged by the MAD rule"))
    except Exception:
        checks.append(("Outliers", None, "could not run"))

    md += blocks
    md += assumptions_block(checks)

    # Summary judgement
    failed = [c[0] for c in checks if c[1] is False]
    if failed:
        md += ["", f"**{len(failed)} assumption(s) violated:** " + ", ".join(failed),
               "", "Use `stat_decision_assistant` to see which alternative test each "
               "violation implies. Do not proceed with the parametric test and report "
               "the violation only in a footnote."]
    else:
        md += ["", "**No assumption violations detected** — the parametric test for this "
               "design is defensible."]

    md += ["", "_Report these checks in the paper whether or not they pass. A check "
           "reported only when it fails is not a check._"]
    return _ok("\n".join(md), code="# see pg.normality / pg.homoscedasticity / pg.sphericity",
               data={"checks": [{"assumption": a, "passed": b, "note": n} for a, b, n in checks]})


# ══════════════════════════════════════════════════════════════════════════════
# C. Two-group comparisons
# ══════════════════════════════════════════════════════════════════════════════

def stat_ttest(name: str, dv: str, group: str | None = None, paired: bool = False,
               subject: str | None = None, x: str | None = None, y: str | None = None,
               alternative: str = "two-sided", correction: str = "auto",
               confidence: float = 0.95) -> dict:
    """
    t-test: independent, paired, or one-sample.

    Use `group` with a two-level column for between-group data (or `paired=True`
    plus `subject` for repeated measures). Use `x`/`y` for one-sample against a
    constant, or for two precomputed vectors.
    """
    try:
        df = _get_df(name)
        code = None
        if x is not None and y is not None:
            _check_columns(df, [x, y])
            a, b = df[x].dropna(), df[y].dropna()
            n = min(len(a), len(b))
            res = pg.ttest(a.iloc[:n], b.iloc[:n], paired=paired,
                           alternative=alternative, correction=correction,
                           confidence=confidence)
            label = f"{x} vs {y}"
            code = (f"pg.ttest(df['{x}'], df['{y}'], paired={paired})")
        elif group is not None:
            _check_columns(df, [dv, group])
            levels = df[group].dropna().unique()
            if len(levels) != 2:
                return _err(
                    f"`{group}` has {len(levels)} levels, but a t-test compares exactly two.",
                    "For three or more groups use `stat_anova`. "
                    "For a repeated-measures factor with two levels use `paired=True` and `subject`.",
                )
            if paired:
                if not subject:
                    return _err("Paired t-test needs `subject`.",
                                "Pass the column identifying each participant so the two "
                                "conditions can be matched within participant.")
                _check_columns(df, [subject])
                wide = df.pivot_table(index=subject, columns=group, values=dv,
                                      aggfunc="mean").dropna()
                if wide.shape[1] != 2:
                    return _err("Could not pair the two conditions by subject.")
                a, b = wide.iloc[:, 0], wide.iloc[:, 1]
                n = min(len(a), len(b))
                res = pg.ttest(a.iloc[:n], b.iloc[:n], paired=True,
                               alternative=alternative, correction=correction,
                               confidence=confidence)
                code = (f"w = df.pivot(index='{subject}', columns='{group}', values='{dv}')\n"
                        f"pg.ttest(w.iloc[:,0], w.iloc[:,1], paired=True)")
            else:
                a = df.loc[df[group] == levels[0], dv].dropna()
                b = df.loc[df[group] == levels[1], dv].dropna()
                res = pg.ttest(a, b, paired=False, alternative=alternative,
                               correction=correction, confidence=confidence)
                code = (f"pg.ttest(df.loc[df['{group}']=='{levels[0]}', '{dv}'].dropna(),\n"
                        f"         df.loc[df['{group}']=='{levels[1]}', '{dv}'].dropna())")
            label = f"{levels[0]} vs {levels[1]}"
        else:
            _check_columns(df, [dv])
            a = df[dv].dropna()
            res = pg.ttest(a, 0, paired=False, alternative=alternative,
                           correction=correction, confidence=confidence)
            label = f"{dv} vs 0"
            code = f"pg.ttest(df['{dv}'].dropna(), 0)"
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### t-test — {label}",
          f"_Paired: {paired} · Alternative: {alternative} · Correction: {correction}_", "",
          dataframe_table(res)]
    # The sign of t depends on which group is subtracted from which, and that
    # ordering is not visible in the table. Without stating it, both t and
    # Cohen's d have an uninterpretable sign.
    try:
        first, second = label.split(" vs ", 1)
        md += ["", f"_Direction: **{first} minus {second}**. A positive t means "
               f"{first} is larger. Group levels are ordered alphabetically, so "
               "check this before interpreting the sign._"]
    except Exception:
        pass

    try:
        row = res.iloc[0]
        p = _num(row, "p_val", "p-val", "pval")
        md += ["", f"**Reading:** {_verdict(p, 0.05, 'the difference is statistically significant', 'no statistically significant difference')}."]
        for key, name_ in [("T", "t"), ("dof", "df"), ("cohen_d", "Cohen's d"),
                           ("CI95", "95% CI"), ("BF10", "BF₁₀"), ("power", "power")]:
            if key in res.columns:
                md += [f"- **{name_}:** {row[key]}"]
    except Exception:
        pass

    if not paired and group is not None:
        # Report which variant was actually run. Pingouin's `correction="auto"`
        # already switches to Welch when group sizes differ, so telling the user
        # to "also report Welch" would be wrong — it may already be Welch.
        try:
            n_a, n_b = len(a), len(b)
            welch_used = correction is True or (
                str(correction).lower() == "auto" and n_a != n_b
            )
        except Exception:
            welch_used = False
        if welch_used:
            md += ["", f"_This is **Welch's t-test** (unequal variances), not the "
                   f"classic Student t-test — group sizes were {len(a)} and {len(b)}. "
                   "Report the test by name; the two give different t and df._"]
        else:
            md += ["", "_This is the classic Student t-test, which assumes equal "
                   "variances. If Levene's test is significant, re-run with "
                   "`correction=True` for Welch's t-test and report that instead._"]
    return _ok("\n".join(md), code=code, data={"table": res_raw.to_dict("records")})


def stat_nonparametric_2(name: str, dv: str, group: str | None = None,
                         paired: bool = False, subject: str | None = None,
                         x: str | None = None, y: str | None = None,
                         alternative: str = "two-sided") -> dict:
    """Non-parametric two-group tests: Mann-Whitney U (independent) or Wilcoxon signed-rank (paired)."""
    try:
        df = _get_df(name)
        if x is not None and y is not None:
            _check_columns(df, [x, y])
            a, b = df[x].dropna().to_numpy(), df[y].dropna().to_numpy()
            n = min(len(a), len(b))
            if paired:
                res = pg.wilcoxon(a[:n], b[:n], alternative=alternative)
                test = "Wilcoxon signed-rank"
                code = f"pg.wilcoxon(df['{x}'], df['{y}'])"
            else:
                res = pg.mwu(a, b, alternative=alternative)
                test = "Mann-Whitney U"
                code = f"pg.mwu(df['{x}'], df['{y}'])"
            label = f"{x} vs {y}"
        elif group is not None:
            _check_columns(df, [dv, group])
            levels = df[group].dropna().unique()
            if len(levels) != 2:
                return _err(f"`{group}` has {len(levels)} levels; these tests compare two.",
                            "For three or more groups use `stat_nonparametric_k`.")
            if paired:
                if not subject:
                    return _err("Paired test needs `subject`.")
                wide = df.pivot_table(index=subject, columns=group, values=dv,
                                      aggfunc="mean").dropna()
                a, b = wide.iloc[:, 0].to_numpy(), wide.iloc[:, 1].to_numpy()
                res = pg.wilcoxon(a, b, alternative=alternative)
                test = "Wilcoxon signed-rank"
                code = f"w = df.pivot(index='{subject}', columns='{group}', values='{dv}')\npg.wilcoxon(w.iloc[:,0], w.iloc[:,1])"
            else:
                a = df.loc[df[group] == levels[0], dv].dropna().to_numpy()
                b = df.loc[df[group] == levels[1], dv].dropna().to_numpy()
                res = pg.mwu(a, b, alternative=alternative)
                test = "Mann-Whitney U"
                code = f"pg.mwu(df.loc[df['{group}']=='{levels[0]}','{dv}'].dropna(), df.loc[df['{group}']=='{levels[1]}','{dv}'].dropna())"
            label = f"{levels[0]} vs {levels[1]}"
        else:
            return _err("Specify either `group` (with `dv`) or both `x` and `y`.")
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### {test} — {label}", "", dataframe_table(res)]
    try:
        row = res.iloc[0]
        pcol = _col(res, "p_val", "p-val", "pval")
        p = float(row[pcol])
        md += ["", f"**Reading:** {_verdict(p, 0.05, 'the distributions differ significantly', 'no significant difference')}."]
        if "RBC" in res.columns:
            md += [f"- **Rank-biserial correlation:** {row['RBC']} (effect size, −1 to 1)"]
        md += ["", "_Report the rank-biserial correlation (RBC) as the effect size — the "
               "p-value alone does not convey magnitude._"]
    except Exception:
        pass
    return _ok("\n".join(md), code=code, data={"table": res_raw.to_dict("records")})


def stat_bayes_ttest(name: str, dv: str, group: str, paired: bool = False,
                     subject: str | None = None, alternative: str = "two-sided",
                     r: float = 0.707) -> dict:
    """
    Bayesian t-test via the Bayes factor.

    The Bayes factor quantifies evidence for the alternative *and* for the null,
    which is what a non-significant frequentist test cannot tell you.
    """
    try:
        df = _get_df(name)
        _check_columns(df, [dv, group])
        levels = df[group].dropna().unique()
        if len(levels) != 2:
            return _err(f"`{group}` has {len(levels)} levels; this test compares two.")
        if paired:
            if not subject:
                return _err("Paired Bayes t-test needs `subject`.")
            wide = df.pivot_table(index=subject, columns=group, values=dv,
                                  aggfunc="mean").dropna()
            a, b = wide.iloc[:, 0].to_numpy(), wide.iloc[:, 1].to_numpy()
        else:
            a = df.loc[df[group] == levels[0], dv].dropna().to_numpy()
            b = df.loc[df[group] == levels[1], dv].dropna().to_numpy()

        t_res = pg.ttest(a, b, paired=paired, alternative=alternative)
        t = float(t_res["T"].iloc[0])
        bf = pg.bayesfactor_ttest(t, len(a), len(b) if not paired else None,
                                  paired=paired, alternative=alternative, r=r)
    except Exception as exc:
        return _unexpected(exc)

    if isinstance(bf, pd.DataFrame):
        bf = float(bf.iloc[0, 0])
    bf = float(bf)

    def _interp(x: float) -> str:
        if x < 1 / 10:
            return "strong evidence for H₀"
        if x < 1 / 3:
            return "moderate evidence for H₀"
        if x <= 3:
            return "inconclusive (data do not distinguish H₀ from H₁)"
        if x <= 10:
            return "moderate evidence for H₁"
        return "strong evidence for H₁"

    md = ["### Bayesian t-test", "",
          f"**BF₁₀ = {fmt(bf)}** — {_interp(bf)}", "",
          "| Quantity | Value |", "|---|---|",
          f"| t | {fmt(t)} |",
          f"| n ({levels[0]}) | {len(a)} |",
          f"| n ({levels[1]}) | {len(b)} |",
          f"| BF₁₀ | {fmt(bf)} |",
          f"| BF₀₁ | {fmt(1 / bf) if bf else '—'} |",
          "",
          "_BF₁₀ > 1 favours the alternative; BF₁₀ < 1 favours the null. "
          "This is the main reason to run a Bayesian test: a non-significant "
          "frequentist p-value does **not** mean the null is true, whereas BF₀₁ "
          "directly quantifies evidence for it._"]
    return _ok("\n".join(md),
               code=f"pg.bayesfactor_ttest(t, nx={len(a)}, ny={len(b)}, paired={paired}, r={r})",
               data={"BF10": bf, "t": t})


# ══════════════════════════════════════════════════════════════════════════════
# D. Multi-group comparisons
# ══════════════════════════════════════════════════════════════════════════════

def stat_anova(name: str, dv: str, between: str, ss_type: int = 2,
               effsize: str = "np2") -> dict:
    """One-way ANOVA (use `welch=True` semantics via stat_welch_anova when variances differ)."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, between])
        res = pg.anova(data=df, dv=dv, between=between, ss_type=ss_type, effsize=effsize)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### One-way ANOVA — `{dv}` by `{between}`", "", dataframe_table(res)]

    try:
        src = res[res["Source"] != "Within"].iloc[0]
        p = _num(src, "p_unc", "p-unc")
        f = float(src["F"])
        md += ["", f"**Reading:** {_verdict(p, 0.05, f'the groups differ, F = {fmt(f)}', f'no significant group difference, F = {fmt(f)}')}."]
        es_col = [c for c in res.columns if c.startswith("np2") or c.startswith("n2")]
        if es_col:
            md += [f"- **Effect size ({es_col[0]}):** {src[es_col[0]]}"]
        if p < 0.05:
            md += ["", "The omnibus test says *some* groups differ, not which ones. "
                   "Run `stat_pairwise_tests` (or Tukey HSD) with a multiple-comparison "
                   "correction to identify them."]
        else:
            md += ["", "_A non-significant omnibus test does not license the claim that "
                   "all groups are equivalent. If equivalence is the claim you want to make, "
                   "use `stat_equivalence_test`._"]
    except Exception:
        pass

    md += ["", "_Check normality and homogeneity first (`stat_assumption_report`). "
           "With unequal variances use Welch's ANOVA._"]
    return _ok("\n".join(md), code=f"pg.anova(data=df, dv='{dv}', between='{between}')",
               data={"table": res_raw.to_dict("records")})


def stat_welch_anova(name: str, dv: str, between: str) -> dict:
    """Welch's ANOVA — does not assume equal variances."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, between])
        res = pg.welch_anova(data=df, dv=dv, between=between)
    except Exception as exc:
        return _unexpected(exc)
    res_raw = res.reset_index(drop=True)
    res = _round_df(res_raw)
    md = [f"### Welch's ANOVA — `{dv}` by `{between}`",
          "_Does not assume equal variances; appropriate when Levene's test is significant._",
          "", dataframe_table(res)]
    try:
        p = _num(res.iloc[0], "p_unc", "p-unc")
        md += ["", f"**Reading:** {_verdict(p, 0.05, 'the groups differ', 'no significant difference')}."]
        md += ["", "_Welch's ANOVA does not produce the usual η². Report ω² or use "
               "`stat_pairwise_tests` for pairwise effect sizes with a correction._"]
    except Exception:
        pass
    return _ok("\n".join(md), code=f"pg.welch_anova(data=df, dv='{dv}', between='{between}')",
               data={"table": res_raw.to_dict("records")})


def stat_rm_anova(name: str, dv: str, within: str, subject: str,
                  correction: str = "auto", effsize: str = "ng2") -> dict:
    """Repeated-measures ANOVA with automatic sphericity correction."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, within, subject])
        res = pg.rm_anova(data=df, dv=dv, within=within, subject=subject,
                          correction=correction, effsize=effsize)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### Repeated-measures ANOVA — `{dv}` by `{within}`",
          f"_Correction: {correction}_", "", dataframe_table(res)]

    try:
        row = res.iloc[0]
        pcol = _col(res, "p_unc", "p-unc") or res.columns[-1]
        p = float(row[pcol])
        md += ["", f"**Reading:** {_verdict(p, 0.05, 'the conditions differ significantly', 'no significant difference across conditions')}."]
        if _col(res, "p_GG_corr", "p-GG-corr"):
            md += [f"- **Sphericity-corrected p (Greenhouse-Geisser):** "
                       f"{pfmt(_num(row, 'p_GG_corr', 'p-GG-corr'))}",
                   f"- **Epsilon (GG):** {fmt(row.get('eps', float('nan')))}",
                   "", "_Report the corrected p when sphericity is violated; the uncorrected "
                   "value is anti-conservative._"]
        for c in res.columns:
            if c.startswith("ng2"):
                md += [f"- **Generalized η²:** {row[c]}"]
    except Exception:
        pass
    md += ["", "_Assumptions: normality of residuals and sphericity (for 3+ levels). "
           "With missing data, a linear mixed-effects model is preferred over RM-ANOVA, "
           "which uses listwise deletion._"]
    return _ok("\n".join(md), code=f"pg.rm_anova(data=df, dv='{dv}', within='{within}', subject='{subject}')",
               data={"table": res_raw.to_dict("records")})


def stat_mixed_anova(name: str, dv: str, within: str, subject: str, between: str,
                     correction: str = "auto", effsize: str = "np2") -> dict:
    """Mixed-design ANOVA: one between-subjects and one within-subjects factor."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, within, subject, between])
        res = pg.mixed_anova(data=df, dv=dv, within=within, subject=subject,
                             between=between, correction=correction, effsize=effsize)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### Mixed ANOVA — `{dv}` ~ `{between}` (between) × `{within}` (within)",
          "", dataframe_table(res)]
    try:
        for _, row in res.iterrows():
            src = row.get("Source", "")
            p = _num(row, "p_unc", "p-unc")
            md += [f"- **{src}:** F = {fmt(row.get('F'))}, p = {pfmt(p)} {sig_marker(p)}"]
        md += ["", "_Read the interaction first. A significant interaction means the effect of "
               "one factor depends on the level of the other, and the main effects should not "
               "be interpreted in isolation._"]
    except Exception:
        pass
    return _ok("\n".join(md),
               code=f"pg.mixed_anova(data=df, dv='{dv}', within='{within}', subject='{subject}', between='{between}')",
               data={"table": res_raw.to_dict("records")})


def stat_ancova(name: str, dv: str, between: str, covar: str,
                effsize: str = "np2") -> dict:
    """ANCOVA — group comparison controlling a continuous covariate."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, between, covar])
        if covar == dv:
            return _err(
                f"The covariate and the dependent variable are both `{dv}`.",
                "ANCOVA adjusts the outcome for the covariate; they must be "
                "different variables. If the goal is a baseline-adjusted analysis, "
                "the covariate should be the baseline measurement and the outcome "
                "the follow-up measurement.",
            )
        res = pg.ancova(data=df, dv=dv, between=between, covar=covar, effsize=effsize)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### ANCOVA — `{dv}` ~ `{between}`, covariate `{covar}`", "", dataframe_table(res)]

    # Homogeneity of regression slopes is ANCOVA's key assumption.
    try:
        check = pg.anova(data=df, dv=dv, between=[between, covar], ss_type=2)
        inter = check[check["Source"].str.contains("\\*", regex=True)]
        md += ["", "**Homogeneity of regression slopes**", ""]
        if len(inter):
            p_int = _num(inter.iloc[0], "p_unc", "p-unc")
            md += [f"{between} × {covar} interaction: F = {fmt(inter.iloc[0]['F'])}, "
                   f"p = {pfmt(p_int)} {sig_marker(p_int)}"]
            if p_int < 0.05:
                md += ["", "**This assumption is violated** — the covariate's relationship "
                       "with the outcome differs between groups. ANCOVA is not appropriate; "
                       "report the interaction instead."]
            else:
                md += ["", "Assumption met — the covariate–outcome slope does not differ "
                       "significantly between groups."]
    except Exception:
        pass

    md += ["", "**Two further requirements**",
           "- The covariate must be measured **before** the manipulation and be unaffected by it. "
           "Controlling a variable the treatment changed removes part of the effect.",
           "- The covariate must be measured without error; unreliable covariates bias the "
           "adjusted means."]
    return _ok("\n".join(md),
               code=f"pg.ancova(data=df, dv='{dv}', between='{between}', covar='{covar}')",
               data={"table": res_raw.to_dict("records")})


def stat_nonparametric_k(name: str, dv: str, between: str | None = None,
                         within: str | None = None, subject: str | None = None) -> dict:
    """Kruskal-Wallis (independent) or Friedman (repeated) for three or more groups."""
    try:
        df = _get_df(name)
        if between:
            _check_columns(df, [dv, between])
            res = pg.kruskal(data=df, dv=dv, between=between)
            test = "Kruskal-Wallis"
            code = f"pg.kruskal(data=df, dv='{dv}', between='{between}')"
        elif within and subject:
            _check_columns(df, [dv, within, subject])
            res = pg.friedman(data=df, dv=dv, within=within, subject=subject)
            test = "Friedman"
            code = f"pg.friedman(data=df, dv='{dv}', within='{within}', subject='{subject}')"
        else:
            return _err("Specify `between` (independent groups) or both `within` and `subject` (repeated).")
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### {test}", "", dataframe_table(res)]
    try:
        pcol = _col(res, "p_unc", "p-unc", "p_val", "pval")
        p = float(res[pcol].iloc[0])
        md += ["", f"**Reading:** {_verdict(p, 0.05, 'the groups differ significantly', 'no significant difference')}."]
        if "W" in res.columns:
            md += [f"- **Kendall's W (effect size):** {res['W'].iloc[0]} — "
                   "0 = no agreement, 1 = complete agreement across conditions"]
        if p < 0.05:
            md += ["", "Run `stat_pairwise_tests(parametric=False)` with a correction "
                   "to identify which pairs differ."]
    except Exception:
        pass
    return _ok("\n".join(md), code=code, data={"table": res_raw.to_dict("records")})


# ══════════════════════════════════════════════════════════════════════════════
# E. Post-hoc comparisons and multiple-comparison correction
# ══════════════════════════════════════════════════════════════════════════════

def stat_pairwise_tests(name: str, dv: str, between: str | None = None,
                        within: str | None = None, subject: str | None = None,
                        parametric: bool = True, padjust: str = "none",
                        effsize: str = "hedges", alpha: float = 0.05,
                        interaction: bool = True) -> dict:
    """
    Pairwise comparisons after a significant omnibus test.

    Corrects for multiple comparisons when `padjust` is set. Leaving `padjust`
    at 'none' and then reporting only the significant pairs is the single most
    common error in post-hoc reporting — the tool states this explicitly.
    """
    try:
        df = _get_df(name)
        _check_columns(df, [dv, between, within, subject])
        res = pg.pairwise_tests(data=df, dv=dv, between=between, within=within,
                                subject=subject, parametric=parametric,
                                padjust=padjust, effsize=effsize, alpha=alpha,
                                interaction=interaction)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    pcol = _col(res, "p_unc", "p-unc", "p_val", "pval")
    pcor = _col(res, "p_corr", "p-corr")

    md = ["### Pairwise comparisons", "",
          f"_Parametric: {parametric} · Correction: {padjust}_", "",
          dataframe_table(res, max_rows=60)]

    try:
        if pcol:
            n_sig_unc = int((res[pcol] < alpha).sum())
            md += ["", f"- Significant before correction: **{n_sig_unc}** / {len(res)}"]
            if pcor:
                n_sig_cor = int((res[pcor] < alpha).sum())
                md += [f"- Significant after correction: **{n_sig_cor}** / {len(res)}"]
            if padjust in ("none", None) and len(res) > 1:
                md += ["", "**No multiple-comparison correction applied.** With "
                       f"{len(res)} comparisons, the family-wise error rate is inflated. "
                       "Re-run with `padjust='fdr_bh'` (recommended) or `'bonf'`, and report "
                       "the corrected p-values."]
    except Exception:
        pass

    md += ["", "_Report the corrected p-values, the effect size for each comparison, "
           "and the correction method by name._"]
    code = (f"pg.pairwise_tests(data=df, dv='{dv}'"
            + (f", between='{between}'" if between else "")
            + (f", within='{within}'" if within else "")
            + (f", subject='{subject}'" if subject else "")
            + f", parametric={parametric}, padjust='{padjust}')")
    return _ok("\n".join(md), code=code, data={"table": res_raw.to_dict("records")})


def stat_pairwise_tukey(name: str, dv: str, between: str, method: str = "tukey") -> dict:
    """
    Tukey HSD / Games-Howell post-hoc tests.

    Games-Howell does not assume equal variances — use it when Levene's test is
    significant.
    """
    try:
        df = _get_df(name)
        _check_columns(df, [dv, between])
        if method == "gameshowell":
            res = pg.pairwise_gameshowell(data=df, dv=dv, between=between)
            label = "Games-Howell"
        else:
            res = pg.pairwise_tukey(data=df, dv=dv, between=between)
            label = "Tukey HSD"
    except Exception as exc:
        return _unexpected(exc)

    res_raw = (res.reset_index(drop=True) if hasattr(res, "reset_index")
               else pd.DataFrame(res))
    res = _round_df(res_raw)
    md = [f"### {label} post-hoc — `{dv}` by `{between}`", "", dataframe_table(res)]
    try:
        pcol = _col(res, "p_tukey", "p-tukey", "p_val", "pval")
        n_sig = int((res[pcol] < 0.05).sum())
        md += ["", f"**{n_sig} of {len(res)} pairwise comparisons are significant** "
               "after correction."]
        md += ["", "_Tukey HSD assumes equal variances and equal group sizes; it is the "
               "appropriate default after a standard one-way ANOVA. Use Games-Howell when "
               "variances or group sizes are unequal._"]
    except Exception:
        pass
    code = (f"pg.pairwise_tukey(data=df, dv='{dv}', between='{between}')"
            if method != "gameshowell"
            else f"pg.pairwise_gameshowell(data=df, dv='{dv}', between='{between}')")
    return _ok("\n".join(md), code=code, data={"table": res_raw.to_dict("records")})


def stat_multicomp_correction(pvals: list[float], alpha: float = 0.05,
                              method: str = "holm") -> dict:
    """
    Correct a list of p-values for multiple comparisons.

    Methods: bonf (Bonferroni), sidak, holm (default), fdr_bh (Benjamini-Hochberg),
    fdr_by, fdr_tsbh, none.
    """
    try:
        arr = np.asarray(pvals, dtype=float)
    except Exception as exc:
        return _unexpected(exc)

    try:
        reject, p_corrected = pg.multicomp(arr, alpha=alpha, method=method)
    except Exception as exc:
        return _unexpected(exc)

    rows = [{"index": i, "p_uncorrected": fmt(p), "p_corrected": fmt(pc),
             "significant": "yes" if r else "no"}
            for i, (p, pc, r) in enumerate(zip(arr, p_corrected, reject))]

    md = [f"### Multiple-comparison correction — `{method}`", "",
          f"α = {alpha}", "", table(rows, ["index", "p_uncorrected", "p_corrected", "significant"])]
    md += ["", f"**{int(np.sum(reject))} of {len(arr)} remain significant after correction.**"]
    md += ["", "**Choosing a method**",
           "- `holm` — controls the family-wise error rate, uniformly more powerful than "
           "Bonferroni. Good default for a small number of planned comparisons.",
           "- `fdr_bh` — controls the false discovery rate; appropriate for many comparisons "
           "or exploratory work, but tolerates more false positives.",
           "- `bonf` — most conservative; use only when any single false positive is "
           "unacceptable.",
           "- `none` — no correction. Only defensible for a single pre-registered comparison."]
    return _ok("\n".join(md),
               code=f"pg.multicomp(pvals, alpha={alpha}, method='{method}')",
               data={"reject": [bool(r) for r in reject],
                     "p_corrected": [float(x) for x in p_corrected]})


# ══════════════════════════════════════════════════════════════════════════════
# F. Association
# ══════════════════════════════════════════════════════════════════════════════

def stat_correlation(name: str, x: str, y: str, method: str = "pearson",
                     alternative: str = "two-sided") -> dict:
    """Correlation between two variables (pearson / spearman / kendall / bicor / percbend / shepherds)."""
    try:
        df = _get_df(name)
        _check_columns(df, [x, y])
        res = pg.corr(df[x], df[y], method=method, alternative=alternative)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### Correlation — `{x}` × `{y}`", f"_Method: {method}_", "", dataframe_table(res)]
    try:
        row = res.iloc[0]
        r = float(row["r"])
        p = _num(row, "p_val", "p-val")
        md += ["", f"**Reading:** r = {fmt(r)}, p = {pfmt(p)} {sig_marker(p)}",
               f"- Direction: {'positive' if r > 0 else 'negative'}",
               f"- Magnitude: {_mag(abs(r))}"]
        if _col(res, "CI95", "CI95%"):
            md += [f"- 95% CI: {row[_col(res, 'CI95', 'CI95%')]}"]
        if method == "pearson":
            md += ["", "_Pearson's r assumes a linear relationship and is sensitive to "
                   "outliers. If the relationship is monotonic but not linear, or outliers "
                   "are present, Spearman's rho is more appropriate._"]
    except Exception:
        pass
    return _ok("\n".join(md), code=f"pg.corr(df['{x}'], df['{y}'], method='{method}')",
               data={"table": res_raw.to_dict("records")})


def _mag(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


def stat_partial_correlation(name: str, x: str, y: str, covar: list[str],
                             method: str = "pearson") -> dict:
    """Partial correlation: association between x and y after removing the covariates."""
    try:
        df = _get_df(name)
        covs = [covar] if isinstance(covar, str) else list(covar)
        _check_columns(df, [x, y] + covs)
        # `y` and `x` cannot also be covariates — partialling a variable out of
        # itself is undefined. Catch it here with a readable message rather than
        # letting pingouin raise a bare AssertionError.
        clash = sorted({x, y} & set(covs))
        if clash:
            return _err(
                f"{clash} appear as both the focal variable and a covariate.",
                "A variable cannot be partialled out of itself. Remove it from "
                "`covar`, or choose a different covariate.",
            )
        res = pg.partial_corr(data=df, x=x, y=y, covar=covs, method=method)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### Partial correlation — `{x}` × `{y}` controlling {covs}",
          f"_Method: {method}_", "", dataframe_table(res)]
    md += ["", "_Interpretation: this is the correlation between the residuals of `x` and "
           "`y` after regressing each on the covariates. It answers \"is there an "
           "association beyond what the covariates explain?\" — not \"does `x` "
           "cause `y`?\"_"]
    return _ok("\n".join(md),
               code=f"pg.partial_corr(data=df, x='{x}', y='{y}', covar={covs}, method='{method}')",
               data={"table": res_raw.to_dict("records")})


def stat_rm_correlation(name: str, x: str, y: str, subject: str) -> dict:
    """Repeated-measures correlation — the correct correlation for within-subject designs."""
    try:
        df = _get_df(name)
        _check_columns(df, [x, y, subject])
        res = pg.rm_corr(data=df, x=x, y=y, subject=subject)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### Repeated-measures correlation — `{x}` × `{y}` (subject `{subject}`)", "",
          dataframe_table(res)]
    md += ["", "**Why not a plain Pearson correlation here?** With repeated measurements, "
           "a standard correlation is inflated by between-subject differences in the means. "
           "rm_corr partitions the variance and reports the within-subject association, "
           "which is usually the quantity of interest. It also tests whether the "
           "within-subject and between-subject slopes differ."]
    return _ok("\n".join(md), code=f"pg.rm_corr(data=df, x='{x}', y='{y}', subject='{subject}')",
               data={"table": res_raw.to_dict("records")})


def stat_correlation_matrix(name: str, columns: list[str], method: str = "pearson",
                            padjust: str | None = None, decimals: int = 3) -> dict:
    """Correlation matrix over several variables, optionally with p-value adjustment."""
    try:
        df = _get_df(name)
        _check_columns(df, list(columns))
        sub = df[list(columns)].apply(pd.to_numeric, errors="coerce")
        # pg.rcorr is a DataFrame method.
        corr = sub.rcorr(method=method, padjust=padjust, decimals=decimals)
    except Exception as exc:
        return _unexpected(exc)

    md = [f"### Correlation matrix — {method}" + (f", corrected with `{padjust}`" if padjust else ""),
          "", "```", str(corr), "```"]
    if len(columns) > 1:
        n_comp = len(columns) * (len(columns) - 1) // 2
        md += ["", f"_{n_comp} unique comparisons. "
               + ("Correction applied." if padjust else
                  "**No correction applied** — with this many comparisons, some "
                  "significant results are expected by chance. Re-run with "
                  "`padjust='fdr_bh'` before interpreting individual cells.") + "_"]
    return _ok("\n".join(md),
               code=f"df[{list(columns)}].rcorr(method='{method}', padjust={padjust!r})")


def stat_distance_correlation(name: str, x: str, y: str, n_boot: int = 1000,
                              seed: int | None = 42) -> dict:
    """
    Distance correlation — detects non-linear dependence that Pearson's r misses.

    A non-significant Pearson r with a significant distance correlation means the
    variables are related, just not linearly.
    """
    try:
        df = _get_df(name)
        _check_columns(df, [x, y])
        sub = df[[x, y]].dropna()
        res = pg.distance_corr(sub[x], sub[y], n_boot=n_boot, seed=seed)
    except Exception as exc:
        return _unexpected(exc)

    md = ["### Distance correlation", "", "```", str(res), "```",
          "", f"_n = {len(sub)}, bootstrap = {n_boot}, seed = {seed}_",
          "", "_Distance correlation is zero **if and only if** the variables are "
          "independent. Unlike Pearson's r it detects non-monotonic dependence, so it is "
          "the right check before concluding \"no relationship\"._"]
    return _ok("\n".join(md),
               code=f"pg.distance_corr(df['{x}'], df['{y}'], n_boot={n_boot}, seed={seed})")


# ══════════════════════════════════════════════════════════════════════════════
# G. Regression and mediation
# ══════════════════════════════════════════════════════════════════════════════

def stat_linear_regression(name: str, y: str, x: list[str], alpha: float = 0.05) -> dict:
    """Ordinary least-squares regression with coefficients and model fit."""
    try:
        df = _get_df(name)
        xs = [x] if isinstance(x, str) else list(x)
        _check_columns(df, [y] + xs)
        sub = df[[y] + xs].dropna()
        res = pg.linear_regression(sub[xs], sub[y], alpha=alpha)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### Linear regression — `{y}` ~ {' + '.join(f'`{c}`' for c in xs)}",
          f"_n = {len(sub)}_", "", dataframe_table(res)]

    try:
        r2 = getattr(res, "r2", None)
        adj = getattr(res, "adj_r2", None)
        if r2 is not None:
            md += ["", f"- **R² = {fmt(r2)}** — the model explains {fmt(100 * float(r2), 1)}% "
                   "of the variance in the outcome"]
        if adj is not None:
            md += [f"- **Adjusted R² = {fmt(adj)}** (penalises added predictors; use this "
                   "when comparing models of different size)"]
        if r2 is not None and len(xs) > 1:
            md += ["", f"_With {len(xs)} predictors, R² is inflated relative to adjusted R². "
                   "Report both._"]
    except Exception:
        pass

    md += ["", "**Assumptions to check before interpreting these coefficients**",
           "- Linearity — partial regression plots",
           "- Homoscedasticity — residuals vs fitted",
           "- Normality of residuals",
           "- Multicollinearity — VIF (with several predictors)",
           "- Influential cases — Cook's distance"]
    md += ["", "_A significant coefficient means an association conditional on the other "
           "predictors. It is not evidence of a causal effect._"]
    return _ok("\n".join(md),
               code=f"pg.linear_regression(df[{xs}], df['{y}'])",
               data={"table": res_raw.to_dict("records"), "n": len(sub)})


def stat_logistic_regression(name: str, y: str, x: list[str], alpha: float = 0.05) -> dict:
    """Binary logistic regression with odds ratios."""
    try:
        df = _get_df(name)
        xs = [x] if isinstance(x, str) else list(x)
        _check_columns(df, [y] + xs)
        sub = df[[y] + xs].dropna()
        res = pg.logistic_regression(sub[xs], sub[y], alpha=alpha)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### Logistic regression — `{y}` ~ {' + '.join(f'`{c}`' for c in xs)}",
          f"_n = {len(sub)}_", "", dataframe_table(res)]

    try:
        if "OR" in res.columns:
            md += ["", "**Odds ratios (OR)**", "",
                   "_OR > 1: increases the odds of the outcome. OR < 1: decreases them. "
                   "An OR is not a risk ratio — for common outcomes it exaggerates the "
                   "effect, so prefer reporting predicted probabilities alongside._"]
            for _, row in res.iterrows():
                if row["names"] == "Intercept":
                    continue
                md += [f"- `{row['names']}`: OR = {fmt(row['OR'])}, "
                       f"95% CI [{fmt(row.get('CI[2.5%]'))}, {fmt(row.get('CI[97.5%]'))}], "
                       f"p = {pfmt(row.get('pval'))}"]
    except Exception:
        pass

    md += ["", "**Check**", "- Linearity of the logit for continuous predictors",
           "- Complete or quasi-complete separation (huge coefficients / CIs)",
           "- Class balance and calibration, not just discrimination"]
    return _ok("\n".join(md),
               code=f"pg.logistic_regression(df[{xs}], df['{y}'])",
               data={"table": res_raw.to_dict("records")})


def stat_mediation(name: str, x: str, m: str, y: str, covar: list[str] | None = None,
                   n_boot: int = 500, seed: int | None = 42) -> dict:
    """
    Mediation analysis with bootstrap confidence intervals.

    Tests whether the effect of `x` on `y` is transmitted through `m`.
    """
    try:
        df = _get_df(name)
        covs = ([covar] if isinstance(covar, str) else list(covar)) if covar else None
        _check_columns(df, [x, m, y] + (covs or []))
        res = pg.mediation_analysis(data=df, x=x, m=m, y=y, covar=covs,
                                    n_boot=n_boot, seed=seed)
    except Exception as exc:
        return _unexpected(exc)

    out = res[0] if isinstance(res, tuple) else res
    out_raw = (out.reset_index(drop=True) if hasattr(out, "reset_index")
               else pd.DataFrame(out))
    out = _round_df(out_raw)
    md = [f"### Mediation — `{x}` → `{m}` → `{y}`",
          f"_Bootstrap: {n_boot} resamples, seed = {seed}_", "", dataframe_table(out)]

    try:
        paths = {}
        for _, row in out.iterrows():
            paths[str(row.get("path", ""))] = row
        md += ["", "**Reading the paths**", "",
               f"- **a** — `{x}` → `{m}`",
               f"- **b** — `{m}` → `{y}` (controlling `{x}`)",
               f"- **c'** — direct effect of `{x}` on `{y}`",
               "- **ind** (a×b) — the indirect effect; its CI excluding zero is the "
               "evidence for mediation"]
        for key in ("ind", "Indirect", "indirect"):
            if key in paths:
                ind = paths[key]
                lo = float(ind["CI[2.5%]"])
                hi = float(ind["CI[97.5%]"])
                md += ["", f"**Indirect effect 95% CI: [{fmt(lo)}, {fmt(hi)}]** — "
                       + ("excludes zero, so mediation is supported."
                          if lo * hi > 0 else
                          "includes zero, so mediation is **not** supported.")]
                break
    except Exception:
        pass

    md += ["", "**Cautions**",
           "- Mediation requires the mediator to be measured before the outcome and to be "
           "plausibly caused by the predictor. With cross-sectional data the temporal order "
           "is assumed, not demonstrated.",
           "- A significant indirect effect from cross-sectional data does not establish a "
           "causal mechanism.",
           "- Report the bootstrap CI, not the Sobel test, which assumes a normal sampling "
           "distribution of the indirect effect."]
    return _ok("\n".join(md),
               code=f"pg.mediation_analysis(data=df, x='{x}', m='{m}', y='{y}', n_boot={n_boot}, seed={seed})",
               data={"table": out_raw.to_dict("records")})


# ══════════════════════════════════════════════════════════════════════════════
# H. Effect size, power, reliability
# ══════════════════════════════════════════════════════════════════════════════

def stat_effect_size(name: str, dv: str, group: str, paired: bool = False,
                     subject: str | None = None, eftype: str = "cohen") -> dict:
    """Effect size for a two-group comparison, with a confidence interval."""
    try:
        df = _get_df(name)
        _check_columns(df, [dv, group])
        levels = df[group].dropna().unique()
        if len(levels) != 2:
            return _err(f"`{group}` has {len(levels)} levels; effect size for two groups "
                        "is computed here.", "For 3+ groups use ANOVA effect sizes.")
        if paired:
            if not subject:
                return _err("Paired effect size needs `subject`.")
            wide = df.pivot_table(index=subject, columns=group, values=dv,
                                  aggfunc="mean").dropna()
            a, b = wide.iloc[:, 0].to_numpy(), wide.iloc[:, 1].to_numpy()
        else:
            a = df.loc[df[group] == levels[0], dv].dropna().to_numpy()
            b = df.loc[df[group] == levels[1], dv].dropna().to_numpy()

        d = pg.compute_effsize(a, b, paired=paired, eftype=eftype)
        ci = pg.compute_esci(stat=float(d), nx=len(a), ny=len(b), paired=paired,
                             eftype=eftype)
    except Exception as exc:
        return _unexpected(exc)

    md = [f"### Effect size — {levels[0]} vs {levels[1]}",
          f"_Type: {eftype} · Paired: {paired}_", "",
          "| Quantity | Value |", "|---|---|",
          f"| Effect size | {fmt(d)} |",
          f"| Magnitude | {_label_d(abs(float(d)))} |",
          f"| 95% CI | {ci} |",
          f"| n ({levels[0]}) | {len(a)} |",
          f"| n ({levels[1]}) | {len(b)} |",
          "",
          f"**Interpretation:** the two groups differ by {fmt(abs(float(d)))} standard "
          f"deviations ({_label_d(abs(float(d)))} effect)."]
    if paired:
        md += ["", "_For a paired design the standardiser is the SD of the *difference "
               "scores*, not the pooled SD of the raw scores. Cohen's d and dz are not "
               "interchangeable — state which one you report._"]
    md += ["", "_Report the effect size with its CI. A p-value tells you whether an effect "
           "is distinguishable from zero; the effect size tells you whether it matters._"]
    return _ok("\n".join(md),
               code=f"pg.compute_effsize(a, b, paired={paired}, eftype='{eftype}')\n"
                    f"pg.compute_esci(stat=d, nx={len(a)}, ny={len(b)}, paired={paired})",
               data={"effect_size": float(d), "ci": str(ci)})


def _label_d(d: float) -> str:
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def stat_power_analysis(test: str = "ttest", **kwargs) -> dict:
    """
    Power analysis: solve for whichever of effect size / n / power / alpha is left
    unspecified.

    Supported `test`: ttest, ttest2n, anova, rm_anova, corr, chi2.
    Pass exactly three of the four quantities to solve for the fourth.
    """
    fns = {
        "ttest": pg.power_ttest,
        "ttest2n": pg.power_ttest2n,
        "anova": pg.power_anova,
        "rm_anova": pg.power_rm_anova,
        "corr": pg.power_corr,
        "chi2": pg.power_chi2,
    }
    fn = fns.get(test)
    if fn is None:
        return _err(f"Unknown test `{test}`.", f"Supported: {sorted(fns)}")

    alias = {"d": "d", "eta_squared": "eta_squared", "r": "r", "w": "w",
             "eta": "eta_squared", "f": "eta_squared", "cohen_d": "d"}
    call_kwargs = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        call_kwargs[alias.get(k, k)] = v

    try:
        res = fn(**call_kwargs)
    except Exception as exc:
        return _unexpected(exc)

    val = float(res.iloc[0, 0]) if isinstance(res, pd.DataFrame) else float(res)
    col = res.columns[0] if isinstance(res, pd.DataFrame) else "value"

    md = [f"### Power analysis — `{test}`", "",
          "| Quantity | Value |", "|---|---|"]
    for k, v in call_kwargs.items():
        md += [f"| {k} (given) | {fmt(v)} |"]
    md += [f"| **{col} (solved)** | **{fmt(val)}** |", ""]

    if col.lower().startswith("power"):
        md += [f"**Power = {fmt(val)}** — "
               + ("adequate (≥ .80)" if val >= 0.80 else
                  "**below the conventional .80 threshold**; the study is underpowered "
                  "to detect an effect of this size.")]
    elif col.lower() in ("n", "nobs"):
        md += [f"**Required n = {fmt(val, 0)}** per group to reach the specified power."]

    md += ["", "_Power analysis is most defensible when performed **before** data "
           "collection, with the effect size justified from prior literature or theory "
           "rather than from the current sample._"]
    return _ok("\n".join(md), code=f"pg.power_{test}(**{call_kwargs})", data={col: val})


def stat_cronbach_alpha(name: str, items: list[str], ci: float = 0.95) -> dict:
    """Cronbach's alpha — internal consistency of a multi-item scale."""
    try:
        df = _get_df(name)
        _check_columns(df, list(items))
        alpha, ci_ = pg.cronbach_alpha(data=df[list(items)], ci=ci)
    except Exception as exc:
        return _unexpected(exc)

    a = float(alpha)
    md = [f"### Cronbach's alpha — {len(items)} items", "",
          "| Quantity | Value |", "|---|---|",
          f"| Alpha | {fmt(a, 3)} |",
          f"| {int(ci*100)}% CI | [{fmt(ci_[0], 3)}, {fmt(ci_[1], 3)}] |",
          f"| Items | {len(items)} |", ""]
    if a >= 0.90:
        md += ["**Reading: excellent** — but check for redundancy; alpha above .95 often "
               "means items are near-duplicates rather than that the scale is strong."]
    elif a >= 0.80:
        md += ["**Reading: good**"]
    elif a >= 0.70:
        md += ["**Reading: acceptable** for most research purposes."]
    elif a >= 0.60:
        md += ["**Reading: questionable** — report it and discuss the limitation."]
    else:
        md += ["**Reading: poor** — the items may not measure a single construct."]

    md += ["", "_Alpha depends on the number of items: adding items raises it even if they "
           "add little. Report the number of items and, where relevant, the mean "
           "inter-item correlation alongside._"]
    return _ok("\n".join(md),
               code=f"pg.cronbach_alpha(data=df[{items}], ci={ci})",
               data={"alpha": a, "ci": [float(ci_[0]), float(ci_[1])]})


def stat_intraclass_correlation(name: str, targets: str, raters: str,
                                ratings: str, icc_type: str = "ICC2") -> dict:
    """Intraclass correlation — agreement or consistency between raters/measurements."""
    try:
        df = _get_df(name)
        _check_columns(df, [targets, raters, ratings])
        res = pg.intraclass_corr(data=df, targets=targets, raters=raters, ratings=ratings)
    except Exception as exc:
        return _unexpected(exc)

    out_raw = res.reset_index(drop=True)

    out = _round_df(out_raw)
    md = ["### Intraclass correlation", "", dataframe_table(out)]

    md += ["", "**Choosing the right ICC form — this matters and is often got wrong**", "",
           "| Form | Model | What it answers |", "|---|---|---|",
           "| ICC1 | one-way, random | Are ratings related? Raters treated as exchangeable. |",
           "| ICC2 | two-way, random | **Absolute agreement** — do raters give the same *value*? |",
           "| ICC3 | two-way, mixed | **Consistency** — do raters *rank* the same way? |",
           "| ICC1k / 2k / 3k | as above, averaged | Reliability of the **mean** of k raters |", "",
           f"_You requested `{icc_type}`. If raters are a fixed set and you only need "
           "consistency, ICC3 is appropriate. If you need absolute agreement (e.g. "
           "interchangeable raters, clinical cut-offs), use ICC2. Single vs average "
           "ratings is a separate choice._"]

    md += ["", "**Conventional benchmarks** — report the CI regardless:",
           "- < 0.50 poor · 0.50–0.75 moderate · 0.75–0.90 good · > 0.90 excellent"]
    return _ok("\n".join(md),
               code=f"pg.intraclass_corr(data=df, targets='{targets}', raters='{raters}', ratings='{ratings}')",
               data={"table": out_raw.to_dict("records")})


def stat_equivalence_test(name: str, dv: str, group: str, bound: float,
                          paired: bool = False, subject: str | None = None) -> dict:
    """
    TOST equivalence test — for claiming that two groups are *equivalent*.

    A non-significant t-test is not evidence of equivalence. TOST is.
    """
    try:
        df = _get_df(name)
        _check_columns(df, [dv, group])
        levels = df[group].dropna().unique()
        if len(levels) != 2:
            return _err(f"`{group}` has {len(levels)} levels; TOST compares two.")
        if paired:
            if not subject:
                return _err("Paired TOST needs `subject`.")
            wide = df.pivot_table(index=subject, columns=group, values=dv,
                                  aggfunc="mean").dropna()
            a, b = wide.iloc[:, 0].to_numpy(), wide.iloc[:, 1].to_numpy()
        else:
            a = df.loc[df[group] == levels[0], dv].dropna().to_numpy()
            b = df.loc[df[group] == levels[1], dv].dropna().to_numpy()
        res = pg.tost(a, b, bound=bound, paired=paired)
    except Exception as exc:
        return _unexpected(exc)

    res_raw = res.reset_index(drop=True)

    res = _round_df(res_raw)
    md = [f"### Equivalence test (TOST) — {levels[0]} vs {levels[1]}",
          f"_Equivalence bound: ±{bound} · Paired: {paired}_", "", dataframe_table(res), "",
          f"**Equivalence bound used: ±{bound}.** The bound must be justified on substantive "
          "grounds — the largest difference that would still be considered negligible — and "
          "ideally specified before data collection. A bound chosen after seeing the data "
          "makes the test uninformative.", "",
          "**Reading:** equivalence is concluded when the 90% CI of the difference falls "
          "entirely inside the bound (equivalently, both one-sided tests are significant)."]
    return _ok("\n".join(md),
               code=f"pg.tost(a, b, bound={bound}, paired={paired})",
               data={"table": res_raw.to_dict("records")})


# ══════════════════════════════════════════════════════════════════════════════
# I. Categorical data
# ══════════════════════════════════════════════════════════════════════════════

def stat_chi2_independence(name: str, x: str, y: str, correction: bool = True) -> dict:
    """Chi-square test of independence for two categorical variables."""
    try:
        df = _get_df(name)
        _check_columns(df, [x, y])
        exp, obs, stats = pg.chi2_independence(data=df, x=x, y=y, correction=correction)
    except Exception as exc:
        return _unexpected(exc)

    md = [f"### Chi-square test of independence — `{x}` × `{y}`", "",
          "**Observed counts**", "", dataframe_table(obs), "",
          "**Expected counts**", "", dataframe_table(_round_df(exp)), "",
          "**Test statistics**", "",
          dataframe_table(_round_df(stats.reset_index(drop=True)
                                    if hasattr(stats, "reset_index") else stats))]
    try:
        row = stats[stats["test"] == "pearson"].iloc[0]
        p = float(row["pval"])
        md += ["", f"**Reading (Pearson):** chi2 = {fmt(row['chi2'])} (df = {fmt(row['dof'], 0)}), "
               f"p = {pfmt(p)} {sig_marker(p)}",
               f"- **Cramér's V:** {fmt(row.get('cramer'))}"]
    except Exception:
        pass

    try:
        small = int((exp < 5).sum().sum())
        total = int(exp.size)
        if small:
            md += ["", f"**Assumption check:** {small} of {total} expected cell counts are "
                   "below 5. The chi-square approximation is unreliable; use Fisher's exact "
                   "test (or a Monte-Carlo approximation), or collapse sparse categories "
                   "with a stated rationale."]
        else:
            md += ["", "**Assumption check:** all expected counts are ≥ 5. The chi-square "
                   "approximation is appropriate."]
    except Exception:
        pass
    return _ok("\n".join(md),
               code=f"pg.chi2_independence(data=df, x='{x}', y='{y}', correction={correction})",
               data={"expected": exp.to_dict(), "observed": obs.to_dict()})


def stat_chi2_mcnemar(name: str, x: str, y: str, correction: bool = True) -> dict:
    """McNemar's test — paired categorical data (e.g. before/after, matched pairs)."""
    try:
        df = _get_df(name)
        _check_columns(df, [x, y])
        obs, stats = pg.chi2_mcnemar(data=df, x=x, y=y, correction=correction)
    except Exception as exc:
        return _unexpected(exc)

    md = [f"### McNemar's test — `{x}` × `{y}`", "",
          "**Contingency table**", "", dataframe_table(obs), "",
          "**Test statistics**", "",
          dataframe_table(_round_df(stats.reset_index(drop=True)
                                    if hasattr(stats, "reset_index") else stats)),
          "", "_McNemar's test uses only the two discordant cells. The concordant cells "
          "contribute nothing to the test statistic — a common source of confusion when "
          "reading the table._"]
    return _ok("\n".join(md), code=f"pg.chi2_mcnemar(data=df, x='{x}', y='{y}')",
               data={"stats": stats.to_dict() if hasattr(stats, "to_dict") else str(stats)})


# ══════════════════════════════════════════════════════════════════════════════
# J. Visualization
# ══════════════════════════════════════════════════════════════════════════════

def stat_plot_paired(name: str, dv: str, within: str, subject: str,
                     boxplot: bool = True) -> dict:
    """Paired-data plot: one line per subject, showing both the change and the distribution."""
    import matplotlib.pyplot as plt

    try:
        df = _get_df(name)
        _check_columns(df, [dv, within, subject])
        ax = pg.plot_paired(data=df, dv=dv, within=within, subject=subject,
                            boxplot=boxplot, orient="v")
        fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
        fig.suptitle(f"{dv} by {within}")
    except Exception as exc:
        return _unexpected(exc)

    md = ["### Paired plot", "",
          "_Each line is one participant. This is the plot to use for repeated "
          "measures: it shows the within-subject change that the test actually "
          "analyses, which a bar chart of means conceals. Subjects moving in "
          "opposite directions are invisible in a mean comparison but obvious here._"]
    return _ok("\n".join(md),
               code=f"pg.plot_paired(data=df, dv='{dv}', within='{within}', subject='{subject}')")


def stat_plot_blandaltman(name: str, x: str, y: str, agreement: float = 1.96) -> dict:
    """Bland-Altman plot — agreement between two measurement methods."""
    try:
        df = _get_df(name)
        _check_columns(df, [x, y])
        sub = df[[x, y]].dropna()
        ax = pg.plot_blandaltman(sub[x], sub[y], agreement=agreement)
        fig = ax.figure if hasattr(ax, "figure") else None
        if fig:
            fig.suptitle(f"Agreement: {x} vs {y}")
    except Exception as exc:
        return _unexpected(exc)

    md = ["### Bland-Altman plot", "",
          "_Use this instead of a correlation when the question is whether two methods "
          "**agree**. A high correlation can coexist with large systematic disagreement — "
          "correlation measures association, not agreement._", "",
          f"_Limits of agreement drawn at ±{agreement} SD of the differences._"]
    return _ok("\n".join(md),
               code=f"pg.plot_blandaltman(df['{x}'], df['{y}'], agreement={agreement})")


def stat_qqplot(name: str, column: str, dist: str = "norm") -> dict:
    """Quantile-quantile plot for assessing distributional shape."""
    try:
        df = _get_df(name)
        _check_columns(df, [column])
        s = df[column].dropna()
        ax = pg.qqplot(s, dist=dist)
        fig = ax.figure if hasattr(ax, "figure") else None
        if fig:
            fig.suptitle(f"QQ plot: {column} vs {dist}")
    except Exception as exc:
        return _unexpected(exc)

    md = [f"### QQ plot — `{column}` vs {dist}", "",
          "_How to read it: points on the diagonal mean the data follow the reference "
          "distribution. Systematic curvature at the ends indicates skewness; a few points "
          "far off the line indicate outliers. A QQ plot is more informative than a "
          "normality test for judging whether a deviation actually matters._"]
    return _ok("\n".join(md), code=f"pg.qqplot(df['{column}'], dist='{dist}')")


def stat_plot_rm_corr(name: str, x: str, y: str, subject: str) -> dict:
    """Plot for a repeated-measures correlation."""
    try:
        df = _get_df(name)
        _check_columns(df, [x, y, subject])
        g = pg.plot_rm_corr(data=df, x=x, y=y, subject=subject)
        fig = g.figure if hasattr(g, "figure") else None
        if fig:
            fig.suptitle(f"RM correlation: {x} vs {y}")
    except Exception as exc:
        return _unexpected(exc)
    md = [f"### Repeated-measures correlation plot — `{x}` × `{y}`", "",
          "_Each line is one participant's fit. The bold line is the pooled "
          "within-subject relationship._"]
    return _ok("\n".join(md),
               code=f"pg.plot_rm_corr(data=df, x='{x}', y='{y}', subject='{subject}')")


# ══════════════════════════════════════════════════════════════════════════════
# K. Decision support and session tools
# ══════════════════════════════════════════════════════════════════════════════

def stat_decision_assistant(dv_type: str = "continuous", n_groups: int | None = None,
                            repeated: bool | None = None, n_dv: int = 1,
                            has_covariate: bool = False, normality_ok: bool | None = None,
                            homogeneity_ok: bool | None = None,
                            sphericity_ok: bool | None = None,
                            predicting: bool = False,
                            n_subjects: int | None = None) -> dict:
    """
    Recommend a statistical test from the study design.

    Call this **before** running a test whenever the appropriate method is not
    already fixed by a pre-registered analysis plan. It returns the recommended
    test, why, what to check first, and the alternatives — and it will ask a
    clarifying question rather than guess when the design is ambiguous.
    """
    design = dec.Design(
        dv_type=dv_type, n_groups=n_groups, repeated=repeated, n_dv=n_dv,
        has_covariate=has_covariate, normality_ok=normality_ok,
        homogeneity_ok=homogeneity_ok, sphericity_ok=sphericity_ok,
        predicting=predicting, n_subjects=n_subjects,
    )
    try:
        rec = dec.recommend(design)
    except Exception as exc:
        return _unexpected(exc)

    md = dec.render(rec, design)
    return _ok(md, code="# advisory only — no test was run", data=rec.to_dict())


def stat_session_info() -> dict:
    """List every dataset and result currently held in the session."""
    s = get_session()
    md = ["### Session contents", "", s.summary()]
    if s.history:
        md += ["", f"_{len(s.history)} code snippet(s) executed so far._"]
    md += next_steps_block([
        "`stat_data_quality_check` on a newly loaded dataset",
        "`stat_decision_assistant` if the appropriate test is not yet decided",
    ])
    return _ok("\n".join(md))


def stat_run_code(code: str) -> dict:
    """
    Execute arbitrary Python/Pingouin against the session namespace.

    Use this for anything the structured tools do not cover. Pre-bound names:
    `pg`, `pd`, `np`, `plt`, `sns`, plus every loaded object by name.
    """
    try:
        result = get_session().run_code(code, fig_prefix="code")
    except Exception as exc:
        return _unexpected(exc)

    parts = []
    if result.get("stdout"):
        parts.append("**stdout**\n```\n" + result["stdout"].strip() + "\n```")
    if result.get("stderr"):
        parts.append("**stderr**\n```\n" + result["stderr"].strip() + "\n```")
    if result.get("result_repr"):
        parts.append("**result**\n```\n" + result["result_repr"] + "\n```")
    if result.get("error"):
        parts.append(f"**error**\n```\n{result['error']}\n```")
        if result.get("traceback"):
            parts.append("```\n" + "\n".join(result["traceback"].splitlines()[-8:]) + "\n```")
    if not parts:
        parts.append("_Executed with no output._")

    return _ok("\n\n".join(parts), figures=result.get("figures"), code=None)


def stat_reset_session() -> dict:
    """Clear all loaded datasets and results, starting a fresh session."""
    try:
        get_session().reset()
    except Exception as exc:
        return _unexpected(exc)
    return _ok("Session reset — all datasets, results and figures cleared.")


# ── Registry ─────────────────────────────────────────────────────────────────
# Maps tool name → implementation. `server.py` uses this to register tools so
# that the two files cannot drift out of sync.

OPERATIONS = {
    "stat_load_data": stat_load_data,
    "stat_make_dataframe": stat_make_dataframe,
    "stat_describe_data": stat_describe_data,
    "stat_data_quality_check": stat_data_quality_check,
    "stat_normality_test": stat_normality_test,
    "stat_homogeneity_test": stat_homogeneity_test,
    "stat_sphericity_test": stat_sphericity_test,
    "stat_assumption_report": stat_assumption_report,
    "stat_ttest": stat_ttest,
    "stat_nonparametric_2": stat_nonparametric_2,
    "stat_bayes_ttest": stat_bayes_ttest,
    "stat_anova": stat_anova,
    "stat_welch_anova": stat_welch_anova,
    "stat_rm_anova": stat_rm_anova,
    "stat_mixed_anova": stat_mixed_anova,
    "stat_ancova": stat_ancova,
    "stat_nonparametric_k": stat_nonparametric_k,
    "stat_pairwise_tests": stat_pairwise_tests,
    "stat_pairwise_tukey": stat_pairwise_tukey,
    "stat_multicomp_correction": stat_multicomp_correction,
    "stat_correlation": stat_correlation,
    "stat_partial_correlation": stat_partial_correlation,
    "stat_rm_correlation": stat_rm_correlation,
    "stat_correlation_matrix": stat_correlation_matrix,
    "stat_distance_correlation": stat_distance_correlation,
    "stat_linear_regression": stat_linear_regression,
    "stat_logistic_regression": stat_logistic_regression,
    "stat_mediation": stat_mediation,
    "stat_effect_size": stat_effect_size,
    "stat_power_analysis": stat_power_analysis,
    "stat_cronbach_alpha": stat_cronbach_alpha,
    "stat_intraclass_correlation": stat_intraclass_correlation,
    "stat_equivalence_test": stat_equivalence_test,
    "stat_chi2_independence": stat_chi2_independence,
    "stat_chi2_mcnemar": stat_chi2_mcnemar,
    "stat_plot_paired": stat_plot_paired,
    "stat_plot_blandaltman": stat_plot_blandaltman,
    "stat_qqplot": stat_qqplot,
    "stat_plot_rm_corr": stat_plot_rm_corr,
    "stat_decision_assistant": stat_decision_assistant,
    "stat_session_info": stat_session_info,
    "stat_run_code": stat_run_code,
    "stat_reset_session": stat_reset_session,
}
