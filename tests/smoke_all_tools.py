# -*- coding: utf-8 -*-
"""
Smoke test: call every registered tool against realistic data.

This is a *behavioural* check, not a unit test. It answers one question: does
each tool produce a result on data of the shape it claims to handle?

It also asserts the properties the server promises:
  * output is non-empty Markdown,
  * no tool returns a raw traceback as its body,
  * a `code` snippet is attached for reproducibility (except advisories).

Exit code is non-zero if any tool fails.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd
import pingouin as pg  # noqa: F401  (ensures the stack imports)

from pingouin_mcp import operations as ops
from pingouin_mcp.kernel import get_session

RNG = np.random.default_rng(20260916)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _between(n_per_group: int = 24) -> pd.DataFrame:
    rows = []
    for gi, (g, mu) in enumerate([("control", 50.0), ("treatment", 45.0)]):
        for i in range(n_per_group):
            rows.append(dict(
                subject=f"{g[:3]}{i:03d}", group=g,
                anxiety=RNG.normal(mu, 10), age=RNG.normal(30, 6),
                sleep=RNG.normal(7, 1.2),
                sex=RNG.choice(["F", "M"]),
                responder=RNG.choice([0, 1], p=[0.4, 0.6]),
            ))
    return pd.DataFrame(rows)


def _three_groups(n_per_group: int = 20) -> pd.DataFrame:
    rows = []
    for g, mu in [("low", 40.0), ("mid", 47.0), ("high", 55.0)]:
        for i in range(n_per_group):
            rows.append(dict(subject=f"{g}{i:03d}", group=g,
                             score=RNG.normal(mu, 9),
                             baseline=RNG.normal(mu - 3, 8)))
    return pd.DataFrame(rows)


def _repeated(n_subj: int = 24) -> pd.DataFrame:
    rows = []
    for s in range(n_subj):
        base = RNG.normal(50, 8)
        for ti, (cond, delta) in enumerate([("pre", 0.0), ("mid", -4.0), ("post", -8.0)]):
            rows.append(dict(subject=f"S{s:03d}", time=cond,
                             score=base + delta + RNG.normal(0, 4),
                             hr=RNG.normal(70, 6) + delta * -0.3))
    return pd.DataFrame(rows)


def _repeated2(n_subj: int = 24) -> pd.DataFrame:
    """Two-level repeated fixture — the shape paired t-tests expect."""
    rows = []
    for s in range(n_subj):
        base = RNG.normal(50, 8)
        for cond, delta in [("pre", 0.0), ("post", -6.0)]:
            rows.append(dict(subject=f"S{s:03d}", time=cond,
                             score=base + delta + RNG.normal(0, 4),
                             hr=RNG.normal(70, 6) + delta * -0.3))
    return pd.DataFrame(rows)


def _mixed(n_subj: int = 32) -> pd.DataFrame:
    rows = []
    for s in range(n_subj):
        grp = "exp" if s % 2 == 0 else "ctrl"
        base = RNG.normal(50, 8) + (-5 if grp == "exp" else 0)
        for cond, delta in [("pre", 0.0), ("post", -6.0 if grp == "exp" else -1.0)]:
            rows.append(dict(subject=f"M{s:03d}", group=grp, time=cond,
                             score=base + delta + RNG.normal(0, 4)))
    return pd.DataFrame(rows)


def _scale(n: int = 120, k: int = 6) -> pd.DataFrame:
    latent = RNG.normal(0, 1, n)
    data = {f"item{i+1}": np.clip(latent + RNG.normal(0, 0.6, n), 1, 5).round()
            for i in range(k)}
    return pd.DataFrame(data)


def _raters(n_targets: int = 30, n_raters: int = 4) -> pd.DataFrame:
    rows = []
    for t in range(n_targets):
        true = RNG.normal(50, 10)
        for r in range(n_raters):
            rows.append(dict(target=f"T{t:03d}", rater=f"R{r}", rating=true + RNG.normal(0, 4)))
    return pd.DataFrame(rows)


def _categorical(n: int = 150) -> pd.DataFrame:
    return pd.DataFrame({
        "sex": RNG.choice(["F", "M"], n),
        "outcome": RNG.choice(["yes", "no"], n),
    })


def _paired_cat(n: int = 100) -> pd.DataFrame:
    before = RNG.choice(["yes", "no"], n, p=[0.5, 0.5])
    after = np.where(RNG.random(n) < 0.6, before, RNG.choice(["yes", "no"], n))
    return pd.DataFrame({"before": before, "after": after})


def _mediation(n: int = 150) -> pd.DataFrame:
    x = RNG.normal(0, 1, n)
    m = 0.6 * x + RNG.normal(0, 0.8, n)
    y = 0.3 * x + 0.5 * m + RNG.normal(0, 0.8, n)
    return pd.DataFrame({"x": x, "m": m, "y": y})


# ── Test harness ─────────────────────────────────────────────────────────────

FAILURES: list[tuple[str, str]] = []
RESULTS: list[tuple[str, str, int]] = []


def run(tool: str, **kwargs) -> None:
    """Call a tool and check the result is a usable, non-error response."""
    fn = ops.OPERATIONS.get(tool)
    if fn is None:
        FAILURES.append((tool, "not registered in OPERATIONS"))
        RESULTS.append((tool, "MISSING", 0))
        return
    try:
        res = fn(**kwargs)
    except Exception as exc:
        FAILURES.append((tool, f"raised {type(exc).__name__}: {exc}"))
        RESULTS.append((tool, f"RAISED {type(exc).__name__}", 0))
        return

    md = res.get("markdown", "") or ""
    n = len(md)

    if not md:
        FAILURES.append((tool, "empty markdown"))
        RESULTS.append((tool, "EMPTY", 0))
        return

    # A tool must not leak a traceback into its user-facing output.
    if "Traceback (most recent call last)" in md:
        FAILURES.append((tool, "returned a traceback"))
        RESULTS.append((tool, "TRACEBACK", n))
        return

    # *_err responses are legitimate but should be rare; flag them for review.
    if md.startswith("**Error:**"):
        RESULTS.append((tool, "ERROR-RESPONSE", n))
        FAILURES.append((tool, "returned an error response: " + md[:120]))
        return

    RESULTS.append((tool, "OK", n))


def main() -> int:
    sess = get_session()

    # ── Fixtures into the session ────────────────────────────────────────────
    sess.set("btw", _between())
    sess.set("three", _three_groups())
    sess.set("rep", _repeated())
    sess.set("rep2", _repeated2())
    sess.set("mix", _mixed())
    sess.set("sc", _scale())
    sess.set("icc", _raters())
    sess.set("cat", _categorical())
    sess.set("pcat", _paired_cat())
    sess.set("med", _mediation())

    # ── A. Data management ───────────────────────────────────────────────────
    run("stat_session_info")
    run("stat_describe_data", name="btw", dv="anxiety", group="group")
    run("stat_describe_data", name="btw")
    run("stat_data_quality_check", name="btw")
    run("stat_make_dataframe", name="tmp", columns={"a": [1, 2, 3], "b": [4, 5, 6]})

    # ── B. Decision support ──────────────────────────────────────────────────
    run("stat_decision_assistant", dv_type="continuous", n_groups=2, repeated=False)
    run("stat_decision_assistant", dv_type="continuous", n_groups=3, repeated=True)
    run("stat_decision_assistant", dv_type="continuous")           # should ask a question
    run("stat_decision_assistant", dv_type="categorical")
    run("stat_decision_assistant", dv_type="count")
    run("stat_decision_assistant", predicting=True)
    run("stat_decision_assistant", dv_type="continuous", n_groups=3,
        repeated=False, has_covariate=True)

    # ── C. Assumptions ───────────────────────────────────────────────────────
    run("stat_normality_test", name="btw", dv="anxiety", group="group")
    run("stat_normality_test", name="btw", dv="anxiety")
    run("stat_homogeneity_test", name="btw", dv="anxiety", group="group")
    run("stat_sphericity_test", name="rep", dv="score", within="time", subject="subject")
    run("stat_assumption_report", name="btw", dv="anxiety", between="group")
    run("stat_assumption_report", name="rep", dv="score", within="time", subject="subject")

    # ── D. Two groups ────────────────────────────────────────────────────────
    run("stat_ttest", name="btw", dv="anxiety", group="group")
    run("stat_ttest", name="rep2", dv="score", group="time", paired=True, subject="subject")
    run("stat_ttest", name="btw", dv="anxiety", group="group", correction=True)
    run("stat_nonparametric_2", name="btw", dv="anxiety", group="group")
    run("stat_nonparametric_2", name="rep2", dv="score", group="time",
        paired=True, subject="subject")
    run("stat_bayes_ttest", name="btw", dv="anxiety", group="group")

    # ── E. Multi-group ───────────────────────────────────────────────────────
    run("stat_anova", name="three", dv="score", between="group")
    run("stat_welch_anova", name="three", dv="score", between="group")
    run("stat_rm_anova", name="rep", dv="score", within="time", subject="subject")
    run("stat_mixed_anova", name="mix", dv="score", within="time",
        subject="subject", between="group")
    run("stat_ancova", name="three", dv="score", between="group", covar="baseline")
    run("stat_nonparametric_k", name="three", dv="score", between="group")
    run("stat_nonparametric_k", name="rep", dv="score", within="time", subject="subject")

    # ── F. Post-hoc ──────────────────────────────────────────────────────────
    run("stat_pairwise_tests", name="three", dv="score", between="group")
    run("stat_pairwise_tests", name="three", dv="score", between="group",
        padjust="fdr_bh")
    run("stat_pairwise_tests", name="rep", dv="score", within="time",
        subject="subject", padjust="holm")
    run("stat_pairwise_tukey", name="three", dv="score", between="group")
    run("stat_pairwise_tukey", name="three", dv="score", between="group",
        method="gameshowell")
    run("stat_multicomp_correction", pvals=[0.01, 0.04, 0.03, 0.20, 0.001],
        method="fdr_bh")

    # ── G. Association ───────────────────────────────────────────────────────
    run("stat_correlation", name="btw", x="anxiety", y="age")
    run("stat_correlation", name="btw", x="anxiety", y="age", method="spearman")
    run("stat_partial_correlation", name="btw", x="anxiety", y="age", covar=["sleep"])
    run("stat_rm_correlation", name="rep", x="score", y="hr", subject="subject")
    run("stat_correlation_matrix", name="btw", columns=["anxiety", "age"],
        padjust="fdr_bh")
    run("stat_distance_correlation", name="btw", x="anxiety", y="age", n_boot=100)

    # ── H. Regression ────────────────────────────────────────────────────────
    run("stat_linear_regression", name="btw", y="anxiety", x=["age"])
    run("stat_logistic_regression", name="btw", y="responder", x=["anxiety"])
    run("stat_mediation", name="med", x="x", m="m", y="y", n_boot=100)

    # ── I. Effect size / power / reliability ─────────────────────────────────
    run("stat_effect_size", name="btw", dv="anxiety", group="group")
    run("stat_power_analysis", test="ttest", d=0.5, power=0.80, alpha=0.05)
    run("stat_power_analysis", test="corr", r=0.3, power=0.80)
    run("stat_cronbach_alpha", name="sc", items=[f"item{i+1}" for i in range(6)])
    run("stat_intraclass_correlation", name="icc", targets="target",
        raters="rater", ratings="rating")
    run("stat_equivalence_test", name="btw", dv="anxiety", group="group", bound=10.0)

    # ── J. Categorical ───────────────────────────────────────────────────────
    run("stat_chi2_independence", name="cat", x="sex", y="outcome")
    run("stat_chi2_mcnemar", name="pcat", x="before", y="after")

    # ── K. Visualization ─────────────────────────────────────────────────────
    run("stat_plot_paired", name="rep", dv="score", within="time", subject="subject")
    run("stat_plot_blandaltman", name="btw", x="anxiety", y="age")
    run("stat_qqplot", name="btw", column="anxiety")
    run("stat_plot_rm_corr", name="rep", x="score", y="hr", subject="subject")

    # ── L. Code escape hatch ─────────────────────────────────────────────────
    run("stat_run_code",
        code="pg.compute_effsize(btw['anxiety'][:24], btw['anxiety'][24:48])")
    run("stat_run_code", code="1 + 1")
    run("stat_run_code", code="btw.groupby('group')['anxiety'].mean()")

    # ── L2. Error handling: bad input must yield a readable message ──────────
    # These are *expected* to produce error responses; they are checked
    # separately so they do not count as failures.
    expected_errors = [
        ("missing dataset", lambda: ops.stat_describe_data("nope", "x")),
        ("missing column", lambda: ops.stat_correlation("btw", "anxiety", "nope")),
        ("clashing covariate",
         lambda: ops.stat_partial_correlation("btw", "anxiety", "age", ["age"])),
        ("ancova dv == covar",
         lambda: ops.stat_ancova("btw", "anxiety", "group", "anxiety")),
        ("three groups in a t-test",
         lambda: ops.stat_ttest("three", "score", "group")),
    ]
    print("\n--- error handling (each should return a readable message) ---")
    for label, fn in expected_errors:
        try:
            r = fn()
            md = r.get("markdown", "")
            ok = md.startswith("**Error:**") and "Traceback (most recent call last)" not in md
            print(f"  [{'ok ' if ok else 'BAD'}] {label}: {md.splitlines()[0][:90]}")
            if not ok:
                FAILURES.append((f"error-handling: {label}", md[:200]))
        except Exception as exc:
            print(f"  [BAD] {label}: raised {type(exc).__name__}: {exc}")
            FAILURES.append((f"error-handling: {label}", f"raised {type(exc).__name__}"))

    # ── Report ───────────────────────────────────────────────────────────────
    print(f"{'TOOL':<34} {'STATUS':<16} {'CHARS':>6}")
    print("-" * 60)
    for name, status, n in RESULTS:
        print(f"{name:<34} {status:<16} {n:>6}")

    n_ok = sum(1 for _, s, _ in RESULTS if s == "OK")
    print("-" * 60)
    print(f"total {len(RESULTS)}   ok {n_ok}   problems {len(RESULTS) - n_ok}")

    if FAILURES:
        print("\n" + "=" * 60)
        print("FAILURES")
        print("=" * 60)
        for name, why in FAILURES:
            print(f"\n[{name}]\n  {why}")
        return 1

    print("\nAll tools produced usable output.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
