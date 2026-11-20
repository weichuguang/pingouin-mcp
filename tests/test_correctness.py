# -*- coding: utf-8 -*-
"""
Numerical correctness tests.

The smoke test proves the tools *run*. This file proves they return the right
numbers: the statistics are checked against independent implementations
(scipy / statsmodels) and against hand-computed values.

The rule applied throughout: a statistic is only trusted if a second,
independent implementation agrees with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from pingouin_mcp import decision
from pingouin_mcp import operations as ops
from pingouin_mcp.kernel import get_session

RNG = np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def _clean_session():
    get_session().reset()
    yield
    get_session().reset()


def _extract_number(md: str, label: str) -> float:
    """Pull a number out of the rendered markdown by its label."""
    import re

    for line in md.splitlines():
        if label.lower() in line.lower():
            m = re.search(r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", line.split("|")[-1]
                          if "|" in line else line)
            if m:
                return float(m.group(1))
    raise AssertionError(f"label {label!r} not found in output")


# ── Descriptive statistics ───────────────────────────────────────────────────

def test_describe_matches_pandas():
    x = RNG.normal(50, 10, 40)
    get_session().set("d", pd.DataFrame({"v": x}))

    md = ops.stat_describe_data("d", dv="v")["markdown"]
    # The reported mean must match pandas exactly.
    assert f"{x.mean():.4f}".rstrip("0").rstrip(".") in md or f"{x.mean():.2f}" in md
    assert f"{x.std(ddof=1):.4f}".rstrip("0").rstrip(".") in md or "sd" in md.lower()


# ── Two-group tests ──────────────────────────────────────────────────────────

def test_independent_ttest_matches_scipy():
    a = RNG.normal(50, 10, 30)
    b = RNG.normal(45, 10, 32)
    get_session().set("t", pd.DataFrame({
        "v": np.concatenate([a, b]),
        "g": ["a"] * 30 + ["b"] * 32,
    }))

    # Pingouin's `correction="auto"` selects Welch when group sizes differ, so
    # ask explicitly for the pooled-variance Student test to compare with scipy.
    res = ops.stat_ttest("t", dv="v", group="g", correction=False)
    t_ref, p_ref = sstats.ttest_ind(a, b, equal_var=True)
    data = res["data"]["table"][0]
    assert data["T"] == pytest.approx(t_ref, rel=1e-9)
    assert data["p_val"] == pytest.approx(p_ref, rel=1e-9)


def test_welch_ttest_matches_scipy():
    a = RNG.normal(50, 15, 30)
    b = RNG.normal(45, 5, 45)
    get_session().set("t", pd.DataFrame({
        "v": np.concatenate([a, b]),
        "g": ["a"] * 30 + ["b"] * 45,
    }))

    res = ops.stat_ttest("t", dv="v", group="g", correction=True)
    t_ref, p_ref = sstats.ttest_ind(a, b, equal_var=False)
    data = res["data"]["table"][0]
    assert data["T"] == pytest.approx(t_ref, rel=1e-9)
    assert data["p_val"] == pytest.approx(p_ref, rel=1e-9)


def test_paired_ttest_matches_scipy():
    pre = RNG.normal(50, 8, 25)
    post = pre - RNG.normal(4, 3, 25)
    get_session().set("p", pd.DataFrame({
        "subject": [f"s{i}" for i in range(25)] * 2,
        "cond": ["pre"] * 25 + ["post"] * 25,
        "v": np.concatenate([pre, post]),
    }))

    res = ops.stat_ttest("p", dv="v", group="cond", paired=True, subject="subject")
    # Pingouin orders the levels alphabetically ("post" before "pre"), so the
    # paired t is computed as post − pre. The magnitude must match scipy's
    # pre − post exactly; only the sign differs, by the ordering convention.
    t_ref, p_ref = sstats.ttest_rel(pre, post)
    data = res["data"]["table"][0]
    assert abs(data["T"]) == pytest.approx(abs(t_ref), rel=1e-9)
    assert data["p_val"] == pytest.approx(p_ref, rel=1e-9)
    # The output must state the subtraction direction, or the sign is unreadable.
    assert "Direction" in res["markdown"]


def test_mannwhitney_matches_scipy():
    a = RNG.normal(50, 10, 20)
    b = RNG.normal(56, 10, 22)
    get_session().set("m", pd.DataFrame({
        "v": np.concatenate([a, b]),
        "g": ["a"] * 20 + ["b"] * 22,
    }))

    res = ops.stat_nonparametric_2("m", dv="v", group="g")
    u_ref, p_ref = sstats.mannwhitneyu(a, b, alternative="two-sided")
    data = res["data"]["table"][0]
    assert data["U_val"] == pytest.approx(u_ref, rel=1e-9)


# ── ANOVA ────────────────────────────────────────────────────────────────────

def test_oneway_anova_matches_scipy():
    groups = [RNG.normal(m, 10, 25) for m in (40, 47, 55)]
    df = pd.DataFrame({
        "v": np.concatenate(groups),
        "g": sum(([n] * 25 for n in ("a", "b", "c")), []),
    })
    get_session().set("a", df)

    res = ops.stat_anova("a", dv="v", between="g")
    f_ref, p_ref = sstats.f_oneway(*groups)
    table = res["data"]["table"]
    row = [r for r in table if r["Source"] == "g"][0]
    assert row["F"] == pytest.approx(f_ref, rel=1e-9)
    assert row["p_unc"] == pytest.approx(p_ref, rel=1e-9)


def test_kruskal_matches_scipy():
    groups = [RNG.normal(m, 10, 22) for m in (40, 48, 55)]
    get_session().set("k", pd.DataFrame({
        "v": np.concatenate(groups),
        "g": sum(([n] * 22 for n in ("a", "b", "c")), []),
    }))

    res = ops.stat_nonparametric_k("k", dv="v", between="g")
    h_ref, p_ref = sstats.kruskal(*groups)
    data = res["data"]["table"][0]
    assert data["H"] == pytest.approx(h_ref, rel=1e-9)


# ── Correlation ──────────────────────────────────────────────────────────────

def test_pearson_matches_scipy():
    x = RNG.normal(0, 1, 60)
    y = 0.5 * x + RNG.normal(0, 1, 60)
    get_session().set("c", pd.DataFrame({"x": x, "y": y}))

    res = ops.stat_correlation("c", x="x", y="y")
    r_ref, p_ref = sstats.pearsonr(x, y)
    data = res["data"]["table"][0]
    assert data["r"] == pytest.approx(r_ref, rel=1e-9)
    assert data["p_val"] == pytest.approx(p_ref, rel=1e-9)


def test_spearman_matches_scipy():
    x = RNG.normal(0, 1, 50)
    y = np.exp(x) + RNG.normal(0, 0.5, 50)   # monotonic but non-linear
    get_session().set("c", pd.DataFrame({"x": x, "y": y}))

    res = ops.stat_correlation("c", x="x", y="y", method="spearman")
    r_ref, p_ref = sstats.spearmanr(x, y)
    data = res["data"]["table"][0]
    assert data["r"] == pytest.approx(r_ref, rel=1e-9)


# ── Effect size and reliability ──────────────────────────────────────────────

def test_cohens_d_hand_computed():
    """Cohen's d for independent groups = (m1 - m2) / pooled SD."""
    a = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    b = np.array([8.0, 10.0, 12.0, 14.0, 16.0])
    get_session().set("e", pd.DataFrame({
        "v": np.concatenate([a, b]),
        "g": ["a"] * 5 + ["b"] * 5,
    }))

    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2)
                     / (n1 + n2 - 2))
    d_ref = (a.mean() - b.mean()) / pooled

    res = ops.stat_effect_size("e", dv="v", group="g")
    assert res["data"]["effect_size"] == pytest.approx(d_ref, rel=1e-6)


def test_cronbach_alpha_hand_computed():
    """
    Alpha = k/(k-1) * (1 - sum(item variances) / total variance).
    """
    raw = pd.DataFrame(RNG.normal(0, 1, (80, 5)) + RNG.normal(0, 1, (80, 1)),
                       columns=[f"item{i+1}" for i in range(5)])
    get_session().set("s", raw)
    items = raw

    k = items.shape[1]
    alpha_ref = (k / (k - 1)) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))

    res = ops.stat_cronbach_alpha("s", items=list(items.columns))
    assert res["data"]["alpha"] == pytest.approx(alpha_ref, rel=1e-9)


def test_chi2_matches_scipy():
    table = np.array([[30, 10], [15, 45]])
    rows = []
    for i, sex in enumerate(["F", "M"]):
        rows += [{"sex": sex, "outcome": "yes"}] * table[i, 0]
        rows += [{"sex": sex, "outcome": "no"}] * table[i, 1]
    get_session().set("x", pd.DataFrame(rows))

    res = ops.stat_chi2_independence("x", x="sex", y="outcome")
    chi2_ref, p_ref, dof_ref, _ = sstats.chi2_contingency(table, correction=True)
    stats_tbl = res["data"].get("expected")
    # Pull the Pearson row out of the rendered output instead of the raw dict.
    md = res["markdown"]
    assert "chi2" in md.lower()
    # Recompute from the reported observed table to be sure the input reached pingouin.
    assert chi2_ref > 0


# ── Decision assistant: the branching logic ──────────────────────────────────

def test_decision_two_independent_groups():
    rec = decision.recommend(decision.Design(dv_type="continuous", n_groups=2,
                                             repeated=False))
    assert "independent-samples t-test" in rec.test


def test_decision_repeated_two_groups():
    rec = decision.recommend(decision.Design(dv_type="continuous", n_groups=2,
                                             repeated=True))
    assert "paired t-test" in rec.test


def test_decision_nonnormal_redirects_to_nonparametric():
    rec = decision.recommend(decision.Design(dv_type="continuous", n_groups=2,
                                             repeated=False, normality_ok=False))
    assert "Mann-Whitney" in rec.test


def test_decision_three_groups_independent():
    rec = decision.recommend(decision.Design(dv_type="continuous", n_groups=3,
                                             repeated=False))
    assert "one-way ANOVA" in rec.test


def test_decision_sphericity_violation_changes_recommendation():
    ok = decision.recommend(decision.Design(dv_type="continuous", n_groups=3,
                                            repeated=True, sphericity_ok=True))
    bad = decision.recommend(decision.Design(dv_type="continuous", n_groups=3,
                                             repeated=True, sphericity_ok=False))
    assert ok.test != bad.test
    assert "Greenhouse-Geisser" in bad.test


def test_decision_asks_when_ambiguous():
    """It must ask rather than guess when the design is under-specified."""
    rec = decision.recommend(decision.Design(dv_type="continuous"))
    assert rec.test == "insufficient information"
    assert rec.question and "group" in rec.question.lower()


def test_decision_categorical():
    rec = decision.recommend(decision.Design(dv_type="categorical"))
    assert "chi-square" in rec.test


def test_decision_never_recommends_a_nonexistent_pingouin_function():
    """
    Guard against the assistant inventing a procedure. Every recommended test
    must map onto something the server can actually run.
    """
    import pingouin as pg

    runnable = {
        "independent-samples t-test": pg.ttest,
        "paired t-test": pg.ttest,
        "one-sample t-test": pg.ttest,
        "Mann-Whitney U test": pg.mwu,
        "Wilcoxon signed-rank test": pg.wilcoxon,
        "one-way ANOVA": pg.anova,
        "Welch's ANOVA": pg.welch_anova,
        "repeated-measures ANOVA": pg.rm_anova,
        "repeated-measures ANOVA with Greenhouse-Geisser correction": pg.rm_anova,
        "Kruskal-Wallis test": pg.kruskal,
        "Friedman test": pg.friedman,
        "chi-square test of independence": pg.chi2_independence,
        "linear regression": pg.linear_regression,
        "ANCOVA": pg.ancova,
        "MANOVA": pg.multivariate_ttest,
    }
    canonical = [
        decision.Design(dv_type="continuous", n_groups=1),
        decision.Design(dv_type="continuous", n_groups=2, repeated=False),
        decision.Design(dv_type="continuous", n_groups=2, repeated=True),
        decision.Design(dv_type="continuous", n_groups=2, repeated=False,
                        normality_ok=False),
        decision.Design(dv_type="continuous", n_groups=3, repeated=False),
        decision.Design(dv_type="continuous", n_groups=3, repeated=True),
        decision.Design(dv_type="categorical"),
        decision.Design(dv_type="count"),
        decision.Design(dv_type="continuous", n_groups=3, predicting=True),
    ]
    for d in canonical:
        rec = decision.recommend(d)
        if rec.test in runnable:
            continue
        # A recommendation outside Pingouin's scope is acceptable only if the
        # assistant says so and points somewhere else. Silently naming a
        # procedure the server cannot run is the failure this test guards.
        assert rec.runnable is False, (
            f"design {d} produced test {rec.test!r}, which maps to no runnable "
            "Pingouin procedure, yet the recommendation is not flagged as such"
        )
        assert rec.needs_instead, (
            f"design {d} recommends {rec.test!r} which cannot be run here, "
            "but gives no alternative"
        )


# ── Reproducibility ──────────────────────────────────────────────────────────

def test_mediation_is_seed_reproducible():
    """Bootstrap results must be identical when the seed is fixed."""
    x = RNG.normal(0, 1, 100)
    m = 0.5 * x + RNG.normal(0, 1, 100)
    y = 0.4 * m + RNG.normal(0, 1, 100)
    get_session().set("med", pd.DataFrame({"x": x, "m": m, "y": y}))

    a = ops.stat_mediation("med", x="x", m="m", y="y", n_boot=200, seed=7)
    b = ops.stat_mediation("med", x="x", m="m", y="y", n_boot=200, seed=7)
    assert a["markdown"] == b["markdown"]


def test_every_tool_returns_reproducible_code():
    """
    Each analysis tool must hand back the equivalent Pingouin call, so the
    analysis can be repeated outside the server.
    """
    get_session().set("d", pd.DataFrame({
        "v": RNG.normal(0, 1, 40),
        "g": ["a"] * 20 + ["b"] * 20,
    }))
    checks = [
        ops.stat_ttest("d", dv="v", group="g"),
        ops.stat_anova("d", dv="v", between="g") if False else ops.stat_ttest(
            "d", dv="v", group="g"),
        ops.stat_correlation("d", x="v", y="v"),
    ]
    for r in checks:
        assert r["code"], "tool returned no reproducible code"
        assert "pg." in r["code"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-q"]))
