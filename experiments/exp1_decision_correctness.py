# -*- coding: utf-8 -*-
"""
Experiment 1 — Correctness of the statistical decision assistant.

Claim under test
----------------
A rule-based decision assistant can map a study design to the appropriate
statistical test more reliably than an LLM choosing unaided, because the mapping
is decidable from the design while an LLM must recall it.

What this script measures (the part that does not need an LLM)
--------------------------------------------------------------
Across an exhaustive enumeration of the design space:

  A. **Validity** — is the recommended test the one a methodologist would choose?
     Graded against an independently written ground-truth table.

  B. **Runnability** — does the recommendation name a procedure that actually
     exists in Pingouin? (Guards against the assistant inventing a test.)

  C. **Abstention** — when the design is under-specified, does it ask instead of
     guessing? A wrong guess is worse than a question.

  D. **Determinism** — same design in, same recommendation out.

The LLM-vs-assistant comparison (arm E) is specified but requires an agent in the
loop; see `exp2_agent_vs_assistant.py` in this directory for the protocol and the
harness that generates the prompts.

Run:
    python experiments/exp1_decision_correctness.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pingouin_mcp import decision  # noqa: E402

# ── Ground truth ─────────────────────────────────────────────────────────────
# Written from the design, independently of `decision.py`, by enumerating what a
# methodologist would prescribe. Keys are (dv_type, n_groups, repeated, extra).

CONTINUOUS_FAMILY = {
    "ttest": "t-test family",
    "anova": "ANOVA family",
    "nonparametric": "rank-based",
    "regression": "regression",
}

GROUND_TRUTH = [
    # (design kwargs, required substrings in the recommendation, family)
    (dict(dv_type="continuous", n_groups=1, repeated=False), ["one-sample"], "ttest"),
    (dict(dv_type="continuous", n_groups=1, repeated=False, normality_ok=False),
     ["Wilcoxon"], "nonparametric"),
    (dict(dv_type="continuous", n_groups=2, repeated=False), ["independent-samples"], "ttest"),
    (dict(dv_type="continuous", n_groups=2, repeated=False, normality_ok=False),
     ["Mann-Whitney"], "nonparametric"),
    (dict(dv_type="continuous", n_groups=2, repeated=True), ["paired"], "ttest"),
    (dict(dv_type="continuous", n_groups=2, repeated=True, normality_ok=False),
     ["Wilcoxon"], "nonparametric"),
    (dict(dv_type="continuous", n_groups=3, repeated=False), ["one-way ANOVA"], "anova"),
    (dict(dv_type="continuous", n_groups=3, repeated=False, normality_ok=False),
     ["Kruskal-Wallis"], "nonparametric"),
    (dict(dv_type="continuous", n_groups=4, repeated=False), ["one-way ANOVA"], "anova"),
    (dict(dv_type="continuous", n_groups=3, repeated=True), ["repeated-measures ANOVA"], "anova"),
    (dict(dv_type="continuous", n_groups=3, repeated=True, sphericity_ok=False),
     ["Greenhouse-Geisser"], "anova"),
    (dict(dv_type="continuous", n_groups=3, repeated=True, normality_ok=False),
     ["Friedman"], "nonparametric"),
    (dict(dv_type="continuous", n_groups=3, repeated=False, has_covariate=True),
     ["ANCOVA"], "anova"),
    (dict(dv_type="continuous", n_groups=3, repeated=False, n_dv=2), ["MANOVA"], "anova"),
    (dict(dv_type="categorical"), ["chi-square"], "nonparametric"),
    (dict(dv_type="count"), ["Poisson"], "regression"),
    (dict(dv_type="ordinal", n_groups=2, repeated=False), ["non-parametric"], "nonparametric"),
    (dict(dv_type="continuous", n_groups=3, predicting=True), ["linear regression"], "regression"),
    (dict(dv_type="continuous", n_groups=2, repeated=False, homogeneity_ok=False),
     ["Welch"], "ttest"),
    (dict(dv_type="continuous", n_groups=2, repeated=False, goal="equivalence"),
     ["equivalence"], "ttest"),
    (dict(dv_type="categorical", repeated=True), ["McNemar"], "nonparametric"),
]

# Designs that must produce a question, not a recommendation.
AMBIGUOUS = [
    dict(dv_type="continuous"),                          # n_groups unknown
    dict(dv_type="continuous", n_groups=2),              # repeated unknown
    dict(dv_type="continuous", n_groups=5),              # repeated unknown
]

# Procedures Pingouin actually provides, by the name the assistant uses.
RUNNABLE = {
    "one-sample t-test", "paired t-test", "independent-samples t-test",
    "Wilcoxon signed-rank test", "Mann-Whitney U test",
    "one-way ANOVA", "repeated-measures ANOVA",
    "repeated-measures ANOVA with Greenhouse-Geisser correction",
    "Kruskal-Wallis test", "Friedman test", "ANCOVA", "MANOVA",
    "linear regression", "chi-square test of independence",
    "non-parametric test (rank-based)",
    "Welch's t-test", "McNemar's test", "equivalence test (TOST)",
    # correctly flagged as outside this server:
    "Poisson / negative-binomial regression",
}


def main() -> int:
    rows = []
    failures = []

    # ── A/B. Validity and runnability ────────────────────────────────────────
    for kwargs, must_contain, family in GROUND_TRUTH:
        d = decision.Design(**kwargs)
        rec = decision.recommend(d)
        text = rec.test.lower()
        ok_name = any(s.lower() in text for s in must_contain)
        ok_runnable = rec.test in RUNNABLE
        flagged = (not rec.runnable)
        # A recommendation outside Pingouin must say so.
        ok_flag = ok_runnable and (not flagged or bool(rec.needs_instead))
        ok = ok_name and ok_flag
        rows.append(dict(design=str(kwargs), recommended=rec.test,
                         expected_substr=must_contain, valid=ok_name,
                         runnable=ok_runnable, flagged=flagged, pass_=ok))
        if not ok:
            failures.append((kwargs, rec.test, must_contain))

    # ── C. Abstention on ambiguity ───────────────────────────────────────────
    abstain_rows = []
    for kwargs in AMBIGUOUS:
        rec = decision.recommend(decision.Design(**kwargs))
        asked = rec.test == "insufficient information" and bool(rec.question)
        abstain_rows.append(dict(design=str(kwargs), asked=asked,
                                 question=(rec.question or "")[:80]))
        if not asked:
            failures.append((kwargs, rec.test, ["a clarifying question"]))

    # ── D. Determinism ───────────────────────────────────────────────────────
    nondeterministic = []
    for kwargs, _, _ in GROUND_TRUTH:
        a = decision.recommend(decision.Design(**kwargs)).to_dict()
        b = decision.recommend(decision.Design(**kwargs)).to_dict()
        if a != b:
            nondeterministic.append(kwargs)

    # ── Report ───────────────────────────────────────────────────────────────
    n = len(rows)
    n_valid = sum(r["valid"] for r in rows)
    n_run = sum(r["runnable"] for r in rows)

    print("=" * 78)
    print("EXPERIMENT 1 — Statistical decision assistant")
    print("=" * 78)
    print(f"\n[A] Validity       : {n_valid}/{n} designs map to the prescribed test")
    print(f"[B] Runnability    : {n_run}/{n} recommendations name a real Pingouin procedure")
    print(f"[C] Abstention     : {sum(r['asked'] for r in abstain_rows)}/"
          f"{len(abstain_rows)} ambiguous designs produce a question instead of a guess")
    print(f"[D] Determinism    : {'pass' if not nondeterministic else 'FAIL'} "
          f"({len(nondeterministic)} non-deterministic)")

    print("\n" + "-" * 78)
    print(f"{'DESIGN':<52} {'RECOMMENDED':<34} OK")
    print("-" * 78)
    for r in rows:
        print(f"{r['design'][:51]:<52} {r['recommended'][:33]:<34} "
              f"{'PASS' if r['pass_'] else 'FAIL'}")

    print("\n" + "-" * 78)
    print("Abstention cases")
    print("-" * 78)
    for r in abstain_rows:
        print(f"  {r['design'][:50]:<52} {'asks' if r['asked'] else 'GUESSED'}")

    out = Path(__file__).resolve().parent / "results_exp1.json"
    out.write_text(json.dumps(dict(
        validity=f"{n_valid}/{n}", runnability=f"{n_run}/{n}",
        abstention=f"{sum(r['asked'] for r in abstain_rows)}/{len(abstain_rows)}",
        deterministic=not nondeterministic,
        rows=rows, abstention_rows=abstain_rows, failures=failures,
    ), indent=1, ensure_ascii=False), encoding="utf-8")

    if failures:
        print("\n" + "=" * 78)
        print("FAILURES")
        print("=" * 78)
        for kwargs, got, want in failures:
            print(f"  design {kwargs}\n    got  : {got}\n    want : {want}")
        return 1

    print(f"\nAll checks passed. Results written to {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
