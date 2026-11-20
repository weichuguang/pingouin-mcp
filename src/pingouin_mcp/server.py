"""
Pingouin MCP server — tool definitions.

Every tool is a thin wrapper over :mod:`pingouin_mcp.operations`. The docstrings
matter more than usual here: they are the *only* thing an LLM reads when
deciding whether to call a tool. They therefore state, for each tool:

* what it does, in the terms of a research question rather than an API;
* **when to use it** and, just as importantly, **when not to**;
* the assumptions it carries;
* what the caller should run first.

A single ``Session`` is held for the whole process, so loaded datasets and
results survive across tool calls. ``stat_run_code`` is the escape hatch for
anything the structured tools do not cover.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager

from fastmcp import Context, FastMCP

from pingouin_mcp import operations as ops
from pingouin_mcp.config import detect_capabilities, get_timeout
from pingouin_mcp.kernel import get_session


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    caps = detect_capabilities()
    sys.stderr.write("Starting Pingouin MCP server...\n")
    if caps.get("pingouin"):
        sys.stderr.write(f"  Pingouin   : available v{caps['pingouin_version']}\n")
    else:
        sys.stderr.write("  Pingouin   : NOT FOUND (pip install pingouin)\n")
    sys.stderr.write(
        f"  statsmodels: {'available' if caps.get('statsmodels') else 'NOT FOUND'}\n"
        f"  Results dir: {caps.get('results_dir')}\n"
        f"  Session    : empty (datasets load on demand)\n"
    )
    yield {"capabilities": caps}
    sys.stderr.write("Shutting down Pingouin MCP server.\n")


mcp = FastMCP("Pingouin", lifespan=server_lifespan)

# The session is shared mutable state; serialize execution so overlapping calls
# cannot interleave on the same objects. Statistical work is CPU-bound and fast,
# so there is nothing to gain from parallelism here.
_EXEC_LOCK = asyncio.Lock()


def _format(result: dict) -> str:
    parts = []
    if result.get("markdown"):
        parts.append(result["markdown"])
    for fig in result.get("figures", []) or []:
        parts.append(f"\n> Figure: `{fig}`")
    if result.get("figures"):
        parts.append("\n_(Read the PNG path(s) above to view the figure.)_")
    if result.get("code"):
        parts.append(f"\n**Equivalent Pingouin code**\n```python\n{result['code']}\n```")
    return "\n".join(parts) if parts else "_No output._"


async def _exec(fn, *args, **kwargs) -> str:
    """Run a synchronous operation in a worker thread with a timeout."""
    caps = detect_capabilities()
    if not caps.get("pingouin"):
        return ("Error: Pingouin is not installed in the server environment. "
                "Install it with `pip install pingouin`.")
    try:
        async with _EXEC_LOCK:
            result = await asyncio.wait_for(
                asyncio.to_thread(fn, *args, **kwargs), timeout=get_timeout()
            )
    except asyncio.TimeoutError:
        return (f"Error: operation timed out after {get_timeout()}s. "
                "Bootstrap procedures (mediation, power, distance correlation) are the usual "
                "cause — reduce n_boot, or raise PINGOUIN_MCP_TIMEOUT.")
    except Exception as exc:  # noqa: BLE001
        return f"Error: {type(exc).__name__}: {exc}"
    return _format(result)


# ══════════════════════════════════════════════════════════════════════════════
# Data management
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_load_data(path: str, name: str = "df", ctx: Context = None) -> str:
    """Load a tabular dataset (CSV/TSV/Excel/JSON/Parquet/SPSS) into the session.

    Use this as the first step for any dataset stored as a file. The loaded
    object is stored under `name` and every other tool refers to it by that name.

    When to use: at the start of an analysis session, once per dataset.
    When NOT to use: if the data are already in the session (check with
    `stat_session_info` first), or if you need to construct a small example —
    use `stat_make_dataframe` instead.

    Args:
        path: Path to the file.
        name: Session name to store the DataFrame under (default "df").
    """
    return await _exec(ops.stat_load_data, path, name)


@mcp.tool()
async def stat_make_dataframe(name: str, columns: dict, ctx: Context = None) -> str:
    """Build a DataFrame from a {column: [values]} mapping.

    Use this for worked examples, sanity checks, and illustrating a method with
    hypothetical data.

    When NOT to use: never for real analysis — fabricated data must not enter an
    analysis of real results.
    """
    return await _exec(ops.stat_make_dataframe, name, columns)


@mcp.tool()
async def stat_describe_data(name: str, dv: str = None, group: str = None,
                             ctx: Context = None) -> str:
    """Descriptive statistics, optionally broken down by group.

    Use this to report the sample before any inference: means, SDs, medians and
    group sizes are expected in every results section.

    When to use: immediately after loading, and again when writing up.
    When NOT to use: as a substitute for checking assumptions — use
    `stat_data_quality_check` and `stat_assumption_report` for that.

    Args:
        name: Session dataset name.
        dv: Dependent variable (numeric column). If omitted, all numeric columns are described.
        group: Optional grouping column for a per-group breakdown.
    """
    return await _exec(ops.stat_describe_data, name, dv, group)


@mcp.tool()
async def stat_data_quality_check(name: str, columns: list[str] = None,
                                  ctx: Context = None) -> str:
    """Screen a dataset for missingness, outliers and distribution shape.

    Run this on every new dataset before planning any analysis: it determines
    whether the planned analysis is even possible (e.g. a between-subjects test
    needs enough complete cases per group).

    When to use: first thing after loading real data.
    Returns: missing counts per column, outliers flagged by the robust MAD rule,
    skewness and kurtosis.
    """
    return await _exec(ops.stat_data_quality_check, name, columns)


# ══════════════════════════════════════════════════════════════════════════════
# Decision support
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_decision_assistant(
    dv_type: str = "continuous",
    n_groups: int = None,
    repeated: bool = None,
    n_dv: int = 1,
    has_covariate: bool = False,
    normality_ok: bool = None,
    homogeneity_ok: bool = None,
    sphericity_ok: bool = None,
    predicting: bool = False,
    n_subjects: int = None,
    ctx: Context = None,
) -> str:
    """Recommend which statistical test fits a study design — call this BEFORE running a test.

    This exists because choosing the wrong test is the most consequential error
    in statistical analysis, and it invalidates the p-value rather than merely
    producing a suboptimal number. If you are not certain which test applies,
    call this first.

    When to use:
      - Before any test whose appropriateness you have not already verified.
      - When the design is unusual (mixed within/between, covariates, multiple DVs).
      - When a reviewer or the researcher asks "should we have used a different test?"

    When NOT to use:
      - When the analysis plan is pre-registered and fixes the test. Follow the plan.
      - As a substitute for checking assumptions — it *tells you* what to check.

    It will ask a clarifying question rather than guess if the design is
    ambiguous. Answer it and call again.

    Args:
        dv_type: continuous | ordinal | categorical | count.
        n_groups: Number of groups/conditions compared (1, 2, or 3+).
        repeated: True if the same participants appear in every condition (within-subjects).
        n_dv: How many dependent variables are compared jointly.
        has_covariate: True if a continuous covariate must be controlled.
        normality_ok: Result of the normality check, if already run.
        homogeneity_ok: Result of Levene's test, if already run.
        sphericity_ok: Result of Mauchly's test, if already run.
        predicting: True if the goal is prediction from continuous predictors rather than group comparison.
        n_subjects: Total number of participants, if known.
    """
    return await _exec(
        ops.stat_decision_assistant, dv_type, n_groups, repeated, n_dv,
        has_covariate, normality_ok, homogeneity_ok, sphericity_ok,
        predicting, n_subjects,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Assumption testing
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_assumption_report(name: str, dv: str, between: str = None,
                                 within: str = None, subject: str = None,
                                 covar: str = None, ctx: Context = None) -> str:
    """Run every assumption check implied by a design in one pass.

    Use this instead of checking assumptions one at a time, so that none is
    silently skipped. Report the output whether or not the assumptions pass — a
    check reported only when it fails is not a check.

    When to use: after loading and before the main test, for any parametric test.
    When NOT to use: for purely non-parametric analyses (normality is irrelevant
    there), or for descriptive work.

    Args:
        name: Session dataset name.
        dv: Dependent variable.
        between: Between-subjects factor, if any.
        within: Within-subjects factor, if any.
        subject: Subject identifier (required for within-subjects designs).
        covar: Covariate, if the design has one.
    """
    return await _exec(ops.stat_assumption_report, name, dv, between, within, subject, covar)


@mcp.tool()
async def stat_normality_test(name: str, dv: str, group: str = None,
                              method: str = "shapiro", ctx: Context = None) -> str:
    """Test whether a variable is normally distributed, optionally within each group.

    When to use: before any parametric test (t-test, ANOVA, Pearson correlation).
    When NOT to use: as a gatekeeper that alone decides parametric vs
    non-parametric. With large n the test detects trivial deviations; with small
    n it has low power. Judge with the QQ plot and skewness too.

    Args:
        name: Session dataset name.
        dv: The variable to test.
        group: Optional grouping column — normality is then tested per group.
        method: shapiro (default, best for n < 50) | normaltest | jarque_bera | omni.
    """
    return await _exec(ops.stat_normality_test, name, dv, group, method)


@mcp.tool()
async def stat_homogeneity_test(name: str, dv: str, group: str,
                                method: str = "levene", ctx: Context = None) -> str:
    """Test equality of variances across groups.

    When to use: before an independent-samples t-test or a one-way ANOVA.
    When it fails: use Welch's t-test or Welch's ANOVA, which do not assume
    equal variances.

    Args:
        method: levene (default, robust to non-normality) | bartlett (more powerful, assumes normality).
    """
    return await _exec(ops.stat_homogeneity_test, name, dv, group, method)


@mcp.tool()
async def stat_sphericity_test(name: str, dv: str, within: str, subject: str,
                               method: str = "mauchly", ctx: Context = None) -> str:
    """Mauchly's test of sphericity for a repeated-measures factor with 3+ levels.

    When to use: before a repeated-measures ANOVA with three or more conditions.
    When it fails: report the Greenhouse-Geisser corrected p-value and the
    epsilon. The uncorrected F is anti-conservative when sphericity is violated.

    Not applicable with only two levels — a paired t-test needs no sphericity
    assumption.
    """
    return await _exec(ops.stat_sphericity_test, name, dv, within, subject, method)


# ══════════════════════════════════════════════════════════════════════════════
# Two-group comparisons
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_ttest(name: str, dv: str, group: str = None, paired: bool = False,
                     subject: str = None, x: str = None, y: str = None,
                     alternative: str = "two-sided", correction: str = "auto",
                     confidence: float = 0.95, ctx: Context = None) -> str:
    """t-test: independent samples, paired, or one-sample.

    WHEN TO USE
      - Two independent groups, continuous outcome, roughly normal, variances similar.
      - Same participants in two conditions → `paired=True` with `subject`.
      - One sample against a reference value → pass `x` (and `y` as the constant, or omit).

    WHEN NOT TO USE
      - Three or more groups → `stat_anova` (a t-test on all pairs inflates Type I error).
      - Ordinal outcome, or badly non-normal with small n → `stat_nonparametric_2`.
      - Variances unequal → still a t-test, but set `correction=True` to get Welch's.
      - To claim two groups are equivalent → `stat_equivalence_test`; a non-significant
        t-test is not evidence of equivalence.

    Always report the effect size (Cohen's d, given in the output) and the CI.

    Args:
        name: Session dataset name.
        dv: Dependent variable.
        group: Column with exactly two levels (omit when using x/y).
        paired: True for within-subject / matched designs.
        subject: Participant identifier — required when paired=True.
        x: For one-sample or two-vector mode.
        y: Second variable for two-vector mode.
        alternative: two-sided | less | greater.
        correction: 'auto' (Welch when sizes/variances differ) | True | False.
        confidence: Confidence level for the CI.
    """
    return await _exec(ops.stat_ttest, name, dv, group, paired, subject, x, y,
                       alternative, correction, confidence)


@mcp.tool()
async def stat_nonparametric_2(name: str, dv: str, group: str = None,
                               paired: bool = False, subject: str = None,
                               x: str = None, y: str = None,
                               alternative: str = "two-sided", ctx: Context = None) -> str:
    """Mann-Whitney U (independent) or Wilcoxon signed-rank (paired) — two groups, no normality assumption.

    WHEN TO USE
      - Ordinal outcomes.
      - Continuous outcomes that are badly non-normal with small n.
      - Outliers that cannot be justified for removal.
      - Any time the mean is not a meaningful summary of the data.

    WHEN NOT TO USE
      - Three or more groups → `stat_nonparametric_k`.
      - When the data are normal: the parametric test is more powerful, and using
        ranks discards information.

    Report the rank-biserial correlation (RBC) as the effect size — the p-value
    alone says nothing about magnitude.
    """
    return await _exec(ops.stat_nonparametric_2, name, dv, group, paired, subject,
                       x, y, alternative)


@mcp.tool()
async def stat_bayes_ttest(name: str, dv: str, group: str, paired: bool = False,
                           subject: str = None, alternative: str = "two-sided",
                           r: float = 0.707, ctx: Context = None) -> str:
    """Bayesian t-test — quantifies evidence for the null as well as the alternative.

    WHEN TO USE
      - When a non-significant result needs interpreting. A frequentist p > .05
        does not mean the null is true; BF₀₁ directly quantifies evidence for it.
      - When the researcher wants to state "the groups are comparable".

    WHEN NOT TO USE
      - As a way to find significance after a non-significant t-test. Decide on
        the analysis approach in advance.

    Report BF₁₀ and BF₀₁, and state the prior scale (r).
    """
    return await _exec(ops.stat_bayes_ttest, name, dv, group, paired, subject,
                       alternative, r)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-group comparisons
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_anova(name: str, dv: str, between: str, ss_type: int = 2,
                     effsize: str = "np2", ctx: Context = None) -> str:
    """One-way ANOVA — three or more independent groups, continuous outcome.

    WHEN TO USE
      - 3+ independent groups; normality and homogeneity plausible.
      - Instead of running multiple t-tests, which inflates the Type I error rate.

    WHEN NOT TO USE
      - Variances unequal → `stat_welch_anova`.
      - Same participants in every condition → `stat_rm_anova`.
      - Both a between and a within factor → `stat_mixed_anova`.
      - Badly non-normal → `stat_nonparametric_k`.

    A significant omnibus F says only that *some* groups differ, not which. Always
    follow with `stat_pairwise_tests` and a multiple-comparison correction.

    Args:
        ss_type: Sums-of-squares type; 2 (default, recommended for unbalanced designs) | 1 | 3.
        effsize: np2 (partial eta-squared) | n2 (eta-squared).
    """
    return await _exec(ops.stat_anova, name, dv, between, ss_type, effsize)


@mcp.tool()
async def stat_welch_anova(name: str, dv: str, between: str, ctx: Context = None) -> str:
    """Welch's ANOVA — three or more groups without the equal-variance assumption.

    WHEN TO USE: Levene's test is significant, or group sizes and variances differ.
    WHEN NOT TO USE: when variances are equal — the standard ANOVA is more powerful.

    Note: Welch's ANOVA does not yield the usual η²; report ω² or use pairwise
    effect sizes with a correction.
    """
    return await _exec(ops.stat_welch_anova, name, dv, between)


@mcp.tool()
async def stat_rm_anova(name: str, dv: str, within: str, subject: str,
                        correction: str = "auto", effsize: str = "ng2",
                        ctx: Context = None) -> str:
    """Repeated-measures ANOVA — one within-subjects factor with three or more levels.

    WHEN TO USE: the same participants measured under 3+ conditions.
    WHEN NOT TO USE:
      - Only two conditions → a paired t-test is simpler and equivalent.
      - Missing data → a linear mixed-effects model uses all available data,
        whereas RM-ANOVA deletes whole cases.
      - Badly non-normal → Friedman test.

    Always check sphericity first (`stat_sphericity_test`) and report the
    corrected p-value if it is violated.

    Args:
        correction: auto (recommended) | True (always Greenhouse-Geisser) | False (never).
        effsize: ng2 (generalized eta-squared, default) | np2.
    """
    return await _exec(ops.stat_rm_anova, name, dv, within, subject, correction, effsize)


@mcp.tool()
async def stat_mixed_anova(name: str, dv: str, within: str, subject: str, between: str,
                           correction: str = "auto", effsize: str = "np2",
                           ctx: Context = None) -> str:
    """Mixed-design ANOVA — one between-subjects factor crossed with one within-subjects factor.

    WHEN TO USE: e.g. two groups measured at three time points; group × time designs.

    Read the interaction first. A significant interaction means the effect of one
    factor depends on the level of the other, and the main effects must not be
    interpreted in isolation.
    """
    return await _exec(ops.stat_mixed_anova, name, dv, within, subject, between,
                       correction, effsize)


@mcp.tool()
async def stat_ancova(name: str, dv: str, between: str, covar: str,
                      effsize: str = "np2", ctx: Context = None) -> str:
    """ANCOVA — compare groups while statistically controlling a continuous covariate.

    WHEN TO USE: a continuous variable is expected to influence the outcome and
    differs between groups (e.g. age, baseline score).

    WHEN NOT TO USE — two conditions that are frequently violated:
      - The covariate was affected by the manipulation. Controlling a variable
        the treatment changed removes part of the effect you are measuring.
      - The covariate was measured after the outcome, or with substantial error.

    The tool tests homogeneity of regression slopes automatically and tells you if
    ANCOVA is inappropriate.
    """
    return await _exec(ops.stat_ancova, name, dv, between, covar, effsize)


@mcp.tool()
async def stat_nonparametric_k(name: str, dv: str, between: str = None,
                               within: str = None, subject: str = None,
                               ctx: Context = None) -> str:
    """Kruskal-Wallis (independent) or Friedman (repeated) — three or more groups, no normality assumption.

    WHEN TO USE: 3+ groups with ordinal outcomes, or continuous outcomes where
    normality is badly violated.
    WHEN NOT TO USE: when the parametric assumptions hold — the rank test is less
    powerful.

    Follow a significant result with `stat_pairwise_tests(parametric=False)` and
    a correction.
    """
    return await _exec(ops.stat_nonparametric_k, name, dv, between, within, subject)


# ══════════════════════════════════════════════════════════════════════════════
# Post-hoc and multiple comparisons
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_pairwise_tests(name: str, dv: str, between: str = None,
                              within: str = None, subject: str = None,
                              parametric: bool = True, padjust: str = "none",
                              effsize: str = "hedges", alpha: float = 0.05,
                              interaction: bool = True, ctx: Context = None) -> str:
    """Pairwise comparisons with optional multiple-comparison correction.

    WHEN TO USE
      - After a significant ANOVA / RM-ANOVA / mixed ANOVA, to identify which pairs differ.
      - For planned pairwise comparisons, including for main effects and interactions.

    WHEN NOT TO USE: as an exploratory scan of every possible pair without a
    correction — that is p-hacking.

    Set `padjust` to control the error rate. Reporting only the significant pairs
    from an uncorrected set is the most common mistake in post-hoc reporting; the
    tool warns when `padjust='none'` with multiple comparisons.

    Args:
        parametric: True for t-based tests, False for Mann-Whitney/Wilcoxon pairs.
        padjust: none | bonf | sidak | holm | fdr_bh | fdr_by.
                 Use 'fdr_bh' for many exploratory comparisons, 'holm' for a few planned ones.
        effsize: hedges (default) | cohen | r.
    """
    return await _exec(ops.stat_pairwise_tests, name, dv, between, within, subject,
                       parametric, padjust, effsize, alpha, interaction)


@mcp.tool()
async def stat_pairwise_tukey(name: str, dv: str, between: str, method: str = "tukey",
                              ctx: Context = None) -> str:
    """Tukey HSD or Games-Howell post-hoc tests after a one-way ANOVA.

    WHEN TO USE
      - Tukey HSD: equal variances and roughly equal group sizes (the standard default).
      - Games-Howell (`method="gameshowell"`): unequal variances or unequal group sizes —
        i.e. whenever Levene's test is significant.

    WHEN NOT TO USE: for within-subjects designs — use `stat_pairwise_tests` with
    the `subject` argument instead.
    """
    return await _exec(ops.stat_pairwise_tukey, name, dv, between, method)


@mcp.tool()
async def stat_multicomp_correction(pvals: list[float], alpha: float = 0.05,
                                    method: str = "holm", ctx: Context = None) -> str:
    """Correct a list of p-values for multiple comparisons.

    WHEN TO USE: when you have a set of p-values computed outside Pingouin, or
    when applying a correction across several tests reported together.

    Choosing a method:
      - holm  — controls family-wise error rate; more powerful than Bonferroni. Default.
      - fdr_bh — controls false discovery rate; for many comparisons or exploratory work.
      - bonf  — most conservative; only when any false positive is unacceptable.
      - none  — no correction; only for a single pre-registered comparison.

    State the method by name in the write-up; "we corrected for multiple
    comparisons" without naming the method is not reproducible.
    """
    return await _exec(ops.stat_multicomp_correction, pvals, alpha, method)


# ══════════════════════════════════════════════════════════════════════════════
# Association
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_correlation(name: str, x: str, y: str, method: str = "pearson",
                           alternative: str = "two-sided", ctx: Context = None) -> str:
    """Correlation between two continuous variables.

    WHEN TO USE
      - pearson: both continuous, relationship linear, no severe outliers.
      - spearman: ordinal data, monotonic but non-linear relationships, or outliers.
      - kendall: small n with many ties.
      - percbend / shepherds / bicor: robust alternatives when outliers are a concern.

    WHEN NOT TO USE
      - Repeated measurements on the same participants → `stat_rm_correlation`; a plain
        correlation is inflated by between-subject variance.
      - To establish agreement between two measurement methods → `stat_plot_blandaltman`;
        correlation measures association, not agreement.
      - To claim "no relationship" from a non-significant r → check
        `stat_distance_correlation`, which detects non-linear dependence.

    Always report r (or rho) with its CI, not just the p-value.
    """
    return await _exec(ops.stat_correlation, name, x, y, method, alternative)


@mcp.tool()
async def stat_partial_correlation(name: str, x: str, y: str, covar: list[str],
                                   method: str = "pearson", ctx: Context = None) -> str:
    """Partial correlation — association between two variables after removing covariates.

    WHEN TO USE: a third variable plausibly drives both x and y, and you want the
    association net of it.

    WHEN NOT TO USE: as a substitute for modelling. If several covariates are
    involved, regression gives a clearer picture. Partial correlation also assumes
    the covariate is measured without error.
    """
    return await _exec(ops.stat_partial_correlation, name, x, y, covar, method)


@mcp.tool()
async def stat_rm_correlation(name: str, x: str, y: str, subject: str,
                              ctx: Context = None) -> str:
    """Repeated-measures correlation — the correct correlation for within-subject data.

    WHEN TO USE: whenever each participant contributes more than one observation
    and you want the association between x and y.

    WHEN NOT TO USE: when each participant appears once — use `stat_correlation`.

    Why it matters: a standard Pearson correlation on repeated measures is
    inflated by between-subject differences. rm_corr reports the within-subject
    association, which is usually the quantity of interest, and tests whether
    within- and between-subject slopes differ.
    """
    return await _exec(ops.stat_rm_correlation, name, x, y, subject)


@mcp.tool()
async def stat_correlation_matrix(name: str, columns: list[str], method: str = "pearson",
                                  padjust: str = None, decimals: int = 3,
                                  ctx: Context = None) -> str:
    """Correlation matrix across several variables, optionally with p-value correction.

    WHEN TO USE: exploratory screening across a set of variables, or a correlation
    table for a paper.

    WHEN NOT TO USE: as a basis for causal claims, or when the number of variables
    is large relative to n — correlations become unstable.

    With more than two variables, always pass `padjust` (e.g. "fdr_bh"). A matrix
    of 10 variables implies 45 comparisons; several will be significant by chance
    alone.
    """
    return await _exec(ops.stat_correlation_matrix, name, columns, method, padjust, decimals)


@mcp.tool()
async def stat_distance_correlation(name: str, x: str, y: str, n_boot: int = 1000,
                                    seed: int = 42, ctx: Context = None) -> str:
    """Distance correlation — detects any dependence, linear or not.

    WHEN TO USE
      - Before concluding "no relationship" from a non-significant Pearson r.
        Distance correlation is zero if and only if the variables are independent.
      - When a non-linear or non-monotonic relationship is plausible.

    WHEN NOT TO USE: as a replacement for interpretable effect sizes — the
    coefficient has no simple direction or magnitude reading.

    Bootstrap-based, so it is slower; reduce `n_boot` for exploratory passes.
    """
    return await _exec(ops.stat_distance_correlation, name, x, y, n_boot, seed)


# ══════════════════════════════════════════════════════════════════════════════
# Regression and mediation
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_linear_regression(name: str, y: str, x: list[str], alpha: float = 0.05,
                                 ctx: Context = None) -> str:
    """Ordinary least-squares regression with coefficients, CIs and model fit.

    WHEN TO USE: continuous outcome predicted from one or more continuous or
    dummy-coded predictors; or when adjusting for covariates in a continuous-outcome model.

    WHEN NOT TO USE:
      - Binary outcome → `stat_logistic_regression`.
      - Repeated measures / clustered data → a mixed-effects model; OLS standard
        errors will be wrong.
      - Severe multicollinearity → address it before interpreting coefficients.

    The output includes R² and adjusted R². Check the assumptions listed in the
    output before interpreting coefficients; a significant coefficient is not
    evidence of a causal effect.
    """
    return await _exec(ops.stat_linear_regression, name, y, x, alpha)


@mcp.tool()
async def stat_logistic_regression(name: str, y: str, x: list[str],
                                   alpha: float = 0.05, ctx: Context = None) -> str:
    """Binary logistic regression with odds ratios and confidence intervals.

    WHEN TO USE: a binary outcome (yes/no, present/absent) predicted from one or
    more variables.

    WHEN NOT TO USE:
      - Continuous outcome → `stat_linear_regression`.
      - Ordinal outcome with 3+ levels → an ordinal model, not this.
      - Rare events (fewer than ~10 outcomes per predictor) → the estimates are
        biased; use penalised regression (Firth).

    An odds ratio is not a risk ratio: for common outcomes it exaggerates the
    effect. Report predicted probabilities alongside when the substantive
    question is about risk.
    """
    return await _exec(ops.stat_logistic_regression, name, y, x, alpha)


@mcp.tool()
async def stat_mediation(name: str, x: str, m: str, y: str, covar: list[str] = None,
                         n_boot: int = 500, seed: int = 42, ctx: Context = None) -> str:
    """Mediation analysis — does x affect y through m?

    WHEN TO USE: a plausible causal chain x → m → y, with m measured after x and
    before y, and a theoretical reason to expect transmission.

    WHEN NOT TO USE
      - Cross-sectional data where the temporal order is only assumed. A
        significant indirect effect from cross-sectional data does not establish
        a mechanism; the output states this.
      - As a fishing expedition across many candidate mediators — with enough
        mediators, one will show a significant indirect effect by chance.

    The indirect effect's bootstrap CI excluding zero is the evidence for
    mediation. Report that CI, not the Sobel test.
    """
    return await _exec(ops.stat_mediation, name, x, m, y, covar, n_boot, seed)


# ══════════════════════════════════════════════════════════════════════════════
# Effect size, power and reliability
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_effect_size(name: str, dv: str, group: str, paired: bool = False,
                           subject: str = None, eftype: str = "cohen",
                           ctx: Context = None) -> str:
    """Effect size with a confidence interval for a two-group comparison.

    WHEN TO USE: alongside every t-test and every pairwise comparison. A p-value
    says whether an effect is distinguishable from zero; the effect size says
    whether it matters.

    WHEN NOT TO USE: for 3+ groups in a single call — use the effect sizes
    reported by ANOVA or by `stat_pairwise_tests`.

    Note on paired designs: the standardiser is the SD of the difference scores,
    not the pooled SD. Cohen's d and dz are different quantities — state which
    you report.

    Args:
        eftype: cohen (d) | hedges (small-sample correction) | glass | r | eta-squared.
    """
    return await _exec(ops.stat_effect_size, name, dv, group, paired, subject, eftype)


@mcp.tool()
async def stat_power_analysis(test: str = "ttest", d: float = None, eta_squared: float = None,
                              r: float = None, w: float = None, n: float = None,
                              power: float = None, alpha: float = 0.05,
                              k: int = None, m: int = None, contrast: str = "two-samples",
                              ctx: Context = None) -> str:
    """Power analysis — solve for whichever of effect size / n / power / alpha is omitted.

    WHEN TO USE
      - Before data collection, to justify the sample size (a priori power analysis).
      - To report the achieved power of a completed study.

    WHEN NOT TO USE: to compute "observed power" from the current sample's effect
    size and then interpret it — that is circular and adds nothing to the p-value.

    Supply exactly three of the four quantities; the fourth is solved for.

    Args:
        test: ttest | ttest2n | anova | rm_anova | corr | chi2.
        d / eta_squared / r / w: effect size, matching the test family.
        n: sample size per group.
        power: target power (0.80 is the conventional threshold).
        alpha: significance level.
        k: number of groups (ANOVA).
        m: number of repeated measurements (RM-ANOVA).
        contrast: two-samples | one-sample | paired (for ttest).

    A priori power analysis should justify the effect size from prior literature
    or theory, not from the current data.
    """
    kwargs = {k_: v for k_, v in dict(
        d=d, eta_squared=eta_squared, r=r, w=w, n=n, power=power,
        alpha=alpha, k=k, m=m, contrast=contrast).items() if v is not None}
    return await _exec(ops.stat_power_analysis, test, **kwargs)


@mcp.tool()
async def stat_cronbach_alpha(name: str, items: list[str], ci: float = 0.95,
                              ctx: Context = None) -> str:
    """Cronbach's alpha — internal consistency of a multi-item scale.

    WHEN TO USE: reporting the reliability of a questionnaire or scale in the
    current sample.

    WHEN NOT TO USE
      - As the sole evidence of unidimensionality — alpha can be high for a
        multidimensional scale. Confirm with factor analysis.
      - For binary items (use KR-20) or for inter-rater agreement (use ICC).

    Alpha rises with the number of items regardless of their quality: report the
    item count and the CI, and consider the mean inter-item correlation.
    """
    return await _exec(ops.stat_cronbach_alpha, name, items, ci)


@mcp.tool()
async def stat_intraclass_correlation(name: str, targets: str, raters: str,
                                      ratings: str, icc_type: str = "ICC2",
                                      ctx: Context = None) -> str:
    """Intraclass correlation — agreement or consistency between raters or measurements.

    WHEN TO USE: inter-rater reliability, test-retest reliability, agreement
    between measurement devices.

    WHEN NOT TO USE: as a substitute for Bland-Altman when the question is
    whether two methods agree in absolute terms — ICC is bounded by the range of
    the sample and can look excellent while agreement is poor.

    Choosing the form matters and is often got wrong:
      - ICC2 — absolute agreement, raters treated as random and interchangeable.
      - ICC3 — consistency only (do raters rank the same way?), raters fixed.
      - the "k" variants — reliability of the *mean* of k raters, which is higher.
    """
    return await _exec(ops.stat_intraclass_correlation, name, targets, raters,
                       ratings, icc_type)


@mcp.tool()
async def stat_equivalence_test(name: str, dv: str, group: str, bound: float,
                                paired: bool = False, subject: str = None,
                                ctx: Context = None) -> str:
    """TOST equivalence test — for *demonstrating* that two groups are equivalent.

    WHEN TO USE: the research claim is that there is no meaningful difference
    (e.g. a new measure matches an established one).

    WHEN NOT TO USE: as a post-hoc rescue of a non-significant t-test. TOST
    answers a different question and needs its bound specified in advance.

    A non-significant difference is not evidence of equivalence — absence of
    evidence is not evidence of absence. The `bound` must be justified on
    substantive grounds (the largest difference that would still be negligible),
    ideally before data collection.
    """
    return await _exec(ops.stat_equivalence_test, name, dv, group, bound, paired, subject)


# ══════════════════════════════════════════════════════════════════════════════
# Categorical data
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_chi2_independence(name: str, x: str, y: str, correction: bool = True,
                                 ctx: Context = None) -> str:
    """Chi-square test of independence between two categorical variables.

    WHEN TO USE: two categorical variables from independent observations.

    WHEN NOT TO USE
      - The same participants measured twice on a categorical variable →
        `stat_chi2_mcnemar`.
      - Any expected cell count < 5 → Fisher's exact test (the tool flags this).

    Report Cramér's V as the effect size.
    """
    return await _exec(ops.stat_chi2_independence, name, x, y, correction)


@mcp.tool()
async def stat_chi2_mcnemar(name: str, x: str, y: str, correction: bool = True,
                            ctx: Context = None) -> str:
    """McNemar's test — paired categorical data (before/after, matched pairs, twins).

    WHEN TO USE: the same units contribute a categorical value under two
    conditions (pre/post intervention, two raters, matched pairs).

    WHEN NOT TO USE: for independent groups — use `stat_chi2_independence`.

    Only the two discordant cells contribute to the statistic.
    """
    return await _exec(ops.stat_chi2_mcnemar, name, x, y, correction)


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_plot_paired(name: str, dv: str, within: str, subject: str,
                           boxplot: bool = True, ctx: Context = None) -> str:
    """Paired-data plot — one line per participant across conditions.

    WHEN TO USE: reporting any within-subjects comparison. It shows the
    participant-level change that the test actually analyses.

    WHEN NOT TO USE: for independent groups — a boxplot or raincloud plot is
    appropriate there.

    A bar chart of means hides participants moving in opposite directions, which
    is exactly the pattern that makes a mean difference misleading.
    """
    return await _exec(ops.stat_plot_paired, name, dv, within, subject, boxplot)


@mcp.tool()
async def stat_plot_blandaltman(name: str, x: str, y: str, agreement: float = 1.96,
                                ctx: Context = None) -> str:
    """Bland-Altman plot — agreement between two measurement methods.

    WHEN TO USE: method comparison studies, test-retest agreement, comparing a
    new device or scale against a criterion.

    WHEN NOT TO USE: when the question is association rather than agreement — for
    that, use a scatter plot with a correlation.
    """
    return await _exec(ops.stat_plot_blandaltman, name, x, y, agreement)


@mcp.tool()
async def stat_qqplot(name: str, column: str, dist: str = "norm", ctx: Context = None) -> str:
    """Quantile-quantile plot for judging distributional shape.

    WHEN TO USE: alongside every normality test. The plot shows *how* the data
    depart from normality, which the test's p-value cannot tell you.

    WHEN NOT TO USE: as the only check — report the test statistic too.
    """
    return await _exec(ops.stat_qqplot, name, column, dist)


@mcp.tool()
async def stat_plot_rm_corr(name: str, x: str, y: str, subject: str,
                            ctx: Context = None) -> str:
    """Plot for a repeated-measures correlation, with a per-participant fit.

    WHEN TO USE: alongside `stat_rm_correlation`, to show whether the
    within-subject relationship is consistent across participants.
    """
    return await _exec(ops.stat_plot_rm_corr, name, x, y, subject)


# ══════════════════════════════════════════════════════════════════════════════
# Session
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
async def stat_session_info(ctx: Context = None) -> str:
    """List the datasets and results currently held in the session.

    Call this when unsure what data are available, or after loading several
    datasets, to confirm the names the other tools expect.
    """
    return await _exec(ops.stat_session_info)


@mcp.tool()
async def stat_reset_session(ctx: Context = None) -> str:
    """Clear all datasets, results and figures, starting a fresh session.

    WHEN TO USE: before starting a genuinely separate analysis, to avoid reading
    stale objects.
    WHEN NOT TO USE: casually — loaded data are lost.
    """
    return await _exec(ops.stat_reset_session)


@mcp.tool()
async def stat_run_code(code: str, ctx: Context = None) -> str:
    """Execute arbitrary Python against the session namespace — the escape hatch.

    Use this for anything the structured tools do not cover: custom recoding,
    a Pingouin function not exposed as a tool, or a bespoke plot.

    Pre-bound names: `pg` (pingouin), `pd` (pandas), `np` (numpy), `plt`
    (matplotlib), `sns` (seaborn), plus every loaded object by name.

    The value of a trailing expression is returned. Matplotlib figures are saved
    automatically and their paths returned.

    WHEN TO USE: beyond the structured tools.
    WHEN NOT TO USE: for routine analyses — the structured tools report
    assumptions and effect sizes that a bare call would omit.
    """
    return await _exec(ops.stat_run_code, code)


def main() -> None:
    """Console entry point."""
    mcp.run()


if __name__ == "__main__":
    main()
