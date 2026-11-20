# Pingouin-MCP — Tool Reference

**43 tools** across 12 groups, organised by research task rather than by Pingouin function name.

> Generated from the running server by `scripts/gen_tools_reference.py`. Do not edit by hand — the file is regenerated from the registered tools and their docstrings.

Every tool returns Markdown for the agent to relay, the equivalent Pingouin code for reproduction, and (where applicable) the path to a saved figure.

---

## Data management

### `stat_load_data`

Load a tabular dataset (CSV/TSV/Excel/JSON/Parquet/SPSS) into the session.

Use this as the first step for any dataset stored as a file. The loaded
object is stored under `name` and every other tool refers to it by that name.

When to use: at the start of an analysis session, once per dataset.
When NOT to use: if the data are already in the session (check with
`stat_session_info` first), or if you need to construct a small example —
use `stat_make_dataframe` instead.

| Parameter | Type | Default |
|---|---|---|
| `path` | str | — |
| `name` | str | 'df' |
| `read_kwargs` | — | — |

### `stat_make_dataframe`

Build a DataFrame from a {column: [values]} mapping.

Use this for worked examples, sanity checks, and illustrating a method with
hypothetical data.

When NOT to use: never for real analysis — fabricated data must not enter an
analysis of real results.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `columns` | dict | — |

### `stat_describe_data`

Descriptive statistics, optionally broken down by group.

Use this to report the sample before any inference: means, SDs, medians and
group sizes are expected in every results section.

When to use: immediately after loading, and again when writing up.
When NOT to use: as a substitute for checking assumptions — use
`stat_data_quality_check` and `stat_assumption_report` for that.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | None | None |
| `group` | str | None | None |

### `stat_data_quality_check`

Screen a dataset for missingness, outliers and distribution shape.

Run this on every new dataset before planning any analysis: it determines
whether the planned analysis is even possible (e.g. a between-subjects test
needs enough complete cases per group).

When to use: first thing after loading real data.
Returns: missing counts per column, outliers flagged by the robust MAD rule,
skewness and kurtosis.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `columns` | list[str] | None | None |

---

## Decision support

### `stat_decision_assistant`

Recommend which statistical test fits a study design — call this BEFORE running a test.

This exists because choosing the wrong test is the most consequential error
in statistical analysis, and it invalidates the p-value rather than merely
producing a suboptimal number. If you are not certain which test applies,
call this first.

| Parameter | Type | Default |
|---|---|---|
| `dv_type` | str | 'continuous' |
| `n_groups` | int | None | None |
| `repeated` | bool | None | None |
| `n_dv` | int | 1 |
| `has_covariate` | bool | False |
| `normality_ok` | bool | None | None |
| `homogeneity_ok` | bool | None | None |
| `sphericity_ok` | bool | None | None |
| `predicting` | bool | False |
| `n_subjects` | int | None | None |

---

## Assumption testing

### `stat_assumption_report`

Run every assumption check implied by a design in one pass.

Use this instead of checking assumptions one at a time, so that none is
silently skipped. Report the output whether or not the assumptions pass — a
check reported only when it fails is not a check.

When to use: after loading and before the main test, for any parametric test.
When NOT to use: for purely non-parametric analyses (normality is irrelevant
there), or for descriptive work.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `between` | str | None | None |
| `within` | str | None | None |
| `subject` | str | None | None |
| `covar` | str | None | None |

### `stat_normality_test`

Test whether a variable is normally distributed, optionally within each group.

When to use: before any parametric test (t-test, ANOVA, Pearson correlation).
When NOT to use: as a gatekeeper that alone decides parametric vs
non-parametric. With large n the test detects trivial deviations; with small
n it has low power. Judge with the QQ plot and skewness too.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `group` | str | None | None |
| `method` | str | 'shapiro' |

### `stat_homogeneity_test`

Test equality of variances across groups.

When to use: before an independent-samples t-test or a one-way ANOVA.
When it fails: use Welch's t-test or Welch's ANOVA, which do not assume
equal variances.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `group` | str | — |
| `method` | str | 'levene' |

### `stat_sphericity_test`

Mauchly's test of sphericity for a repeated-measures factor with 3+ levels.

When to use: before a repeated-measures ANOVA with three or more conditions.
When it fails: report the Greenhouse-Geisser corrected p-value and the
epsilon. The uncorrected F is anti-conservative when sphericity is violated.

Not applicable with only two levels — a paired t-test needs no sphericity
assumption.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `within` | str | — |
| `subject` | str | — |
| `method` | str | 'mauchly' |

---

## Two-group comparisons

### `stat_ttest`

t-test: independent samples, paired, or one-sample.

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

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `group` | str | None | None |
| `paired` | bool | False |
| `subject` | str | None | None |
| `x` | str | None | None |
| `y` | str | None | None |
| `alternative` | str | 'two-sided' |
| `correction` | str | 'auto' |
| `confidence` | float | 0.95 |

### `stat_nonparametric_2`

Mann-Whitney U (independent) or Wilcoxon signed-rank (paired) — two groups, no normality assumption.

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

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `group` | str | None | None |
| `paired` | bool | False |
| `subject` | str | None | None |
| `x` | str | None | None |
| `y` | str | None | None |
| `alternative` | str | 'two-sided' |

### `stat_bayes_ttest`

Bayesian t-test — quantifies evidence for the null as well as the alternative.

WHEN TO USE
  - When a non-significant result needs interpreting. A frequentist p > .05
    does not mean the null is true; BF₀₁ directly quantifies evidence for it.
  - When the researcher wants to state "the groups are comparable".

WHEN NOT TO USE
  - As a way to find significance after a non-significant t-test. Decide on
    the analysis approach in advance.

Report BF₁₀ and BF₀₁, and state the prior scale (r).

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `group` | str | — |
| `paired` | bool | False |
| `subject` | str | None | None |
| `alternative` | str | 'two-sided' |
| `r` | float | 0.707 |

---

## Multi-group comparisons

### `stat_anova`

One-way ANOVA — three or more independent groups, continuous outcome.

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

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `between` | str | — |
| `ss_type` | int | 2 |
| `effsize` | str | 'np2' |

### `stat_welch_anova`

Welch's ANOVA — three or more groups without the equal-variance assumption.

WHEN TO USE: Levene's test is significant, or group sizes and variances differ.
WHEN NOT TO USE: when variances are equal — the standard ANOVA is more powerful.

Note: Welch's ANOVA does not yield the usual η²; report ω² or use pairwise
effect sizes with a correction.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `between` | str | — |

### `stat_rm_anova`

Repeated-measures ANOVA — one within-subjects factor with three or more levels.

WHEN TO USE: the same participants measured under 3+ conditions.
WHEN NOT TO USE:
  - Only two conditions → a paired t-test is simpler and equivalent.
  - Missing data → a linear mixed-effects model uses all available data,
    whereas RM-ANOVA deletes whole cases.
  - Badly non-normal → Friedman test.

Always check sphericity first (`stat_sphericity_test`) and report the
corrected p-value if it is violated.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `within` | str | — |
| `subject` | str | — |
| `correction` | str | 'auto' |
| `effsize` | str | 'ng2' |

### `stat_mixed_anova`

Mixed-design ANOVA — one between-subjects factor crossed with one within-subjects factor.

WHEN TO USE: e.g. two groups measured at three time points; group × time designs.

Read the interaction first. A significant interaction means the effect of one
factor depends on the level of the other, and the main effects must not be
interpreted in isolation.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `within` | str | — |
| `subject` | str | — |
| `between` | str | — |
| `correction` | str | 'auto' |
| `effsize` | str | 'np2' |

### `stat_ancova`

ANCOVA — compare groups while statistically controlling a continuous covariate.

WHEN TO USE: a continuous variable is expected to influence the outcome and
differs between groups (e.g. age, baseline score).

WHEN NOT TO USE — two conditions that are frequently violated:
  - The covariate was affected by the manipulation. Controlling a variable
    the treatment changed removes part of the effect you are measuring.
  - The covariate was measured after the outcome, or with substantial error.

The tool tests homogeneity of regression slopes automatically and tells you if
ANCOVA is inappropriate.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `between` | str | — |
| `covar` | str | — |
| `effsize` | str | 'np2' |

### `stat_nonparametric_k`

Kruskal-Wallis (independent) or Friedman (repeated) — three or more groups, no normality assumption.

WHEN TO USE: 3+ groups with ordinal outcomes, or continuous outcomes where
normality is badly violated.
WHEN NOT TO USE: when the parametric assumptions hold — the rank test is less
powerful.

Follow a significant result with `stat_pairwise_tests(parametric=False)` and
a correction.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `between` | str | None | None |
| `within` | str | None | None |
| `subject` | str | None | None |

---

## Post-hoc and multiple comparisons

### `stat_pairwise_tests`

Pairwise comparisons with optional multiple-comparison correction.

WHEN TO USE
  - After a significant ANOVA / RM-ANOVA / mixed ANOVA, to identify which pairs differ.
  - For planned pairwise comparisons, including for main effects and interactions.

WHEN NOT TO USE: as an exploratory scan of every possible pair without a
correction — that is p-hacking.

Set `padjust` to control the error rate. Reporting only the significant pairs
from an uncorrected set is the most common mistake in post-hoc reporting; the
tool warns when `padjust='none'` with multiple comparisons.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `between` | str | None | None |
| `within` | str | None | None |
| `subject` | str | None | None |
| `parametric` | bool | True |
| `padjust` | str | 'none' |
| `effsize` | str | 'hedges' |
| `alpha` | float | 0.05 |
| `interaction` | bool | True |

### `stat_pairwise_tukey`

Tukey HSD or Games-Howell post-hoc tests after a one-way ANOVA.

WHEN TO USE
  - Tukey HSD: equal variances and roughly equal group sizes (the standard default).
  - Games-Howell (`method="gameshowell"`): unequal variances or unequal group sizes —
    i.e. whenever Levene's test is significant.

WHEN NOT TO USE: for within-subjects designs — use `stat_pairwise_tests` with
the `subject` argument instead.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `between` | str | — |
| `method` | str | 'tukey' |

### `stat_multicomp_correction`

Correct a list of p-values for multiple comparisons.

WHEN TO USE: when you have a set of p-values computed outside Pingouin, or
when applying a correction across several tests reported together.

Choosing a method:
  - holm  — controls family-wise error rate; more powerful than Bonferroni. Default.
  - fdr_bh — controls false discovery rate; for many comparisons or exploratory work.
  - bonf  — most conservative; only when any false positive is unacceptable.
  - none  — no correction; only for a single pre-registered comparison.

State the method by name in the write-up; "we corrected for multiple
comparisons" without naming the method is not reproducible.

| Parameter | Type | Default |
|---|---|---|
| `pvals` | list[float] | — |
| `alpha` | float | 0.05 |
| `method` | str | 'holm' |

---

## Association

### `stat_correlation`

Correlation between two continuous variables.

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

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `y` | str | — |
| `method` | str | 'pearson' |
| `alternative` | str | 'two-sided' |

### `stat_partial_correlation`

Partial correlation — association between two variables after removing covariates.

WHEN TO USE: a third variable plausibly drives both x and y, and you want the
association net of it.

WHEN NOT TO USE: as a substitute for modelling. If several covariates are
involved, regression gives a clearer picture. Partial correlation also assumes
the covariate is measured without error.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `y` | str | — |
| `covar` | list[str] | — |
| `method` | str | 'pearson' |

### `stat_rm_correlation`

Repeated-measures correlation — the correct correlation for within-subject data.

WHEN TO USE: whenever each participant contributes more than one observation
and you want the association between x and y.

WHEN NOT TO USE: when each participant appears once — use `stat_correlation`.

Why it matters: a standard Pearson correlation on repeated measures is
inflated by between-subject differences. rm_corr reports the within-subject
association, which is usually the quantity of interest, and tests whether
within- and between-subject slopes differ.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `y` | str | — |
| `subject` | str | — |

### `stat_correlation_matrix`

Correlation matrix across several variables, optionally with p-value correction.

WHEN TO USE: exploratory screening across a set of variables, or a correlation
table for a paper.

WHEN NOT TO USE: as a basis for causal claims, or when the number of variables
is large relative to n — correlations become unstable.

With more than two variables, always pass `padjust` (e.g. "fdr_bh"). A matrix
of 10 variables implies 45 comparisons; several will be significant by chance
alone.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `columns` | list[str] | — |
| `method` | str | 'pearson' |
| `padjust` | str | None | None |
| `decimals` | int | 3 |

### `stat_distance_correlation`

Distance correlation — detects any dependence, linear or not.

WHEN TO USE
  - Before concluding "no relationship" from a non-significant Pearson r.
    Distance correlation is zero if and only if the variables are independent.
  - When a non-linear or non-monotonic relationship is plausible.

WHEN NOT TO USE: as a replacement for interpretable effect sizes — the
coefficient has no simple direction or magnitude reading.

Bootstrap-based, so it is slower; reduce `n_boot` for exploratory passes.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `y` | str | — |
| `n_boot` | int | 1000 |
| `seed` | int | None | 42 |

---

## Regression and mediation

### `stat_linear_regression`

Ordinary least-squares regression with coefficients, CIs and model fit.

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

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `y` | str | — |
| `x` | list[str] | — |
| `alpha` | float | 0.05 |

### `stat_logistic_regression`

Binary logistic regression with odds ratios and confidence intervals.

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

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `y` | str | — |
| `x` | list[str] | — |
| `alpha` | float | 0.05 |

### `stat_mediation`

Mediation analysis — does x affect y through m?

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

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `m` | str | — |
| `y` | str | — |
| `covar` | list[str] | None | None |
| `n_boot` | int | 500 |
| `seed` | int | None | 42 |

---

## Effect size, power, reliability

### `stat_effect_size`

Effect size with a confidence interval for a two-group comparison.

WHEN TO USE: alongside every t-test and every pairwise comparison. A p-value
says whether an effect is distinguishable from zero; the effect size says
whether it matters.

WHEN NOT TO USE: for 3+ groups in a single call — use the effect sizes
reported by ANOVA or by `stat_pairwise_tests`.

Note on paired designs: the standardiser is the SD of the difference scores,
not the pooled SD. Cohen's d and dz are different quantities — state which
you report.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `group` | str | — |
| `paired` | bool | False |
| `subject` | str | None | None |
| `eftype` | str | 'cohen' |

### `stat_power_analysis`

Power analysis — solve for whichever of effect size / n / power / alpha is omitted.

WHEN TO USE
  - Before data collection, to justify the sample size (a priori power analysis).
  - To report the achieved power of a completed study.

WHEN NOT TO USE: to compute "observed power" from the current sample's effect
size and then interpret it — that is circular and adds nothing to the p-value.

Supply exactly three of the four quantities; the fourth is solved for.

| Parameter | Type | Default |
|---|---|---|
| `test` | str | 'ttest' |
| `kwargs` | — | — |

### `stat_cronbach_alpha`

Cronbach's alpha — internal consistency of a multi-item scale.

WHEN TO USE: reporting the reliability of a questionnaire or scale in the
current sample.

WHEN NOT TO USE
  - As the sole evidence of unidimensionality — alpha can be high for a
    multidimensional scale. Confirm with factor analysis.
  - For binary items (use KR-20) or for inter-rater agreement (use ICC).

Alpha rises with the number of items regardless of their quality: report the
item count and the CI, and consider the mean inter-item correlation.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `items` | list[str] | — |
| `ci` | float | 0.95 |

### `stat_intraclass_correlation`

Intraclass correlation — agreement or consistency between raters or measurements.

WHEN TO USE: inter-rater reliability, test-retest reliability, agreement
between measurement devices.

WHEN NOT TO USE: as a substitute for Bland-Altman when the question is
whether two methods agree in absolute terms — ICC is bounded by the range of
the sample and can look excellent while agreement is poor.

Choosing the form matters and is often got wrong:
  - ICC2 — absolute agreement, raters treated as random and interchangeable.
  - ICC3 — consistency only (do raters rank the same way?), raters fixed.
  - the "k" variants — reliability of the *mean* of k raters, which is higher.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `targets` | str | — |
| `raters` | str | — |
| `ratings` | str | — |
| `icc_type` | str | 'ICC2' |

### `stat_equivalence_test`

TOST equivalence test — for *demonstrating* that two groups are equivalent.

WHEN TO USE: the research claim is that there is no meaningful difference
(e.g. a new measure matches an established one).

WHEN NOT TO USE: as a post-hoc rescue of a non-significant t-test. TOST
answers a different question and needs its bound specified in advance.

A non-significant difference is not evidence of equivalence — absence of
evidence is not evidence of absence. The `bound` must be justified on
substantive grounds (the largest difference that would still be negligible),
ideally before data collection.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `group` | str | — |
| `bound` | float | — |
| `paired` | bool | False |
| `subject` | str | None | None |

---

## Categorical data

### `stat_chi2_independence`

Chi-square test of independence between two categorical variables.

WHEN TO USE: two categorical variables from independent observations.

WHEN NOT TO USE
  - The same participants measured twice on a categorical variable →
    `stat_chi2_mcnemar`.
  - Any expected cell count < 5 → Fisher's exact test (the tool flags this).

Report Cramér's V as the effect size.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `y` | str | — |
| `correction` | bool | True |

### `stat_chi2_mcnemar`

McNemar's test — paired categorical data (before/after, matched pairs, twins).

WHEN TO USE: the same units contribute a categorical value under two
conditions (pre/post intervention, two raters, matched pairs).

WHEN NOT TO USE: for independent groups — use `stat_chi2_independence`.

Only the two discordant cells contribute to the statistic.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `y` | str | — |
| `correction` | bool | True |

---

## Visualization

### `stat_plot_paired`

Paired-data plot — one line per participant across conditions.

WHEN TO USE: reporting any within-subjects comparison. It shows the
participant-level change that the test actually analyses.

WHEN NOT TO USE: for independent groups — a boxplot or raincloud plot is
appropriate there.

A bar chart of means hides participants moving in opposite directions, which
is exactly the pattern that makes a mean difference misleading.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `dv` | str | — |
| `within` | str | — |
| `subject` | str | — |
| `boxplot` | bool | True |

### `stat_plot_blandaltman`

Bland-Altman plot — agreement between two measurement methods.

WHEN TO USE: method comparison studies, test-retest agreement, comparing a
new device or scale against a criterion.

WHEN NOT TO USE: when the question is association rather than agreement — for
that, use a scatter plot with a correlation.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `y` | str | — |
| `agreement` | float | 1.96 |

### `stat_qqplot`

Quantile-quantile plot for judging distributional shape.

WHEN TO USE: alongside every normality test. The plot shows *how* the data
depart from normality, which the test's p-value cannot tell you.

WHEN NOT TO USE: as the only check — report the test statistic too.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `column` | str | — |
| `dist` | str | 'norm' |

### `stat_plot_rm_corr`

Plot for a repeated-measures correlation, with a per-participant fit.

WHEN TO USE: alongside `stat_rm_correlation`, to show whether the
within-subject relationship is consistent across participants.

| Parameter | Type | Default |
|---|---|---|
| `name` | str | — |
| `x` | str | — |
| `y` | str | — |
| `subject` | str | — |

---

## Session

### `stat_session_info`

List the datasets and results currently held in the session.

Call this when unsure what data are available, or after loading several
datasets, to confirm the names the other tools expect.

_No parameters._

### `stat_reset_session`

Clear all datasets, results and figures, starting a fresh session.

WHEN TO USE: before starting a genuinely separate analysis, to avoid reading
stale objects.
WHEN NOT TO USE: casually — loaded data are lost.

_No parameters._

### `stat_run_code`

Execute arbitrary Python against the session namespace — the escape hatch.

Use this for anything the structured tools do not cover: custom recoding,
a Pingouin function not exposed as a tool, or a bespoke plot.

Pre-bound names: `pg` (pingouin), `pd` (pandas), `np` (numpy), `plt`
(matplotlib), `sns` (seaborn), plus every loaded object by name.

The value of a trailing expression is returned. Matplotlib figures are saved
automatically and their paths returned.

WHEN TO USE: beyond the structured tools.
WHEN NOT TO USE: for routine analyses — the structured tools report
assumptions and effect sizes that a bare call would omit.

| Parameter | Type | Default |
|---|---|---|
| `code` | str | — |

---

## Conventions

- **Session-scoped.** Datasets are referenced by name across calls. `stat_session_info` lists what is loaded.
- **Reproducible.** Every analysis returns the equivalent Pingouin code.
- **Assumptions are explicit.** Tools report the checks their test requires rather than silently assuming them.
- **Errors are messages, not exceptions.** A bad column or dataset name comes back as readable text so the caller can correct it.

## Version

Verified against **Pingouin 0.6.1**, pandas 3.x, Python 3.13.
Pingouin renamed its result columns from hyphenated to underscored in 0.6 (`p-val` → `p_val`); the server resolves both spellings.
