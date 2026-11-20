---
name: stat-analyst
description: >
  Run statistical analysis conversationally through the Pingouin MCP server: load a dataset, screen it
  for missingness and outliers, check assumptions, choose the right test from the study design, run it,
  get effect sizes with confidence intervals, correct for multiple comparisons, and plot the result.
  Covers t-tests, ANOVA (one-way, repeated-measures, mixed, ANCOVA), non-parametric tests, correlation,
  partial and repeated-measures correlation, regression, logistic regression, mediation, reliability
  (Cronbach alpha, ICC), power analysis, equivalence testing, Bayesian t-tests, chi-square and McNemar.
  Use when the user asks to analyse data statistically, compare groups, test a hypothesis, check
  assumptions, report effect sizes, or interpret a p-value. Triggers: statistics, t-test, ANOVA,
  regression, correlation, p-value, effect size, 统计, 显著性, 方差分析, 相关, 回归, 中介, 信度,
  效应量, 假设检验, 统计方法选择.
---

# Statistical Analyst

Drive statistical analysis conversationally through the Pingouin MCP server. The server keeps **one
persistent session**, so load a dataset once and build the analysis up step by step.

The server returns the **equivalent Pingouin code** with every result. Show it to the user when they
ask "how was this computed", and use it to move an analysis into a script.

## Quick start

```
stat_load_data path=... name=df          # 1. load
stat_data_quality_check name=df          # 2. LOOK: missingness, outliers, shape
stat_decision_assistant ...              # 3. decide the test BEFORE running one
stat_assumption_report name=df ...       # 4. check what the chosen test requires
stat_ttest / stat_anova / ...            # 5. run it
stat_effect_size ...                     # 6. report the magnitude, not just p
```

## The one rule that matters most

**Never run a test without first establishing that it is the right test.**

If the user says "run a t-test" but the design has three groups, or repeated measures, or violates
normality — say so. Run `stat_decision_assistant` when the design is at all unusual, and
`stat_assumption_report` before any parametric test. The output of those tools is meant to be shown
to the user, not swallowed.

If you cannot tell from the request whether the groups are independent or repeated, **ask**. That
distinction changes the test, and using the wrong one invalidates the p-value.

## Request → tools (routing)

| User wants | Call (in order) |
|---|---|
| "Is there a difference between two groups?" | `stat_assumption_report` → `stat_ttest` |
| "Same people, two conditions" | `stat_assumption_report` → `stat_ttest paired=true subject=...` |
| "Three or more groups" | `stat_assumption_report` → `stat_anova` → `stat_pairwise_tests padjust=fdr_bh` |
| "Same people, three or more conditions" | `stat_sphericity_test` → `stat_rm_anova` |
| "Two groups over time" | `stat_mixed_anova` |
| "Control for a covariate" | `stat_ancova` |
| "Data aren't normal" | `stat_nonparametric_2` / `stat_nonparametric_k` |
| "Are these related?" | `stat_correlation` (→ `stat_rm_correlation` if repeated) |
| "Related after controlling for X" | `stat_partial_correlation` |
| "Predict Y from X" | `stat_linear_regression` / `stat_logistic_regression` |
| "Does X affect Y through M?" | `stat_mediation` |
| "Is my scale reliable?" | `stat_cronbach_alpha` / `stat_intraclass_correlation` |
| "How many subjects do I need?" | `stat_power_analysis` |
| "Show there is NO difference" | `stat_equivalence_test` or `stat_bayes_ttest` |
| "Two categorical variables" | `stat_chi2_independence` / `stat_chi2_mcnemar` |
| Anything else | `stat_run_code` |

## Golden rules

- ✅ **Screen first.** `stat_data_quality_check` before anything else. Missingness determines whether
  the planned analysis is even possible.
- ✅ **Check assumptions and report them whether or not they pass.** An assumption reported only when
  it fails is not a check.
- ✅ **Always report an effect size with its CI**, not just a p-value. `stat_effect_size`,
  `stat_pairwise_tests` and the ANOVA tools all provide one.
- ✅ **Name the correction method.** "We corrected for multiple comparisons" is not reproducible;
  `padjust="fdr_bh"` is.
- ✅ **State the direction of a difference.** Pingouin orders group levels alphabetically; the sign of
  t and d depends on that order. The t-test output states it — relay it.
- ❌ **Don't run multiple t-tests instead of an ANOVA.** With three groups that inflates the Type I
  error rate.
- ❌ **Don't set a peak-amplitude rejection threshold** without reporting how much data it discarded
  and whether it fell equally on all conditions.
- ❌ **Don't interpret a non-significant result as "no effect".** That is what `stat_equivalence_test`
  and `stat_bayes_ttest` are for.
- ❌ **Don't compute "observed power" from the current sample's effect size** and present it as
  evidence — it is a monotone function of the p-value and adds nothing.

## Worked example

> User: "比较实验组和对照组的焦虑得分，各 24 人"

1. `stat_data_quality_check name=df` → note missingness and the two outliers in `anxiety`
2. `stat_decision_assistant dv_type=continuous n_groups=2 repeated=false`
   → recommends the independent-samples t-test, lists normality and Levene's as prerequisites
3. `stat_assumption_report name=df dv=anxiety between=group`
   → normality met; **Levene's test significant (p = .01)**
4. Because homogeneity failed, run `stat_ttest name=df dv=anxiety group=group correction=true`
   → Welch's t-test, reported by name, with Cohen's d and its CI
5. `stat_plot_paired` is wrong here (independent groups) — use a box or raincloud plot via
   `stat_run_code` if the user wants a figure
6. Report: test name, t, df, p, Cohen's d with 95% CI, and the fact that Welch's was used **because**
   Levene's test was significant

## Choosing the reporting language

When relaying results, state:
- the **test by name** (including the variant — Welch's, not just "t-test"),
- the **assumption checks**, pass or fail,
- the **statistic with degrees of freedom**,
- the **exact p-value** (not just "p < .05"),
- the **effect size with its CI**,
- the **correction method** if more than one comparison was made.

## References

- **[references/statistics-guide.md](references/statistics-guide.md)** — full decision tree, assumption
  checks per design, and reporting templates.
- **[references/common-mistakes.md](references/common-mistakes.md)** — the errors that make a
  statistical result invalid, and how each one is prevented here.
