---
name: stat-mcp-guard
description: >
  Prevent the statistical mistakes that invalidate a result before they happen, when using the Pingouin
  MCP server. Use this whenever an analysis is about to be run or reported: it checks that the test
  matches the design (independent vs repeated, number of groups, DV type), that the assumptions the
  test requires have actually been tested, that effect sizes and confidence intervals are reported
  rather than p-values alone, that multiple comparisons are corrected, and that a non-significant
  result is not misinterpreted as evidence of no effect. Triggers: statistics check, assumption,
  normality, sphericity, Levene, p-hacking, multiple comparisons, effect size, power, 统计误用,
  假设检验前提, 多重比较, 效应量, 统计方法选错.
---

# Statistical MCP Guard

A checklist to run **before** a statistical result is presented to a researcher, and **before** it is
written into a manuscript. Each item below is a mistake that makes a result invalid or misleading —
not merely suboptimal.

Run through this list when any of the following is true: a test is about to be run, a result is about
to be reported, or the user asks "is this analysis right?"

## 1. Does the test match the design?

| Question | If yes | If no |
|---|---|---|
| Same participants in every condition? | paired / repeated-measures test | independent-samples test |
| More than two groups? | ANOVA family, then post-hoc | t-test is fine |
| More than one DV compared jointly? | MANOVA or separate ANOVAs with correction | single test |
| DV ordinal, or badly non-normal with small n? | rank-based test | parametric |
| A continuous covariate to control? | ANCOVA (and check slopes) | plain comparison |
| Claim is "no difference"? | **TOST or Bayes factor** | a non-significant test proves nothing |

**If any row is uncertain, run `stat_decision_assistant` — it asks rather than guesses.**

## 2. Have the assumptions actually been tested?

- [ ] **Normality** — `stat_normality_test`, plus a QQ plot. Applies to the *differences* in paired
      designs, not the raw scores.
- [ ] **Homogeneity of variance** — `stat_homogeneity_test`. If violated → Welch's t-test or Welch's
      ANOVA, and say so by name.
- [ ] **Sphericity** — `stat_sphericity_test`, for any repeated-measures factor with 3+ levels. If
      violated → report the Greenhouse-Geisser corrected p and the epsilon.
- [ ] **Independence of observations** — is there clustering (classes, families, sites) that the
      model ignores?
- [ ] **Outliers** — inspected, not silently removed. If removed, report the analysis both ways.

**Report every check, pass or fail.** Running them and mentioning only the failures is not a check —
it is a filter. `stat_assumption_report` produces the whole block in one call.

## 3. Is the result reported completely?

A complete report contains all of:

| Element | Why it is required |
|---|---|
| Test **by name and variant** (Welch's, GG-corrected, …) | Different variants give different numbers |
| **Exact p-value**, not "p < .05" | Readers need to judge, and meta-analysis needs it |
| **Degrees of freedom** | Reveals the design (e.g. paired vs independent) |
| **Effect size with 95% CI** | A p-value says whether, the CI says how much and how precisely |
| **n per group** | Power and representativeness |
| **Correction method, by name** | "Corrected for multiple comparisons" is not reproducible |
| **Direction of the difference** | Pingouin orders levels alphabetically; the sign depends on it |

## 4. Multiple comparisons

- Any set of more than one comparison needs a correction, **named**:
  - `holm` — family-wise error rate, small number of planned comparisons
  - `fdr_bh` — false discovery rate, many or exploratory comparisons
  - `bonf` — most conservative, only when any single false positive is unacceptable
  - `none` — only for a single pre-registered comparison
- Reporting only the significant pairs from an uncorrected set is p-hacking. `stat_pairwise_tests`
  warns when `padjust="none"` with more than one comparison — heed the warning.

## 4b. The sign of t and d

Pingouin orders group levels **alphabetically**. So with groups `control` and `treatment`, t is
computed as *control − treatment*, which may be the opposite of what the researcher expects
("treatment improved scores"). The t-test output states the direction — always relay it, or the sign
of t, d and the CI is uninterpretable.

## 5. The misinterpretations to refuse

| Statement | Problem | Say instead |
|---|---|---|
| "p > .05, so there is no difference" | Absence of evidence ≠ evidence of absence | "The data do not distinguish the groups; run `stat_equivalence_test` or `stat_bayes_ttest` to test for equivalence" |
| "Observed power is .35" | A monotone function of p; adds nothing | Report the effect size and CI |
| "The effect is large (p = .001)" | p reflects precision, not magnitude | Report the effect size |
| "We removed 8 outliers" | Can manufacture significance | Report both analyses |
| "Trend toward significance (p = .08)" | Either pre-specified or it is nothing | Report the estimate and CI |
| "Significant after correction for one of the three outcomes" | Selective outcome reporting | Pre-specify, or correct across all |
| "The correlation proves X causes Y" | Correlation is not causation | Describe the association |
| "ANCOVA controls for baseline" | Only valid if the groups' slopes are parallel and the covariate is unaffected by the treatment | Check the group × covariate interaction (`stat_ancova` does) |

## 6. Effect sizes — do not skip them

- Two groups → `stat_effect_size` (Cohen's d or Hedges' g with CI).
- 3+ groups → η²/ω² from ANOVA, plus pairwise effect sizes.
- Correlation → r with CI, and the sample size.
- Rank tests → rank-biserial correlation (`RBC`, given in the output).
- **A paired design's standardiser is the SD of the differences.** Cohen's d and d_z are not
  interchangeable — state which is reported.

## 7. Before the result goes into a manuscript

- [ ] The analysis plan was fixed before looking at the outcome (or this is labelled exploratory).
- [ ] Every test reported was pre-specified — no fishing across outcomes, subgroups or windows.
- [ ] Sample size and any exclusions are stated with reasons.
- [ ] The software and version are stated (`Pingouin 0.6.1`).
- [ ] The figure matches the test: paired data → line plot per participant, not a bar chart of means.

## Escalation

When a check fails, the correct behaviour is to **stop and tell the user**, not to run the test
anyway and footnote it. The failure may change the conclusion, which is exactly what needs to be known
before the analysis is written up.
