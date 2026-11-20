"""
Statistical decision assistant.

The problem this module solves
------------------------------
An LLM will happily run a test you ask for. It will also — much more often than
a trained analyst — run the *wrong* test: a t-test on three groups, an ANOVA on
repeated measures without checking sphericity, a parametric test on data that
violates its assumptions, or an independent-samples test on paired data.

The fix is not to make the model smarter. It is to make the *correct choice*
computable from the design, and to state it together with the assumptions that
must hold for it to be valid.

Design of this module
---------------------
* Every recommendation carries: the recommended test, **why**, what to check
  first, and what the alternatives are (parametric ↔ non-parametric).
* Nothing here runs a test. It only *advises*. Running is a separate,
  explicit step — the agent (or the researcher) must decide to proceed.
* When the design is ambiguous, it returns a question rather than a guess.

This is deliberately "advisory, not authoritative": the returned block always
ends with a note that the final choice is the researcher's.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Design:
    """A description of the study design, as far as it is known."""

    dv_type: str = "continuous"          # continuous | ordinal | categorical | count
    n_groups: int | None = None          # number of levels of the factor
    repeated: bool | None = None         # True = within-subject / paired
    n_dv: int = 1                        # number of dependent variables
    has_covariate: bool = False          # ANCOVA situation
    normality_ok: bool | None = None     # None = not yet checked
    homogeneity_ok: bool | None = None
    sphericity_ok: bool | None = None
    n_per_group: list[int] = field(default_factory=list)
    n_subjects: int | None = None
    predicting: bool = False             # regression rather than group comparison
    # What the study is trying to establish. A difference and an equivalence are
    # different questions with different tests; a non-significant difference
    # test is not evidence of equivalence.
    goal: str = "difference"             # difference | equivalence | associate | predict
    extra: dict = field(default_factory=dict)


@dataclass
class Recommendation:
    test: str
    why: str
    why_zh: str
    check_first: list[str]
    alternatives: list[tuple[str, str]]   # (when, test)
    caveats: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    question: str | None = None
    # Whether this server can actually run the recommendation. Some correct
    # choices (Poisson regression, ordinal regression, mixed-effects models)
    # live outside Pingouin — saying so is more useful than silently implying
    # the tool can do it.
    runnable: bool = True
    needs_instead: str | None = None

    def to_dict(self) -> dict:
        return {
            "recommended_test": self.test,
            "reason": self.why,
            "reason_zh": self.why_zh,
            "check_first": self.check_first,
            "alternatives": [{"when": w, "test": t} for w, t in self.alternatives],
            "caveats": self.caveats,
            "blocked_reason": self.blocked_reason,
            "question": self.question,
            "runnable_here": self.runnable,
            "needs_instead": self.needs_instead,
        }


# ── Core decision logic ───────────────────────────────────────────────────────

def recommend(design: Design) -> Recommendation:
    """
    Map a study design to a recommended test, its rationale and its assumptions.

    Returns a :class:`Recommendation`. If essential design information is
    missing, ``question`` is populated and ``test`` is ``"insufficient information"``
    — the assistant must ask rather than guess.
    """

    # ---- Guard: is the design specified enough to advise? -------------------
    if design.dv_type == "continuous" and not design.predicting:
        if design.n_groups is None:
            return Recommendation(
                test="insufficient information",
                why="The number of groups (factor levels) determines which test family applies.",
                why_zh="需要知道分组数量（因素水平数）才能确定检验族。",
                check_first=[],
                alternatives=[],
                question=(
                    "How many groups / conditions are being compared? "
                    "(1 = one sample vs a reference value, 2 = two groups, "
                    "3+ = three or more groups)"
                ),
            )
        if design.n_groups >= 2 and design.repeated is None:
            return Recommendation(
                test="insufficient information",
                why=(
                    "The same participants measured twice requires a paired test; "
                    "different participants requires an independent-samples test. "
                    "These are not interchangeable."
                ),
                why_zh=(
                    "同一批被试重复测量需用配对检验；不同被试需用独立样本检验。"
                    "两者不可互换。"
                ),
                check_first=[],
                alternatives=[],
                question=(
                    "Are the groups independent (different participants in each group) "
                    "or repeated/paired (the same participants measured under each condition)?"
                ),
            )

    # ---- Equivalence claims ------------------------------------------------
    # Guard first: an equivalence claim needs a different test family entirely,
    # regardless of how many groups there are.
    if design.goal == "equivalence" and design.dv_type == "continuous":
        return Recommendation(
            test="equivalence test (TOST)",
            why=(
                "The study claims the groups are equivalent, not that they differ. "
                "A non-significant difference test cannot establish equivalence — "
                "failing to reject the null is not accepting it."
            ),
            why_zh=(
                "研究要主张的是两组等价，而非存在差异。差异检验不显著无法证明等价——"
                "不能拒绝原假设不等于接受原假设。"
            ),
            check_first=[
                "an equivalence bound justified on substantive grounds, "
                "ideally specified before data collection",
                "normality of the difference (for the paired form)",
            ],
            alternatives=[
                ("no defensible bound can be set", "Bayesian t-test: report BF₀₁ as "
                 "evidence for the null instead"),
                ("the bound is chosen after seeing the data", "the test is uninformative — "
                 "say so rather than running it"),
            ],
            caveats=[
                "The bound is the substance of the test. ±0.5 SD, ±5 points on the "
                "outcome scale, or a clinically negligible difference are defensible; "
                "'whatever makes it significant' is not.",
            ],
        )

    # ---- Non-continuous outcomes -------------------------------------------
    if design.dv_type == "categorical":
        if design.repeated:
            return Recommendation(
                test="McNemar's test",
                why=(
                    "The same units contribute a categorical value under two conditions "
                    "(paired categorical data), so the observations are not independent "
                    "and the chi-square test of independence does not apply."
                ),
                why_zh=(
                    "同一批对象在两种条件下各贡献一个分类取值（配对分类数据），"
                    "观测不独立，不能使用独立性卡方检验。"
                ),
                check_first=[
                    "the two discordant cells are the only ones that contribute to the test",
                    "with few discordant pairs, use the exact binomial form",
                ],
                alternatives=[
                    ("independent groups instead", "chi-square test of independence"),
                    ("three or more paired categorical measurements", "Cochran's Q"),
                ],
            )
        return Recommendation(
            test="chi-square test of independence",
            why=(
                "Both variables are categorical, so the question is association between "
                "frequencies rather than a difference of means."
            ),
            why_zh="两变量均为分类变量，问题为频数间的关联而非均值差异。",
            check_first=[
                "expected frequency ≥ 5 in at least 80% of cells",
                "all expected frequencies > 1",
            ],
            alternatives=[
                ("any expected count < 5", "Fisher's exact test"),
                ("paired/repeated categorical design", "McNemar's test"),
            ],
            caveats=[
                "With large tables and small cells, consider exact or Monte-Carlo methods.",
            ],
        )

    if design.dv_type == "count":
        return Recommendation(
            test="Poisson / negative-binomial regression",
            why=(
                "Count outcomes are bounded below at zero and their variance scales with "
                "the mean, which violates the constant-variance assumption of linear models."
            ),
            why_zh="计数变量以 0 为下界且方差随均值变化，违反线性模型的方差齐性假设。",
            check_first=[
                "overdispersion (variance >> mean) — if present, use negative binomial",
                "excess zeros — consider zero-inflated models",
            ],
            alternatives=[
                ("no overdispersion", "Poisson regression"),
                ("overdispersion present", "negative-binomial regression"),
            ],
            runnable=False,
            needs_instead=(
                "Pingouin has no count-regression model. Use statsmodels "
                "(`sm.GLM(..., family=sm.families.Poisson())` or "
                "`NegativeBinomial`) — it is installed in this environment."
            ),
        )

    # ---- Prediction rather than group comparison ---------------------------
    if design.predicting:
        return Recommendation(
            test="linear regression",
            why=(
                "The predictor is continuous or the goal is prediction rather than a "
                "difference between groups."
            ),
            why_zh="预测变量为连续变量，或研究目标是预测而非组间差异。",
            check_first=[
                "linearity of the predictor–outcome relationship",
                "residual normality and homoscedasticity",
                "multicollinearity if several predictors (VIF)",
                "influential cases (Cook's distance)",
            ],
            alternatives=[
                ("outcome is binary", "logistic regression"),
                ("outcome is ordinal", "ordinal regression"),
                ("nonlinear relationship", "polynomial or spline regression"),
                ("clustered / repeated data", "mixed-effects model"),
            ],
        )

    # ---- Ordinal outcome ----------------------------------------------------
    if design.dv_type == "ordinal":
        return Recommendation(
            test="non-parametric test (rank-based)",
            why=(
                "Ordinal outcomes carry order but not equal intervals, so means are not "
                "meaningful and rank-based tests are appropriate."
            ),
            why_zh="有序变量只有顺序信息而无等距性，均值无意义，应使用秩检验。",
            check_first=["tie structure (many ties weaken rank tests)"],
            alternatives=[("two independent groups", "Mann-Whitney U"),
                          ("two paired groups", "Wilcoxon signed-rank"),
                          ("3+ independent groups", "Kruskal-Wallis"),
                          ("3+ repeated conditions", "Friedman test")],
            caveats=[
                "A rank-based test is the practical default. A cumulative-link "
                "ordinal model uses the ordering information more fully but is not "
                "available in Pingouin — use statsmodels if that matters."
            ],
        )

    # ---- 1 group ------------------------------------------------------------
    if design.n_groups == 1:
        if design.normality_ok is False:
            return Recommendation(
                test="Wilcoxon signed-rank test",
                why="One sample compared against a reference value; normality is not met.",
                why_zh="单样本与参考值比较；正态性不满足。",
                check_first=["distribution symmetry (required for signed-rank)"],
                alternatives=[("normality holds", "one-sample t-test")],
            )
        return Recommendation(
            test="one-sample t-test",
            why="One sample compared against a known reference value.",
            why_zh="单样本与已知参考值比较。",
            check_first=["normality of the differences from the reference value"],
            alternatives=[("normality violated", "Wilcoxon signed-rank test")],
        )

    # ---- 2 groups -----------------------------------------------------------
    if design.n_groups == 2:
        if design.repeated:
            if design.normality_ok is False:
                return Recommendation(
                    test="Wilcoxon signed-rank test",
                    why="Paired design with non-normal difference scores.",
                    why_zh="配对设计，差值不服从正态分布。",
                    check_first=["symmetry of the difference distribution"],
                    alternatives=[("normality holds", "paired t-test")],
                )
            return Recommendation(
                test="paired t-test",
                why="Same participants measured under two conditions.",
                why_zh="同一批被试在两种条件下测量。",
                check_first=[
                    "normality of the *differences* (not of each condition separately)",
                    "absence of extreme outliers in the differences",
                ],
                alternatives=[
                    ("differences non-normal", "Wilcoxon signed-rank test"),
                    ("binary paired outcome", "McNemar's test"),
                ],
            )
        if design.normality_ok is False:
            return Recommendation(
                test="Mann-Whitney U test",
                why="Two independent groups with non-normal outcome.",
                why_zh="两个独立组，因变量不服从正态分布。",
                check_first=["similar distribution shapes between groups"],
                alternatives=[
                    ("normality holds", "independent-samples t-test"),
                    ("variances unequal and normality holds", "Welch's t-test"),
                ],
            )
        if design.homogeneity_ok is False:
            return Recommendation(
                test="Welch's t-test",
                why=(
                    "Two independent groups, but the equal-variance assumption is "
                    "violated. Welch's t-test does not assume equal variances and "
                    "performs well even when they happen to be equal."
                ),
                why_zh=(
                    "两个独立组，但方差齐性假设不满足。Welch t 检验不要求方差相等，"
                    "即使方差实际相等也表现良好。"
                ),
                check_first=[
                    "normality within each group (Welch still assumes it)",
                    "the direction of the variance difference and the group-size ratio",
                ],
                alternatives=[
                    ("variances equal", "the classic Student t-test"),
                    ("normality also violated", "Mann-Whitney U test"),
                ],
                caveats=[
                    "Report the test by name. Welch's and Student's t give different "
                    "t, different df, and different p — they are not interchangeable.",
                ],
            )
        return Recommendation(
            test="independent-samples t-test",
            why="Two independent groups compared on a continuous outcome.",
            why_zh="两个独立组在连续因变量上比较。",
            check_first=[
                "normality within each group",
                "homogeneity of variance (Levene's test)",
                "equal group sizes (unequal n makes the test less robust to variance differences)",
            ],
            alternatives=[
                ("variances unequal", "Welch's t-test (does not assume equal variances)"),
                ("normality violated", "Mann-Whitney U test"),
            ],
            caveats=[
                "If group sizes are very unequal and variances differ, report Welch's t-test "
                "by default — it performs well even when variances are equal.",
            ],
        )

    # ---- 3+ groups ----------------------------------------------------------
    if design.repeated:
        if design.normality_ok is False:
            return Recommendation(
                test="Friedman test",
                why=(
                    "Three or more repeated conditions with normality violated. The "
                    "rank-based alternative to RM-ANOVA is Friedman's test."
                ),
                why_zh="三个及以上重复条件且正态性不满足，应使用 Friedman 秩检验。",
                check_first=[
                    "the outcome is at least ordinal (ranks must be meaningful)",
                    "complete data per participant, or handle missingness explicitly",
                ],
                alternatives=[
                    ("normality holds", "repeated-measures ANOVA"),
                    ("sphericity also violated but data normal", "RM-ANOVA with Greenhouse-Geisser correction"),
                    ("missing data", "a mixed-effects model handles it better than either"),
                ],
                caveats=[
                    "Friedman's test is sensitive to how ties are broken with few "
                    "levels; report Kendall's W as the effect size.",
                ],
            )
        if design.sphericity_ok is False:
            return Recommendation(
                test="repeated-measures ANOVA with Greenhouse-Geisser correction",
                why=(
                    "Three or more repeated measurements; the sphericity assumption is "
                    "violated, so the uncorrected F test has an inflated Type I error rate."
                ),
                why_zh=(
                    "三次以上重复测量；球形性假设不满足，未校正的 F 检验 I 类错误率偏高。"
                ),
                check_first=["the epsilon value (report it and the corrected p)"],
                alternatives=[
                    ("sphericity holds", "repeated-measures ANOVA (uncorrected)"),
                    ("only two conditions", "paired t-test"),
                    ("ordinal outcome", "Friedman test"),
                ],
            )
        return Recommendation(
            test="repeated-measures ANOVA",
            why="Three or more conditions measured in the same participants.",
            why_zh="三个及以上条件在同一批被试中测量。",
            check_first=[
                "sphericity (Mauchly's test — report epsilon)",
                "normality of the residuals",
                "absence of extreme outliers",
            ],
            alternatives=[
                ("sphericity violated", "RM-ANOVA with Greenhouse-Geisser correction"),
                ("ordinal or badly non-normal", "Friedman test"),
                ("missing data / unbalanced", "linear mixed-effects model"),
            ],
            caveats=[
                "RM-ANOVA with listwise deletion can bias results when data are missing; "
                "a mixed-effects model uses all available data.",
            ],
        )

    # independent groups, 3+
    if design.normality_ok is False and not design.has_covariate and design.n_dv == 1:
        return Recommendation(
            test="Kruskal-Wallis test",
            why=(
                "Three or more independent groups with normality violated. The "
                "rank-based alternative to one-way ANOVA is Kruskal-Wallis."
            ),
            why_zh="三个及以上独立组且正态性不满足，应使用 Kruskal-Wallis 秩检验。",
            check_first=[
                "the outcome is at least ordinal (ranks must be meaningful)",
                "similar distribution shapes across groups, if the claim is about medians",
            ],
            alternatives=[
                ("normality holds", "one-way ANOVA"),
                ("variances unequal but data normal", "Welch's ANOVA"),
                ("a covariate must be controlled", "rank-based ANCOVA or a robust regression"),
            ],
            caveats=[
                "A significant Kruskal-Wallis says some groups differ; follow with "
                "`stat_pairwise_tests(parametric=False)` and a correction. "
                "Report epsilon-squared or the pairwise rank-biserial correlation.",
            ],
        )
    rec = Recommendation(
        test="one-way ANOVA",
        why="Three or more independent groups compared on a continuous outcome.",
        why_zh="三个及以上独立组在连续因变量上比较。",
        check_first=[
            "normality within each group",
            "homogeneity of variance (Levene's test)",
            "independence of observations",
        ],
        alternatives=[
            ("variances unequal", "Welch's ANOVA (does not assume equal variances)"),
            ("normality badly violated", "Kruskal-Wallis test"),
            ("a covariate must be controlled", "ANCOVA"),
            ("two or more dependent variables", "MANOVA"),
        ],
        caveats=[
            "A significant omnibus F says only that *some* groups differ. "
            "Report a post-hoc test with a multiple-comparison correction.",
            "With unequal group sizes and unequal variances, prefer Welch's ANOVA.",
        ],
    )
    if design.has_covariate:
        rec = Recommendation(
            test="ANCOVA",
            why="Group comparison while statistically controlling a continuous covariate.",
            why_zh="在统计上控制一个连续协变量后做组间比较。",
            check_first=[
                "linear relationship between covariate and outcome within each group",
                "homogeneity of regression slopes (group × covariate interaction)",
                "covariate measured reliably and before the manipulation",
            ],
            alternatives=[
                ("slopes differ between groups", "report the interaction; ANCOVA is not appropriate"),
                ("covariate is affected by the manipulation", "do NOT use ANCOVA"),
            ],
            caveats=[
                "ANCOVA on a covariate that the treatment changed removes part of the effect "
                "you are trying to measure.",
            ],
        )
    if design.n_dv > 1:
        rec = Recommendation(
            test="MANOVA",
            why="Several correlated dependent variables compared across groups.",
            why_zh="多个相关因变量在组间比较。",
            check_first=[
                "multivariate normality",
                "homogeneity of covariance matrices (Box's M)",
                "correlation among the DVs (if uncorrelated, separate ANOVAs are more powerful)",
            ],
            alternatives=[
                ("DVs uncorrelated", "separate ANOVAs with appropriate correction"),
                ("more DVs than the design can support", "reduce dimensionality or use a composite"),
            ],
            caveats=[
                "MANOVA with many DVs and small n has very low power. "
                "Check that n per group comfortably exceeds the number of DVs.",
            ],
        )
    return rec


# ── Assumption checklist ─────────────────────────────────────────────────────

def assumption_checklist(design: Design) -> list[tuple[str, str]]:
    """
    Return ``(assumption, how to check it)`` pairs for a design.

    This is what the agent should run *before* the main test — and report,
    whether or not it passes.
    """
    checks: list[tuple[str, str]] = []

    if design.dv_type == "continuous":
        if design.n_groups and design.n_groups >= 2:
            checks.append((
                "Normality",
                "Shapiro-Wilk per group for n < 50; "
                "skewness/kurtosis and QQ plot for larger n",
            ))
        if design.n_groups == 1:
            checks.append((
                "Normality of the sample",
                "Shapiro-Wilk test plus a QQ plot",
            ))
        if design.n_groups and design.n_groups >= 2 and not design.repeated:
            checks.append((
                "Homogeneity of variance",
                "Levene's test (report it even when non-significant)",
            ))
        if design.repeated and (design.n_groups or 0) >= 3:
            checks.append((
                "Sphericity",
                "Mauchly's test; if violated report Greenhouse-Geisser epsilon",
            ))
        if design.repeated:
            checks.append((
                "Normality of differences",
                "For paired designs, normality applies to the *differences*, "
                "not the raw scores",
            ))
        checks.append((
            "Outliers",
            "Inspect with a boxplot / robust z-scores; "
            "report analyses with and without them",
        ))

    if design.n_dv > 1:
        checks.append((
            "Multivariate normality",
            "Mardia's test, or inspect per-variable QQ plots",
        ))
        checks.append((
            "Homogeneity of covariance matrices",
            "Box's M test",
        ))

    if design.predicting:
        checks.append(("Linearity", "Partial regression plots"))
        checks.append(("Homoscedasticity of residuals", "Residuals-vs-fitted plot"))
        checks.append(("Multicollinearity", "VIF for each predictor"))
        checks.append(("Influential cases", "Cook's distance"))

    return checks


# ── Rendered output ──────────────────────────────────────────────────────────

def render(rec: Recommendation, design: Design) -> str:
    """Render a recommendation as Markdown for the agent to relay."""
    from pingouin_mcp.summaries import fmt  # noqa: F401  (kept for symmetry)

    lines = ["### Statistical decision assistant", ""]

    if rec.question:
        lines += [
            f"**Cannot advise yet — the design is not specified enough.**",
            "",
            rec.question,
            "",
            "I am asking rather than guessing because the choice between these test "
            "families is not a matter of preference: using the wrong one invalidates "
            "the p-value.",
        ]
        return "\n".join(lines)

    lines += [f"**Recommended test:** `{rec.test}`", "", f"**Why:** {rec.why}"]
    lines += ["", f"**理由：** {rec.why_zh}"]

    if not rec.runnable:
        lines += [
            "",
            "**This recommendation is not runnable by this server.** "
            + (rec.needs_instead or ""),
        ]

    if rec.check_first:
        lines += ["", "**Check these first:**", ""]
        lines += [f"- {c}" for c in rec.check_first]

    if rec.alternatives:
        lines += ["", "**Alternatives**", "", "| If | Then use |", "|---|---|"]
        lines += [f"| {w} | {t} |" for w, t in rec.alternatives]

    if rec.caveats:
        lines += ["", "**Caveats**", ""]
        lines += [f"- {c}" for c in rec.caveats]

    checks = assumption_checklist(design)
    if checks:
        lines += ["", "**Assumption checklist**", "", "| Assumption | How to check |", "|---|---|"]
        lines += [f"| {a} | {h} |" for a, h in checks]

    lines += [
        "",
        "---",
        "_This is an advisory recommendation based on the design you described. "
        "The choice of statistical method is the researcher's responsibility, "
        "and should follow from the pre-registered analysis plan where one exists._",
    ]
    return "\n".join(lines)
