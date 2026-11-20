# Pingouin-MCP

[![CI](https://github.com/weichuguang/pingouin-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/weichuguang/pingouin-mcp/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MCP](https://img.shields.io/badge/protocol-MCP-6f42c1.svg)](https://modelcontextprotocol.io)

A **Model Context Protocol** server that exposes statistical analysis through
[Pingouin](https://pingouin-stats.org/) to AI research agents (Claude Code,
OpenCode, and any MCP-capable assistant).

It is a companion to [MNE-MCP](https://github.com/Exekiel179/MNE-MCP): MNE-MCP covers
neurophysiological signal processing; Pingouin-MCP covers the statistical inference that
follows it. Together they cover the path from raw data to a reported result.

---

## Why this exists

An LLM will run whatever test you ask for. It will also, far more often than a trained
analyst, run the **wrong** one: a t-test on three groups, ANOVA on repeated measures without
checking sphericity, a parametric test on badly non-normal data, or an independent-samples
test on paired observations.

Three design decisions follow:

1. **The correct choice is made computable.** `stat_decision_assistant` maps a study design to
   a test, states why, lists what to check first and gives the alternatives. When the design
   is ambiguous it asks a question instead of guessing.
2. **Assumptions are never silently skipped.** `stat_assumption_report` runs the checklist for
   a design in one call and reports which assumptions failed. An assumption reported only
   when it fails is not a check.
3. **Every result carries its assumptions, effect size and reproducible code.** Tool output
   includes the equivalent Pingouin snippet, so any analysis can be reproduced outside the
   server.

---

## Requirements

| Component | Requirement | Notes |
| --- | --- | --- |
| Python | 3.12 or newer | Verified with 3.13 |
| Pingouin | 0.6.1 | Pinned to `>=0.6.1,<0.7` |
| pandas / NumPy / SciPy | pandas >= 2.0, NumPy >= 1.23, SciPy >= 1.9 | Installed as dependencies |
| OS | Windows, macOS, Linux | stdio transport; no platform-specific code |

---

## Install

```bash
git clone https://github.com/weichuguang/pingouin-mcp.git
cd pingouin-mcp
python -m venv .venv

# Windows
.venv\Scripts\python -m pip install -e .
# macOS / Linux
.venv/bin/python -m pip install -e .
```

Verify:

```bash
pingouin-mcp status     # JSON capability report; exits non-zero if Pingouin is missing
pingouin-mcp tools      # list the 43 registered tools
```

---

## Register with an MCP client

Use the absolute path to the virtual environment's Python.

```json
{
  "mcpServers": {
    "pingouin": {
      "command": "/absolute/path/to/pingouin-mcp/.venv/bin/python",
      "args": ["-m", "pingouin_mcp.cli", "serve"]
    }
  }
}
```

On Windows use `...\\.venv\\Scripts\\python.exe`.

VS Code (`.vscode/mcp.json`):

```json
{
  "servers": {
    "pingouin": {
      "type": "stdio",
      "command": "/absolute/path/to/pingouin-mcp/.venv/bin/python",
      "args": ["-m", "pingouin_mcp.cli", "serve"]
    }
  }
}
```

---

## Configuration

Optional file at `~/.pingouin-mcp/config.json`, or environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PINGOUIN_MCP_TIMEOUT` | Seconds before a tool call is abandoned | 300 |
| `PINGOUIN_MCP_RESULTS_DIR` | Where figures are written | `~/.pingouin-mcp/results` |
| `PINGOUIN_MCP_DATA_DIR` | Directory searched by data tools | current directory |
| `PINGOUIN_MCP_CONFIG` | Path to a config file | `~/.pingouin-mcp/config.json` |

`pingouin-mcp show-config` prints the resolved values.

---

## Tools (43)

Organised by research task rather than by Pingouin function name. The full list with
when-to-use guidance is in [`TOOLS_REFERENCE.md`](TOOLS_REFERENCE.md).

| Group | Tools |
| --- | --- |
| Data management | `stat_load_data`, `stat_make_dataframe`, `stat_describe_data`, `stat_data_quality_check` |
| Decision support | `stat_decision_assistant` |
| Assumptions | `stat_assumption_report`, `stat_normality_test`, `stat_homogeneity_test`, `stat_sphericity_test` |
| Two groups | `stat_ttest`, `stat_nonparametric_2`, `stat_bayes_ttest` |
| Three or more groups | `stat_anova`, `stat_welch_anova`, `stat_rm_anova`, `stat_mixed_anova`, `stat_ancova`, `stat_nonparametric_k` |
| Post-hoc | `stat_pairwise_tests`, `stat_pairwise_tukey`, `stat_multicomp_correction` |
| Association | `stat_correlation`, `stat_partial_correlation`, `stat_rm_correlation`, `stat_correlation_matrix`, `stat_distance_correlation` |
| Regression and mediation | `stat_linear_regression`, `stat_logistic_regression`, `stat_mediation` |
| Effect size, power, reliability | `stat_effect_size`, `stat_power_analysis`, `stat_cronbach_alpha`, `stat_intraclass_correlation`, `stat_equivalence_test` |
| Categorical | `stat_chi2_independence`, `stat_chi2_mcnemar` |
| Visualisation | `stat_plot_paired`, `stat_plot_blandaltman`, `stat_qqplot`, `stat_plot_rm_corr` |
| Session | `stat_session_info`, `stat_reset_session`, `stat_run_code` |

---

## A worked exchange

```
User:  I have two groups, 20 people each, and I want to compare their
       anxiety scores.

Agent: [calls stat_decision_assistant(dv_type="continuous", n_groups=2,
                                      repeated=False)]
       -> recommends an independent-samples t-test and says to check
          normality and homogeneity of variance first.

Agent: [calls stat_assumption_report(df, dv="anxiety", between="group")]
       -> normality met, but Levene's test is significant (p = .01)

Agent: [calls stat_ttest(df, dv="anxiety", group="group", correction=True)]
       -> Welch's t-test, reports t, df, p, Cohen's d and the 95% CI, and
          notes that Welch's was used because variances differed.
```

The third call differs from the naive one because the second call was made. That is the
whole design.

---

## Reproducibility contract

Every tool returns the equivalent Pingouin code alongside the result. An analysis performed
through this server can therefore be reproduced, audited or moved into a script without the
server. An agent-driven analysis should not be a black box.

---

## Verification

```bash
python -m pytest tests -q          # 29 tests: correctness, tool binding, end-to-end calls
python experiments/exp1_decision_correctness.py
```

Current results on the reference environment (Python 3.13, Pingouin 0.6.1, pandas 3.x):

- `pytest`: **29 passed**
- Decision-assistant evaluation: **21/21** valid recommendations, **21/21** runnable in
  Pingouin, **3/3** abstentions on under-specified designs, deterministic across repeated
  runs. See [`experiments/`](experiments/) and `experiments/results_exp1.json`.

CI runs the test suite on Ubuntu and Windows with Python 3.12 and 3.13.

---

## Scope and limitations

- **Pingouin only.** Tools wrap functions that exist in Pingouin. Capabilities absent from
  Pingouin (for example MANOVA as a standalone procedure, moderation analysis) are not
  approximated.
- **No arbitrary code path is required, but one exists.** `stat_run_code` executes Python in
  the server's session for workflows not covered by a dedicated tool. It is intentionally
  visible and should be treated as a privileged escape hatch; the other 42 tools are
  parameter-validated.
- **`stat_run_code` runs in-process.** It is not a sandbox. Do not expose the server to
  untrusted clients.
- **File paths are not sandboxed.** `stat_load_data` reads the path it is given; use the
  server under an account whose file access matches the data you intend to analyse.
- **Assumption checks are reported, not enforced.** The server tells you what failed; it does
  not silently switch methods unless the tool's parameters ask it to.
- **Verified versions.** Pingouin 0.6.1, pandas 3.x and Python 3.13; other versions may work
  but are not part of CI.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `.github/` | CI workflow and issue / pull-request templates |
| `src/pingouin_mcp/` | Server, 43 tools, decision rules, session kernel, figures and summaries |
| `tests/` | Correctness, MCP binding and end-to-end tests |
| `skills/` | Two Claude skills: statistical analyst and MCP guard |
| `scripts/` | `gen_tools_reference.py` (regenerates `TOOLS_REFERENCE.md`) |
| `experiments/` | Self-contained evaluation of the decision assistant |

---

## Contributing

Issues and pull requests are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) and run the
tests before submitting. Security reports should follow [`SECURITY.md`](SECURITY.md).

## License

MIT. See [`LICENSE`](LICENSE).
