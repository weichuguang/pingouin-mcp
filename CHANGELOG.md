# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-20

### Added

- MCP server exposing 43 statistical tools through Pingouin.
- `stat_decision_assistant`: maps a study design to a test, lists assumptions to check and
  asks when the design is under-specified.
- `stat_assumption_report`: one call returns the full assumption checklist for a design.
- Effect sizes, confidence intervals and equivalent Pingouin code in every result.
- Session kernel with `stat_run_code` for workflows not covered by a dedicated tool.
- Two Claude skills (`stat-analyst`, `stat-mcp-guard`) and a generated tool reference.
- Test suite covering statistical correctness, MCP tool binding and end-to-end calls.
- Self-contained evaluation of the decision assistant (`experiments/`).
