# Contributing to Pingouin-MCP

Thanks for contributing. This project is about statistical correctness: a tool that refuses
an inappropriate test is more valuable than one that returns a number for every request.

## Development setup

```bash
git clone https://github.com/weichuguang/pingouin-mcp.git
cd pingouin-mcp
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"     # Windows: .venv\Scripts\python
```

## Before opening a pull request

1. Run the full suite; it must pass:
   ```bash
   python -m pytest tests -q
   ```
2. If you add or change a tool, regenerate the reference:
   ```bash
   python scripts/gen_tools_reference.py
   ```
   and update the tool table in `README.md`.
3. If you change the decision assistant, run the evaluation:
   ```bash
   python experiments/exp1_decision_correctness.py
   ```
   A change that lowers validity, runnability or abstention is a regression.
4. Keep the reproducibility contract: every tool must return equivalent Pingouin code for
   the analysis it performed.
5. Do not weaken assumption reporting. A check that only appears when it fails is not a check.

## Design rules

- Wrap only functions that exist in Pingouin; do not approximate missing procedures.
- Validate parameters before computation and raise a clear error naming the parameter.
- Prefer returning a structured result plus a readable summary over free text.
- `stat_run_code` is a privileged escape hatch. Do not add more code-execution entry points.

## Commit messages

Use a short imperative subject and explain the statistical rationale in the body. Reference
issues where relevant. Do not commit data files, figures, virtual environments or local
configuration.

## Reporting bugs

Open an issue with the tool name, arguments, returned result, and the versions of Python,
Pingouin, pandas and NumPy. A minimal reproducible dataframe is the most useful attachment.

Security issues should follow [SECURITY.md](SECURITY.md).
