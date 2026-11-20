## What changed

<!-- One or two sentences. Link the issue if one exists. -->

## Why

<!-- The statistical rationale: which design or workflow this fixes. -->

## Checks

- [ ] `python -m pytest tests -q` passes
- [ ] `python experiments/exp1_decision_correctness.py` still passes (if decision rules changed)
- [ ] `TOOLS_REFERENCE.md` regenerated and README tool table updated (if tools changed)
- [ ] Every tool still returns equivalent Pingouin code
- [ ] No data files, figures, virtual environments or local configuration included
