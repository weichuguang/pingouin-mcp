# Evaluation

## Decision-assistant correctness (`exp1_decision_correctness.py`)

This script evaluates `stat_decision_assistant` without calling an LLM. It enumerates the
design space, asks the assistant for a recommendation, and grades four properties:

| Property | Question |
| --- | --- |
| Validity | Is the recommended test the one a methodologist would choose? |
| Runnability | Does the recommendation name a procedure that exists in Pingouin? |
| Abstention | When the design is under-specified, does it ask instead of guessing? |
| Determinism | Does the same design produce the same recommendation every time? |

Ground truth is a separately written table, not the assistant's own rules.

### Run

```bash
python experiments/exp1_decision_correctness.py
```

The script writes `results_exp1.json` next to itself and exits non-zero if any validity,
runnability or abstention check fails.

### Current result

Reference environment: Python 3.13, Pingouin 0.6.1, pandas 3.x.

| Property | Result |
| --- | --- |
| Validity | 21/21 |
| Runnability | 21/21 |
| Abstention | 3/3 |
| Determinism | yes |

The numbers are properties of the enumerated design space in the script, not a claim about
every possible study. Re-run the script after changing `src/pingouin_mcp/decision.py`.

## What is not included

An agent-versus-assistant comparison requires a fixed set of model replies, which depends on
the model, prompt and date. Such a comparison is not shipped here because it cannot be
reproduced from the repository alone.
