"""
Persistent analysis session for Pingouin MCP.

Why a session at all? Statistical work with an agent is *iterative*: load a
dataset, check its quality, test assumptions, choose a test, run it, look at
effect sizes, re-run with a correction, plot it. Re-reading the file on every
call would be wasteful and — more importantly — would lose the analyst's
working state (which variables were recoded, which rows were excluded).

So the server keeps one long-lived :class:`Session` holding a namespace of
named objects: loaded datasets (DataFrames), derived variables, results, and
figures. Tools operate on that namespace by name.

Difference from MNE-MCP: MNE objects are large and mutated in place; statistical
datasets are small and usually *not* mutated. The session therefore stores
results as first-class named objects, which lets an agent say "compare the
effect size from `anova_result` with the one in `ttest_result`".
"""

from __future__ import annotations

import ast
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from pingouin_mcp import figures
from pingouin_mcp.config import get_results_dir

# Names always present in the namespace; never listed as user data.
_PRELOADED = {"pg", "np", "pd", "plt", "sns", "session", "__builtins__", "SESSION"}


class Session:
    """A persistent statistical analysis namespace."""

    def __init__(self) -> None:
        self.namespace: dict = {}
        self.history: list[str] = []
        self._init_namespace()

    def _init_namespace(self) -> None:
        import matplotlib

        matplotlib.use("Agg")  # headless: figures are saved, never shown
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import pingouin as pg

        try:
            import seaborn as sns
        except Exception:  # pragma: no cover - seaborn is a hard dependency
            sns = None

        # Keep pandas output compact when an agent prints a frame.
        try:
            pd.set_option("display.max_columns", 50)
            pd.set_option("display.width", 160)
            pd.set_option("display.precision", 4)
        except Exception:
            pass

        self.namespace = {
            "pg": pg,
            "np": np,
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "session": self,
        }

    # ── object registry ────────────────────────────────────────────────────

    def set(self, name: str, obj) -> None:
        if not name.isidentifier():
            raise ValueError(
                f"'{name}' is not a valid object name. Use a Python identifier "
                "(letters, digits, underscore; cannot start with a digit)."
            )
        self.namespace[name] = obj

    def get(self, name: str):
        if name not in self.namespace:
            raise KeyError(name)
        return self.namespace[name]

    def has(self, name: str) -> bool:
        return name in self.namespace and name not in _PRELOADED

    def data_names(self) -> list[str]:
        """Names of user objects currently held (excludes preloaded modules)."""
        import types

        skip_types = (
            types.ModuleType,
            types.FunctionType,
            types.BuiltinFunctionType,
            type,
        )
        out = []
        for key, val in self.namespace.items():
            if key in _PRELOADED or key.startswith("_"):
                continue
            if isinstance(val, skip_types):
                continue
            out.append(key)
        return out

    def require(self, name: str):
        """Fetch an object, raising a message aimed at an agent rather than a user."""
        try:
            return self.get(name)
        except KeyError:
            available = ", ".join(f"`{n}`" for n in self.data_names()) or "(none)"
            raise KeyError(
                f"No object named '{name}' in the session. "
                f"Available objects: {available}. "
                "Loaded datasets appear here after a load tool; results appear "
                "after an analysis tool runs."
            ) from None

    # ── summaries ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        names = self.data_names()
        if not names:
            return (
                "_Session is empty — no datasets loaded yet._\n\n"
                "Load data with `stat_load_data`, or build one with "
                "`stat_make_dataframe`."
            )
        lines = ["| Name | Kind | Shape / Detail |", "|---|---|---|"]
        for name in names:
            obj = self.namespace[name]
            kind, detail = _describe_briefly(obj)
            lines.append(f"| `{name}` | {kind} | {detail} |")
        return "\n".join(lines)

    def reset(self) -> None:
        figures.close_all()
        self.history.clear()
        self._init_namespace()

    # ── code execution ───────────────────────────────────────────────────────

    def run_code(
        self,
        code: str,
        *,
        capture_figures: bool = True,
        fig_prefix: str = "code",
    ) -> dict:
        """
        Execute ``code`` in the persistent namespace, notebook-style.

        The value of a trailing expression is returned in ``result_repr``.
        stdout/stderr are captured, and any matplotlib figures produced are
        saved as PNG with their paths returned in ``figures``.
        """
        results_dir = get_results_dir()
        before = figures.open_figure_numbers() if capture_figures else set()

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result_value = None
        error = None
        tb = None

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            return {
                "stdout": "",
                "stderr": "",
                "result_repr": None,
                "figures": [],
                "error": f"SyntaxError: {e}",
                "traceback": None,
            }

        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                if tree.body:
                    exec(compile(tree, "<pingouin-mcp>", "exec"),
                         self.namespace, self.namespace)
                if last_expr is not None:
                    result_value = eval(
                        compile(ast.Expression(last_expr.value), "<pingouin-mcp>", "eval"),
                        self.namespace, self.namespace,
                    )
        except Exception as e:  # noqa: BLE001 - surface analysis errors verbatim
            error = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()

        saved = (
            figures.capture_new_figures(before, Path(results_dir), prefix=fig_prefix)
            if capture_figures
            else []
        )

        self.history.append(code)

        result_repr = None
        if result_value is not None:
            text = repr(result_value)
            # DataFrames get a readable table rather than a repr blob.
            try:
                import pandas as pd

                if isinstance(result_value, pd.DataFrame):
                    text = result_value.to_string(max_rows=60)
            except Exception:
                pass
            result_repr = (text[:4000] + " …") if len(text) > 4000 else text

        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result_repr": result_repr,
            "figures": saved,
            "error": error,
            "traceback": tb,
        }


def _describe_briefly(obj) -> tuple[str, str]:
    """Return (kind, one-line detail) for the session listing."""
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return "DataFrame", f"{obj.shape[0]} rows × {obj.shape[1]} cols"
        if isinstance(obj, pd.Series):
            return "Series", f"{len(obj)} values, dtype={obj.dtype}"
        if isinstance(obj, pd.Index):
            return "Index", f"{len(obj)} values"
    except Exception:
        pass
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return "ndarray", f"shape={obj.shape}, dtype={obj.dtype}"
        if isinstance(obj, (np.floating, np.integer)):
            return type(obj).__name__, str(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return "dict", f"{len(obj)} keys"
    if isinstance(obj, (list, tuple)):
        return type(obj).__name__, f"{len(obj)} items"
    if isinstance(obj, (int, float, str, bool)):
        text = repr(obj)
        return type(obj).__name__, (text[:60] + "…") if len(text) > 60 else text
    cls = type(obj).__name__
    text = repr(obj)
    return cls, (text[:60] + "…") if len(text) > 60 else text


# Module-level singleton — the MCP server process holds exactly one session.
_SESSION: Session | None = None


def get_session() -> Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = Session()
    return _SESSION
