"""
Configuration and capability detection for Pingouin MCP.

Mirrors the configuration model of MNE-MCP so that the two servers can be
deployed, configured and reasoned about in the same way.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict = {
    # Where generated figures are written.
    "results_dir": None,  # resolved at runtime to ~/.pingouin-mcp/results
    # Default significance level used when a tool does not specify one.
    "alpha": 0.05,
    # Default multiple-comparison correction method (pingouin's naming).
    "correction": "fdr_bh",
    # Default number of decimal places for rounded output.
    "decimals": 4,
    # Serialize a tool call for this many seconds before giving up.
    "timeout": 300,
    # Default confidence level for intervals.
    "confidence": 0.95,
}


def get_config_path() -> Path:
    """Path of the optional user configuration file."""
    override = os.environ.get("PINGOUIN_MCP_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".pingouin-mcp" / "config.json"


def load_config() -> dict:
    """Load user configuration, falling back to defaults."""
    cfg = dict(DEFAULT_CONFIG)
    path = get_config_path()
    if path.exists():
        try:
            user = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(user, dict):
                cfg.update(user)
        except Exception:
            # A malformed config file must never prevent the server from starting.
            pass
    return cfg


_RUNTIME: dict | None = None


def get_runtime_config() -> dict:
    """Process-wide resolved configuration."""
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = load_config()
    return _RUNTIME


def get_timeout() -> int:
    """Timeout in seconds for a single tool operation."""
    env = os.environ.get("PINGOUIN_MCP_TIMEOUT")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return int(get_runtime_config().get("timeout", 300))


def get_results_dir() -> Path:
    """Directory where figures are written. Created on first use."""
    env = os.environ.get("PINGOUIN_MCP_RESULTS_DIR")
    if env:
        path = Path(env)
    else:
        configured = get_runtime_config().get("results_dir")
        path = Path(configured) if configured else Path.home() / ".pingouin-mcp" / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Directory searched by the file-listing tool."""
    env = os.environ.get("PINGOUIN_MCP_DATA_DIR")
    if env:
        return Path(env)
    return Path.cwd()


def detect_capabilities() -> dict:
    """
    Report which optional pieces of the scientific stack are importable.

    Tools use this to fail with an actionable message instead of an
    ImportError traceback.
    """
    caps: dict = {}

    def _probe(module: str, attr: str = "__version__"):
        try:
            mod = __import__(module)
            return getattr(mod, attr, "unknown")
        except Exception:
            return None

    caps["pingouin_version"] = _probe("pingouin")
    caps["pingouin"] = caps["pingouin_version"] is not None

    for name, module in [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("statsmodels", "statsmodels"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]:
        version = _probe(module)
        caps[name] = version is not None
        caps[f"{name}_version"] = version

    # Pingouin delegates some procedures to statsmodels; report it explicitly
    # because its absence changes which analyses are available.
    if caps.get("statsmodels"):
        caps["mixed_models"] = True
    else:
        caps["mixed_models"] = False

    caps["results_dir"] = str(get_results_dir())
    caps["config_path"] = str(get_config_path())
    caps["config_exists"] = get_config_path().exists()
    return caps
