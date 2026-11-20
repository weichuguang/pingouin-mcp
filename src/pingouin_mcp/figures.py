"""
Matplotlib figure capture.

Tools never return images; they return *paths* to PNG files written under the
results directory. The agent then reads those paths if it needs to inspect the
plot. This keeps tool output textual and token-cheap.

The server runs headless (Agg backend), so nothing is ever shown on screen and
nothing blocks waiting for a window.
"""

from __future__ import annotations

from pathlib import Path

_FIG_COUNTER = 0


def open_figure_numbers() -> set[int]:
    """Numbers of currently-open matplotlib figures."""
    import matplotlib.pyplot as plt

    return set(plt.get_fignums())


def capture_new_figures(before: set[int], out_dir: Path, prefix: str = "fig") -> list[str]:
    """
    Save every figure opened since ``before`` and close it.

    Returns the list of written paths, sorted by figure number so the output
    order matches the order the figures were created.
    """
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    new = sorted(set(plt.get_fignums()) - before)
    paths: list[str] = []

    global _FIG_COUNTER
    for num in new:
        fig = plt.figure(num)
        quality = "low" if fig.get_size_inches()[0] > 15 else "high"
        path = out_dir / f"{prefix}_{_FIG_COUNTER:03d}.png"
        try:
            fig.savefig(path, dpi=(110 if quality == "low" else 150),
                        bbox_inches="tight")
            paths.append(str(path))
        except Exception:
            # A figure that cannot be written must not fail the whole tool call.
            pass
        finally:
            _FIG_COUNTER += 1
            plt.close(fig)

    return paths


def close_all() -> None:
    """Close every open figure (used by session reset)."""
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
