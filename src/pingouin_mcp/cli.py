"""
Command-line entry point for Pingouin MCP.

Mirrors ``mne-mcp``'s CLI so the two servers are operated the same way.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pingouin-mcp",
        description="Model Context Protocol server for statistical analysis with Pingouin.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("serve", help="Run the MCP server over stdio (default).")
    sub.add_parser("status", help="Print detected capabilities as JSON.")
    sub.add_parser("tools", help="List the registered tool names.")

    p_cfg = sub.add_parser("show-config", help="Print the resolved configuration.")
    p_cfg.add_argument("--json", action="store_true", help="Emit JSON.")

    args = parser.parse_args(argv)

    if args.command in (None, "serve"):
        from pingouin_mcp.server import mcp

        mcp.run()
        return 0

    if args.command == "status":
        from pingouin_mcp.config import detect_capabilities

        caps = detect_capabilities()
        print(json.dumps(caps, indent=2, ensure_ascii=False))
        if not caps.get("pingouin"):
            print("\nPingouin is not importable — install it with `pip install pingouin`.",
                  file=sys.stderr)
            return 1
        return 0

    if args.command == "show-config":
        from pingouin_mcp.config import get_runtime_config

        cfg = get_runtime_config()
        if args.json:
            print(json.dumps(cfg, indent=2, ensure_ascii=False))
        else:
            for k, v in sorted(cfg.items()):
                print(f"{k:16s} {v}")
        return 0

    if args.command == "tools":
        from pingouin_mcp import operations

        for name in sorted(operations.OPERATIONS):
            print(name)
        print(f"\n{len(operations.OPERATIONS)} tools registered.", file=sys.stderr)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
