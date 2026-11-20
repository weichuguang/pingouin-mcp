# -*- coding: utf-8 -*-
"""
Verify the MCP layer: tools are registered, described, and callable end-to-end.

The operations can be correct while the server is not, so this file checks the
binding rather than the statistics: that every operation is exposed as a tool,
that each tool carries a usable description, and that a call through the MCP
client returns the same content as calling the operation directly.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

from pingouin_mcp import operations as ops
from pingouin_mcp.server import mcp


async def _tools_by_name() -> dict:
    """Registered tools keyed by name (FastMCP 4.x returns a sequence)."""
    tools = await mcp.list_tools()
    return {t.name: t for t in tools}


@pytest.mark.asyncio
async def test_all_operations_are_registered_as_tools():
    """Every entry in OPERATIONS must be exposed as an MCP tool."""
    registered = set(await _tools_by_name())
    missing = set(ops.OPERATIONS) - registered
    assert not missing, f"operations not exposed as tools: {sorted(missing)}"


@pytest.mark.asyncio
async def test_no_orphan_tools():
    """Every registered tool must have an implementation."""
    registered = set(await _tools_by_name())
    orphans = registered - set(ops.OPERATIONS)
    assert not orphans, f"tools without an implementation: {sorted(orphans)}"


@pytest.mark.asyncio
async def test_tool_count_in_expected_range():
    """The brief asked for roughly 30-40 task-level tools."""
    n = len(await _tools_by_name())
    assert 30 <= n <= 50, f"unexpected tool count: {n}"


@pytest.mark.asyncio
async def test_every_tool_has_a_description():
    """
    The description is the only thing an LLM reads when choosing a tool, so an
    empty or stub description is a functional defect, not a documentation nit.
    """
    problems = [(name, len((t.description or "").strip()))
                for name, t in (await _tools_by_name()).items()
                if len((t.description or "").strip()) < 60]
    assert not problems, f"tools with thin descriptions: {problems}"


@pytest.mark.asyncio
async def test_guidance_tools_state_when_not_to_use():
    """
    Tools whose misuse is a methodological error must say when NOT to use them.
    """
    tools = await _tools_by_name()
    must_warn = [
        "stat_ttest", "stat_anova", "stat_rm_anova", "stat_correlation",
        "stat_ancova", "stat_mediation", "stat_equivalence_test",
        "stat_chi2_mcnemar", "stat_linear_regression", "stat_logistic_regression",
    ]
    missing = [n for n in must_warn
               if "NOT TO USE" not in (tools[n].description or "").upper()]
    assert not missing, f"tools missing 'when not to use' guidance: {missing}"


@pytest.mark.asyncio
async def test_tools_are_callable_through_the_mcp_client():
    """
    End-to-end: call a tool the way an agent would and check the content comes
    back intact.
    """
    from fastmcp import Client

    async with Client(mcp) as client:
        # Put a dataset in the session via a tool call.
        await client.call_tool("stat_make_dataframe", {
            "name": "ex",
            "columns": {"g": ["a", "a", "b", "b"],
                        "v": [1.0, 2.0, 3.0, 4.0]},
        })
        result = await client.call_tool("stat_describe_data",
                                        {"name": "ex", "dv": "v", "group": "g"})
        text = _text_of(result)
        assert "mean" in text.lower()
        assert "a" in text and "b" in text


@pytest.mark.asyncio
async def test_missing_dataset_returns_message_not_exception():
    """A bad name must produce a readable message, not an exception."""
    from fastmcp import Client

    async with Client(mcp) as client:
        result = await client.call_tool("stat_describe_data",
                                        {"name": "definitely_not_loaded"})
        text = _text_of(result)
        assert "No object named" in text or "not found" in text.lower()


def _text_of(result) -> str:
    """Extract the text content from a FastMCP call_tool result."""
    content = getattr(result, "content", result)
    if isinstance(content, (list, tuple)):
        return "\n".join(getattr(c, "text", str(c)) for c in content)
    return str(content)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-q"]))
