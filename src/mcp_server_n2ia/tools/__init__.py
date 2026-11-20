"""Tools submodule — one tool per file.

Each tool module (e.g. `example_tool.py`) exposes a `register(mcp)` function
that attaches its `@mcp.tool`-decorated function to the server. This file
aggregates them into a single `register_tools(mcp)` that `app.py` calls.

To add a tool:
  1. Create `tools/<your_tool>.py` with a `register(mcp)` function.
  2. Import it below and call `<your_tool>.register(mcp)` inside register_tools.
"""

from __future__ import annotations

from mcp_server_n2ia.tools import (
    contract_awards,
    it_spending,
    retrieve_service_category,
    analyze_contract_overlap,
)

def register_tools(mcp) -> None:
    """Register all tools with the MCP server."""
    contract_awards.register(mcp)
    it_spending.register(mcp)
    retrieve_service_category.register(mcp)
    analyze_contract_overlap.register(mcp)


__all__ = ["register_tools"]