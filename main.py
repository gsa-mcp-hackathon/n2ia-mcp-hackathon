"""Root entry point for local development.

Run locally:
    uv run python main.py

Or via the installed script:
    uv run mcp-server-n2ia
"""

from mcp_server_n2ia.app import main

if __name__ == "__main__":
    main()
