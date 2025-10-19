#!/usr/bin/env python3
"""
Seraph MCP - FastMCP Runner

This script uses the FastMCP CLI to run the server properly.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the Seraph MCP server using FastMCP CLI."""
    # Get the project root
    project_root = Path(__file__).parent

    # Run the FastMCP CLI with the server
    cmd = [sys.executable, "-m", "fastmcp.cli.main", "run", str(project_root / "src" / "server.py") + ":mcp"]

    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error running server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
