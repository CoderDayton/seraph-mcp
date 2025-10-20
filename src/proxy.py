"""
Seraph MCP Backend Mounting
----------------------------

Mounts external backend servers (filesystem, GitHub, Postgres, etc.) onto the
main Seraph MCP instance, enabling hybrid architecture where local optimization
tools coexist with proxied external tools.

Architecture:
    Seraph MCP (main) → Local tools (cache_*, budget_*, optimize_*)
                      ↓ (mount)
                      → Backend Proxy → filesystem_*, github_*, postgres_*

    Compression middleware applies transparently to BOTH paths.

Per SDD §10.4.2: Unified server architecture eliminates mode switching,
provides single entry point with automatic backend mounting.

Usage:
    # In server.py:
    mcp = FastMCP("Seraph MCP")  # Create with local tools
    mount_backends_to_server(mcp)  # Auto-mount if proxy.fastmcp.json exists

Configuration (proxy.fastmcp.json):
    {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
                "env": {}
            },
            "github": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"}
            }
        }
    }

Notes:
    - Config path is hardcoded to proxy.fastmcp.json
    - If file missing/empty → runs with local tools only
    - Backend tools auto-prefixed: filesystem_read, github_list_repos
    - Local tools unprefixed: cache_get, budget_status
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


def load_proxy_config() -> dict[str, Any]:
    """
    Load backend server configuration from hardcoded proxy.fastmcp.json.

    Hardcoded configuration (no env vars):
        - Always loads from proxy.fastmcp.json
        - No environment variable overrides
        - No fallback configs
        - Completely transparent to users

    Returns:
        Configuration dictionary with mcpServers structure

    Raises:
        FileNotFoundError: If proxy.fastmcp.json doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    # Hardcoded config path - no env vars, no parameters
    config_path_obj = Path("proxy.fastmcp.json")

    if not config_path_obj.exists():
        raise FileNotFoundError(
            "Proxy config not found: proxy.fastmcp.json\n"
            "Create proxy.fastmcp.json with backend server definitions.\n"
            "See docs/SDD.md §10.4.2 for configuration format."
        )

    # Load and parse config
    with open(config_path_obj) as f:
        config: dict[str, Any] = json.load(f)

    # Validate basic structure
    if "mcpServers" not in config:
        raise ValueError(
            "Invalid proxy config: missing 'mcpServers' key in proxy.fastmcp.json\n"
            "Expected format: {'mcpServers': {'server-name': {...}}}"
        )

    logger.info(
        "Loaded proxy config: proxy.fastmcp.json (%d backend servers)",
        len(config.get("mcpServers", {})),
    )

    return config


def mount_backends_to_server(mcp: FastMCP, config: dict[str, Any] | None = None) -> None:
    """
    Mount backend servers onto existing Seraph MCP server instance.

    Hybrid architecture (Per SDD §10.4.2):
        - Mounts external backends (filesystem, github, postgres) to existing server
        - Each backend's tools auto-prefixed (filesystem_read, github_create_issue)
        - Local tools remain unprefixed (cache_get, budget_check_status)
        - Compression middleware applies transparently to BOTH local + backend tools

    Args:
        mcp: Existing FastMCP server instance with local tools
        config: Backend server configuration (auto-loads if None)

    Architecture:
        1. Load backend config from proxy.fastmcp.json (or use provided)
        2. Create composite proxy from multi-server config
        3. Mount the proxy onto existing mcp instance
        4. All tools/resources now unified under single server
    """
    # Load config if not provided
    if config is None:
        try:
            config = load_proxy_config()
        except FileNotFoundError:
            logger.info("No proxy.fastmcp.json found - running with local tools only")
            return

    backend_count = len(config.get("mcpServers", {}))

    if backend_count == 0:
        logger.info("Empty backend config - running with local tools only")
        return

    logger.info("Mounting %d backend servers to Seraph MCP", backend_count)

    # Create composite proxy from multi-server config
    # Per FastMCP docs: as_proxy(config_dict) automatically creates ProxyClient
    # for each server and handles session isolation
    backend_proxy = FastMCP.as_proxy(
        config,
        name="Seraph Backend Proxy",
    )

    logger.info("Backend proxy created with %d servers", backend_count)

    # Mount the proxy onto existing server (NO prefix - tools auto-prefixed by backend name)
    # Per FastMCP docs: Mounted components accessible with server name prefixes
    # Example: filesystem_read_file, github_create_issue
    mcp.mount(backend_proxy)

    logger.info(
        "Backends mounted successfully - %d backend tools now unified with local tools",
        backend_count,
    )


def main() -> None:
    """
    Standalone entry point for backward compatibility.

    NOTE: This function is deprecated. Use `seraph-mcp` which auto-detects
    proxy.fastmcp.json and mounts backends automatically.

    For manual testing only.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.warning("=== Running in DEPRECATED standalone mode ===")
    logger.warning("Use `seraph-mcp` instead for unified hybrid architecture")

    try:
        # Create minimal server for testing
        from fastmcp import FastMCP

        test_server = FastMCP("Seraph Backend Test")
        mount_backends_to_server(test_server)

        logger.info("Starting test server with stdio transport")
        test_server.run()

    except FileNotFoundError as e:
        logger.error("Configuration error: %s", e)
        raise SystemExit(1) from e

    except ValueError as e:
        logger.error("Configuration validation error: %s", e)
        raise SystemExit(1) from e

    except Exception as e:
        logger.error("Server startup failed: %s", e, exc_info=True)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
