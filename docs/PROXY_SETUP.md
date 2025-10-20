# Seraph MCP Setup Guide

## Overview

**Seraph MCP** provides:
- **20 local tools** (cache management, token counting, semantic search) - always available
- **Backend mounting** (filesystem, github, postgres, slack, etc.) - conditional on `proxy.fastmcp.json` presence
- **Automatic compression** for all responses >1KB using Seraph's Layer 1 compression (50% retention)
- **Centralized metrics** for all MCP interactions (local + backends)

**Key Benefits:**
- **Always functional**: Local tools work even without backend config
- **Simplified Client Config**: One connection for local tools + all backends
- **Unified Compression**: Automatic context optimization for all responses
- **Centralized Observability**: Single metrics namespace for all MCP operations
- **Graceful Degradation**: Missing backends don't prevent server startup

## Installation

### Prerequisites

```bash
# Python 3.11+ with uv package manager
python --version  # Should be 3.11+
uv --version      # Install from https://docs.astral.sh/uv/

# Node.js 18+ (for MCP backend servers)
node --version    # Should be 18+
npm --version
```

### Install Seraph with Proxy

```bash
# Clone repository
git clone https://github.com/yourusername/seraph-mcp.git
cd seraph-mcp

# Install in development mode
uv pip install -e .

# Verify installation
which seraph-mcp  # Should show: /path/to/.venv/bin/seraph-mcp

# Note: seraph-mcp always runs with local tools
# Backends are conditionally mounted if proxy.fastmcp.json exists
```

## Configuration

### Configuration File Format

Seraph Proxy uses the **same `mcpServers` format** as Claude Desktop and other MCP clients.

**File: `proxy.fastmcp.json`**

```json
{
  "$schema": "https://mcp.run/schema/mcp.json",
  "mcpServers": {
    "backend-name": {
      "command": "command-to-run",
      "args": ["arg1", "arg2"],
      "env": {
        "ENV_VAR": "value"
      }
    }
  }
}
```

### Automatic Backend Mounting

Seraph conditionally mounts backends based on configuration presence:

- **`proxy.fastmcp.json` exists** → 20 local tools + backends mounted
- **`proxy.fastmcp.json` missing** → 20 local tools only

**No environment variables required** - backend mounting is automatic.

**Example Usage**
```bash
# With backends
seraph-mcp  # Detects proxy.fastmcp.json and mounts backends

# Without backends
mv proxy.fastmcp.json proxy.fastmcp.json.bak
seraph-mcp  # Runs with 20 local tools only
```

**Architecture Notes:**
- Single unified server always runs
- Backends are **mounted** into the server, not proxied
- Local tools (cache, tokens, semantic search) always available
- Tool naming: local tools unprefixed, backend tools prefixed (e.g., `filesystem_read_file`)

## Backend Server Examples

### 1. Filesystem (Read/Write Local Files)

**Backend**: [@modelcontextprotocol/server-filesystem](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem)

**Configuration**:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/user/documents",
        "/home/user/projects"
      ]
    }
  }
}
```

**Tools Provided**:
- `read_file` - Read file contents
- `write_file` - Write/create files
- `list_directory` - List directory contents
- `search_files` - Search files by pattern
- `get_file_info` - Get file metadata

**Use Case**: Allow AI to read/write files in specific directories.

---

### 2. GitHub (Issues, PRs, Commits)

**Backend**: [@modelcontextprotocol/server-github](https://github.com/modelcontextprotocol/servers/tree/main/src/github)

**Prerequisites**:
```bash
# Create GitHub personal access token
# https://github.com/settings/tokens/new
# Required scopes: repo, read:org, read:user
```

**Configuration**:
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

**Tools Provided**:
- `create_issue` - Create GitHub issue
- `create_pull_request` - Create PR
- `list_issues` - List repo issues
- `get_commit` - Get commit details
- `search_code` - Search code across repos

**Use Case**: Integrate GitHub operations into AI workflows.

---

### 3. PostgreSQL (Database Queries)

**Backend**: [@modelcontextprotocol/server-postgres](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres)

**Prerequisites**:
```bash
# PostgreSQL server running
# Connection string format: postgresql://user:password@host:port/database
```

**Configuration**:
```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_URL": "postgresql://myuser:mypassword@localhost:5432/mydb"
      }
    }
  }
}
```

**Tools Provided**:
- `query` - Execute SQL queries
- `get_schema` - Get database schema
- `list_tables` - List database tables
- `describe_table` - Get table structure

**Use Case**: Allow AI to query databases for data analysis.

**Security Note**: ⚠️ Grant read-only permissions to the database user if possible.

---

### 4. Slack (Channels, Messages)

**Backend**: [@modelcontextprotocol/server-slack](https://github.com/modelcontextprotocol/servers/tree/main/src/slack)

**Prerequisites**:
```bash
# Create Slack app and get bot token
# https://api.slack.com/apps
# Required scopes: channels:read, chat:write, users:read
```

**Configuration**:
```json
{
  "mcpServers": {
    "slack": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-slack"],
      "env": {
        "SLACK_BOT_TOKEN": "xoxb-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}
```

**Tools Provided**:
- `send_message` - Send message to channel
- `list_channels` - List workspace channels
- `get_channel_history` - Get recent messages
- `search_messages` - Search message history

**Use Case**: Integrate Slack messaging into AI workflows.

---

### 5. Google Drive (Files, Sheets)

**Backend**: [@modelcontextprotocol/server-gdrive](https://github.com/modelcontextprotocol/servers/tree/main/src/gdrive)

**Prerequisites**:
```bash
# Create Google Cloud project and enable Drive API
# https://console.cloud.google.com/
# Download credentials.json
```

**Configuration**:
```json
{
  "mcpServers": {
    "gdrive": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-gdrive"],
      "env": {
        "GDRIVE_CREDENTIALS": "/home/user/.credentials/gdrive-credentials.json"
      }
    }
  }
}
```

**Tools Provided**:
- `list_files` - List Drive files
- `read_file` - Read file contents
- `create_file` - Create new file
- `search_files` - Search Drive by query

**Use Case**: Allow AI to read/write Google Drive documents.

---

### 6. Multi-Backend Configuration (Complete Example)

**File: `proxy.fastmcp.json`**

```json
{
  "$schema": "https://mcp.run/schema/mcp.json",
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/user/documents"
      ]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_your_token_here"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_URL": "postgresql://user:pass@localhost:5432/db"
      }
    },
    "slack": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-slack"],
      "env": {
        "SLACK_BOT_TOKEN": "xoxb-your_token_here"
      }
    }
  }
}
```

**Result**: Proxy aggregates tools from all 4 backends into a single MCP server.

## Running Seraph

### With Backends

```bash
# Seraph with backends mounted
seraph-mcp

# Expected output:
# 2025-10-19 15:58:41 - INFO - === Seraph MCP Server Starting ===
# 2025-10-19 15:58:41 - INFO - Backend config detected: proxy.fastmcp.json
# 2025-10-19 15:58:41 - INFO - Loaded proxy config: proxy.fastmcp.json (3 backend servers)
# 2025-10-19 15:58:41 - INFO - Mounting 3 backend servers to Seraph MCP
# 2025-10-19 15:58:41 - INFO - Compression middleware registered: min_size=1000B, timeout=10.0s
# 2025-10-19 15:58:41 - INFO - Starting server with stdio transport
```

### Without Backends

```bash
# Remove backend config
mv proxy.fastmcp.json proxy.fastmcp.json.bak

# Seraph with local tools only
seraph-mcp

# Expected output:
# 2025-10-19 15:58:41 - INFO - === Seraph MCP Server Starting ===
# 2025-10-19 15:58:41 - INFO - No backend config found (checked: proxy.fastmcp.json)
# 2025-10-19 15:58:41 - INFO - Running with 20 local tools (cache, tokens, semantic search)
# 2025-10-19 15:58:41 - INFO - Compression middleware registered: min_size=1000B, timeout=10.0s
# 2025-10-19 15:58:41 - INFO - Starting server with stdio transport
```

### Integration with Claude Desktop

**File: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)**
**File: `%APPDATA%\Claude\claude_desktop_config.json` (Windows)**
**File: `~/.config/Claude/claude_desktop_config.json` (Linux)**

```json
{
  "mcpServers": {
    "seraph": {
      "command": "/path/to/.venv/bin/seraph-mcp"
    }
  }
}
```

**Notes:**
- Create `proxy.fastmcp.json` in the seraph-mcp directory to enable backend mounting
- Without `proxy.fastmcp.json`, Seraph runs with 20 local tools only
- Backend tools are prefixed (e.g., `filesystem_read_file`, `github_create_issue`)
- Local tools are unprefixed (e.g., `cache_get`, `count_tokens`)

**Alternative: Using `uv run`**:
```json
{
  "mcpServers": {
    "seraph": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/seraph-mcp", "seraph-mcp"]
    }
  }
}
}
```

**Verify Integration**:
1. Restart Claude Desktop
2. Open a new conversation
3. Type: "What tools do you have available?"
4. Claude should list tools from all configured backends

## Compression Behavior

### Automatic Compression

Seraph Proxy automatically compresses:
- **Tool results** >1KB (e.g., large file reads, database query results)
- **Resource reads** >1KB (e.g., MCP resource protocol responses)

**Compression Settings**:
- **Algorithm**: Seraph L2 layer (50% retention, ~0.70+ quality score)
- **Threshold**: 1000 bytes (only responses >1KB compressed)
- **Timeout**: 10 seconds (falls back to uncompressed on timeout)

### Example: Filesystem Tool

**Without Proxy** (direct filesystem server):
```
Tool: read_file(/home/user/large-document.txt)
Response: <5000 bytes of text content>
Transmitted to AI: 5000 bytes (uncompressed)
```

**With Proxy** (Seraph compression):
```
Tool: read_file(/home/user/large-document.txt)
Backend Response: <5000 bytes of text content>
Proxy Compression: 5000 bytes → 2500 bytes (50% retention)
Transmitted to AI: 2500 bytes (compressed)
Metadata: {
  "original_size": 5000,
  "compressed_size": 2500,
  "compression_ratio": 0.50,
  "processing_time_ms": 35
}
```

**Benefit**: 50% reduction in context window usage, allowing 2x more tool results per AI conversation.

## Metrics and Observability

### Metrics Namespace

All compression metrics use the `mcp.middleware.*` namespace:

**Tool Result Compression**:
- `mcp.middleware.tool_result.size_before` (gauge) - Original size in bytes
- `mcp.middleware.tool_result.size_after` (gauge) - Compressed size in bytes
- `mcp.middleware.tool_result.compression_ratio` (histogram) - Compression effectiveness
- `mcp.middleware.tool_result.processing_time_ms` (histogram) - Compression latency

**Resource Compression**:
- `mcp.middleware.resource.size_before` (gauge) - Original resource size
- `mcp.middleware.resource.size_after` (gauge) - Compressed resource size
- `mcp.middleware.resource.compression_ratio` (histogram) - Compression effectiveness
- `mcp.middleware.resource.processing_time_ms` (histogram) - Compression latency

### Viewing Metrics

**Option 1: Structured Logs** (default):
```bash
seraph-mcp  # Auto-detects proxy mode from proxy.fastmcp.json 2>&1 | grep "mcp.middleware"
```

**Option 2: Prometheus Export** (requires configuration):
```bash
# Add to seraph configuration (config/seraph.yml)
observability:
  metrics:
    enabled: true
    prometheus_port: 9090
```

**Option 3: Database Export** (SQLite):
```bash
# Metrics automatically stored in observability.db
sqlite3 /path/to/seraph-mcp/observability.db "SELECT * FROM metrics WHERE name LIKE 'mcp.middleware%' ORDER BY timestamp DESC LIMIT 10;"
```

## Troubleshooting

### Problem: "No backend config found"

**Cause**: `proxy.fastmcp.json` not present in project root.

**Impact**: Seraph runs with 20 local tools only (no backends mounted).

**Solution** (if you want backends):
```bash
# Create backend config
cat > proxy.fastmcp.json <<EOF
{
  "$schema": "https://mcp.run/schema/mcp.json",
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
EOF

# Run Seraph (now mounts backends)
seraph-mcp
```

---

### Problem: "Backend server 'xyz' failed to start"

**Cause**: Backend server not installed or incorrect configuration.

**Solution**:
```bash
# Test backend server manually
npx -y @modelcontextprotocol/server-filesystem /tmp

# Check for errors in output
# Common issues:
# - Missing npm package (run: npm install -g @modelcontextprotocol/server-filesystem)
# - Incorrect arguments (check backend documentation)
# - Missing environment variables (check .env or proxy config)
```

---

### Problem: "Compression timeout: returning uncompressed content"

**Cause**: Compression took >10 seconds (very large responses).

**Impact**: Response returned uncompressed (no data loss, just no compression benefit).

**Solution**:
- **Option 1**: Accept uncompressed responses for very large data
- **Option 2**: Increase timeout in code:
  ```python
  # src/proxy.py:149 - Change timeout_seconds
  CompressionMiddleware(
      config=optimization_config,
      min_size_bytes=1000,
      compression_ratio=0.50,
      timeout_seconds=30.0,  # Increase from 10s to 30s
  )
  ```

---

### Problem: "Tools not appearing in Claude Desktop"

**Cause**: Claude Desktop not configured correctly.

**Solution**:
1. **Verify Seraph runs standalone**:
   ```bash
   seraph-mcp
   # Should start without errors
   ```

2. **Check Claude Desktop config path**:
   ```bash
   # macOS
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

   # Linux
   cat ~/.config/Claude/claude_desktop_config.json

   # Windows
   type %APPDATA%\Claude\claude_desktop_config.json
   ```

3. **Verify config syntax**:
   ```bash
   cat claude_desktop_config.json | jq .
   # Should parse without errors
   ```

4. **Check command path**:
   ```bash
   which seraph-mcp
   # Copy this full path to Claude config "command" field
   ```

5. **Restart Claude Desktop completely** (quit and relaunch, not just close window)

---

### Problem: "Permission denied" when starting Seraph

**Cause**: Script not executable or Python environment issues.

**Solution**:
```bash
# Ensure script is executable
chmod +x /path/to/.venv/bin/seraph-mcp

# Or use uv run directly
uv run --directory /path/to/seraph-mcp seraph-mcp

# Or use full Python path
/path/to/.venv/bin/python -m src.server
```

---

### Problem: "ModuleNotFoundError: No module named 'src.server'"

**Cause**: Seraph not installed in development mode.

**Solution**:
```bash
cd /path/to/seraph-mcp
uv pip install -e .

# Verify installation
uv pip list | grep seraph-mcp
# Should show: seraph-mcp (editable installation)
```

## Advanced Configuration

### Custom Compression Settings

**Edit `src/context_optimization/mcp_middleware.py`** to customize compression behavior:

```python
# In CompressionMiddleware.__init__()
self.min_size_bytes = 500         # Compress responses >500 bytes (more aggressive)
self.compression_ratio = 0.30     # 30% retention (higher compression, lower quality)
self.timeout_seconds = 15.0       # 15 second timeout (for very large responses)
```

**Trade-offs**:
- **Lower `min_size_bytes`**: Compress more responses, but adds overhead for small responses
- **Lower `compression_ratio`**: Higher compression, but may reduce quality/readability
- **Higher `timeout_seconds`**: Handles larger responses, but slows down timeouts

### Per-Backend Environment Variables

**Use Case**: Different API keys/credentials per backend.

**Example**:
```json
{
  "mcpServers": {
    "github-work": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_work_token_here",
        "GITHUB_ORG": "my-company"
      }
    },
    "github-personal": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_personal_token_here",
        "GITHUB_ORG": "my-username"
      }
    }
  }
}
```

**Result**: Two separate GitHub backends with different credentials (both mounted into Seraph).

### Custom Backend Servers

**Use Case**: Use custom/private MCP servers instead of official ones.

**Example**:
```json
{
  "mcpServers": {
    "custom-api": {
      "command": "/usr/local/bin/my-custom-mcp-server",
      "args": ["--port", "8080", "--config", "/etc/api-config.json"],
      "env": {
        "API_KEY": "secret_key_here",
        "LOG_LEVEL": "debug"
      }
    }
  }
}
```

**Requirements**:
- Custom server must implement MCP protocol (stdio transport)
- Must accept stdio as input/output (standard MCP requirement)

## Performance Characteristics

### Startup Time
- **Cold Start**: ~300ms (includes Python interpreter + dependency loading)
- **Warm Start**: ~150ms (Python interpreter cached)
- **Per-Backend Overhead**: ~50ms per backend server (serial startup)

**Example**: Seraph with 5 backends = 150ms + (5 × 50ms) = 400ms total startup

### Runtime Overhead
- **Tool Call (no compression)**: <1ms (pass-through, response <1KB)
- **Tool Call (compressed)**: 10-50ms (compression overhead, response >1KB)
- **Resource Read (compressed)**: 15-60ms (depends on resource size)

**Example Latencies**:
- `read_file` (500 bytes): ~5ms (no compression)
- `read_file` (5KB): ~35ms (compression: 5KB → 2.5KB)
- `read_file` (50KB): ~150ms (compression: 50KB → 25KB)

### Memory Usage
- **Base Server**: ~50MB (Python interpreter + FastMCP)
- **Seraph Compressor**: +50MB (model loaded on first compression call)
- **Per-Backend**: ~20MB (Node.js backend server process)

**Example**: Seraph with 3 backends = 50MB + 50MB + (3 × 20MB) = 160MB total

## Security Considerations

### Credential Management

⚠️ **Never commit credentials to version control**

**Best Practices**:
1. **Use environment variables**:
   ```json
   {
     "mcpServers": {
       "github": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-github"],
         "env": {
           "GITHUB_TOKEN": "${GITHUB_TOKEN}"
         }
       }
     }
   }
   ```

   ```bash
   export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
   seraph-mcp  # Auto-detects proxy mode from proxy.fastmcp.json
   ```

2. **Use credential files with restricted permissions**:
   ```bash
   chmod 600 ~/.credentials/github-token.txt
   ```

3. **Use secret management systems** (AWS Secrets Manager, 1Password CLI, etc.):
   ```json
   {
     "mcpServers": {
       "github": {
         "command": "sh",
         "args": ["-c", "GITHUB_TOKEN=$(op read op://vault/github/token) npx -y @modelcontextprotocol/server-github"]
       }
     }
   }
   ```

### Filesystem Access Control

⚠️ **Limit filesystem backend to specific directories**

**Bad** (entire home directory):
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    }
  }
}
```

**Good** (specific directories only):
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/user/documents",
        "/home/user/projects/work"
      ]
    }
  }
}
```

### Database Access Control

⚠️ **Use read-only database users when possible**

**Create Read-Only User** (PostgreSQL):
```sql
-- Create read-only user
CREATE USER mcp_readonly WITH PASSWORD 'secure_password';

-- Grant read-only access to specific schema
GRANT CONNECT ON DATABASE mydb TO mcp_readonly;
GRANT USAGE ON SCHEMA public TO mcp_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mcp_readonly;
```

**Use in Proxy Config**:
```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "POSTGRES_URL": "postgresql://mcp_readonly:secure_password@localhost:5432/mydb"
      }
    }
  }
}
```

## FAQ

### Q: Can I use Seraph without compression?

**A**: Not currently. Compression is built-in and always enabled (but only affects responses >1KB).

**Workaround**: Modify `src/context_optimization/mcp_middleware.py` to skip compression registration in `src/server.py`

---

### Q: Can I compress responses <1KB?

**A**: Yes, change `min_size_bytes` in `src/context_optimization/mcp_middleware.py`:

```python
# In CompressionMiddleware.__init__()
self.min_size_bytes = 100  # Compress responses >100 bytes
```

**Trade-off**: Adds compression overhead to small responses (may slow down tool calls).

---

### Q: Can I use different compression settings per backend?

**A**: Not currently. All backends use the same compression settings.

**Future Enhancement**: Planned for v2.0 (see SDD §10.4.2.1 Future Enhancements).

---

### Q: How do I disable compression for specific tools?

**A**: Not supported at middleware level. Compression applies to all responses >1KB.

**Workaround**: Modify backend server to return responses <1KB (bypass threshold).

---

### Q: Can I see compression metrics per backend?

**A**: Not currently. Metrics are aggregated across all backends.

**Workaround**: Run separate Seraph instances (one per backend) with different metric namespaces.

---

### Q: Does Seraph support HTTP transport?

**A**: No. Seraph only supports stdio transport (standard MCP protocol).

**Rationale**: FastMCP currently only supports stdio for backend mounting (upstream limitation).

---

### Q: Can I use Seraph with non-Node.js backends?

**A**: Yes! Any backend that implements MCP stdio protocol works.

**Example (Python backend)**:
```json
{
  "mcpServers": {
    "python-backend": {
      "command": "python",
      "args": ["/path/to/my_mcp_server.py"]
    }
  }
}
```

---

### Q: How do I test compression is working?

**Test Method**:
1. Create a test file >1KB:
   ```bash
   echo "Lorem ipsum dolor sit amet..." > /tmp/test-large.txt
   # Repeat until >1000 bytes
   ```

2. Run proxy with filesystem backend:
   ```json
   {
     "mcpServers": {
       "filesystem": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
       }
     }
   }
   ```

3. Call `read_file` tool via Claude Desktop or MCP Inspector

4. Check logs for compression metrics:
   ```bash
   seraph-mcp  # Auto-detects proxy mode from proxy.fastmcp.json 2>&1 | grep "mcp.middleware.tool_result"
   ```

**Expected Output**:
```
{"message": "Tool result compressed", "original_size": 5000, "compressed_size": 2500, "compression_ratio": 0.50, "processing_time_ms": 35}
```

## Additional Resources

- **Official MCP Servers**: https://github.com/modelcontextprotocol/servers
- **MCP Protocol Spec**: https://spec.modelcontextprotocol.io/
- **Seraph Architecture**: `/docs/SDD.md` (§10.4.2.1 - Cross-Server MCP Proxy Architecture)
- **FastMCP Documentation**: https://github.com/jlowin/fastmcp
- **Issue Tracker**: https://github.com/yourusername/seraph-mcp/issues

## Support

**Questions?** Open an issue: https://github.com/yourusername/seraph-mcp/issues/new

**Bug Reports**: Include:
1. Proxy config (`proxy.fastmcp.json`)
2. Error logs (`seraph-mcp  # Auto-detects proxy mode from proxy.fastmcp.json 2>&1 | tee error.log`)
3. Python version (`python --version`)
4. Backend versions (`npx @modelcontextprotocol/server-filesystem --version`)

**Feature Requests**: Tag with `enhancement` label.
