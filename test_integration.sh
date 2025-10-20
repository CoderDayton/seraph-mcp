#!/bin/bash
set -e

echo "=== Integration Test: Unified seraph-mcp Entry Point ==="
echo ""

# Test 1: Server mode (no proxy config)
echo "Test 1: Server mode (no proxy.fastmcp.json)"
mv proxy.fastmcp.json proxy.fastmcp.json.bak 2>/dev/null || true
timeout 1 uv run python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from src.server import main
import signal, sys
signal.signal(signal.SIGALRM, lambda *_: sys.exit(0))
signal.alarm(1)
try:
    main()
except SystemExit:
    pass
" 2>&1 | grep "Server Mode" && echo "✅ PASS" || echo "❌ FAIL"
mv proxy.fastmcp.json.bak proxy.fastmcp.json 2>/dev/null || true
echo ""

# Test 2: Proxy mode (with proxy.fastmcp.json)
echo "Test 2: Proxy mode (with proxy.fastmcp.json)"
timeout 1 uv run python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from src.server import main
import signal, sys
signal.signal(signal.SIGALRM, lambda *_: sys.exit(0))
signal.alarm(1)
try:
    main()
except SystemExit:
    pass
" 2>&1 | grep "Proxy Mode" && echo "✅ PASS" || echo "❌ FAIL"
echo ""

# Test 3: Env var override
echo "Test 3: SERAPH_PROXY_CONFIG env var override"
cat > /tmp/test_proxy.json << 'INNER_EOF'
{"mcpServers": {"test": {"command": "echo", "args": [], "env": {}}}}
INNER_EOF
SERAPH_PROXY_CONFIG=/tmp/test_proxy.json timeout 1 uv run python -c "
import logging, os
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
os.environ['SERAPH_PROXY_CONFIG'] = '/tmp/test_proxy.json'
from src.server import main
import signal, sys
signal.signal(signal.SIGALRM, lambda *_: sys.exit(0))
signal.alarm(1)
try:
    main()
except SystemExit:
    pass
" 2>&1 | grep "/tmp/test_proxy.json" && echo "✅ PASS" || echo "❌ FAIL"
rm /tmp/test_proxy.json
echo ""

# Test 4: Type checking
echo "Test 4: Type checking (mypy)"
uv run mypy src/proxy.py src/server.py --no-error-summary 2>&1 | grep -E "(proxy|server).py:" > /dev/null && echo "❌ FAIL (type errors found)" || echo "✅ PASS"
echo ""

# Test 5: Command availability
echo "Test 5: Command availability"
uv run which seraph-mcp > /dev/null && echo "✅ seraph-mcp found" || echo "❌ seraph-mcp NOT found"
uv run which seraph-proxy > /dev/null 2>&1 && echo "❌ seraph-proxy still exists (should be removed)" || echo "✅ seraph-proxy removed"
echo ""

echo "=== All Tests Complete ==="
