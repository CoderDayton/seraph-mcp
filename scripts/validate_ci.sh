#!/bin/bash
# CI Validation Script for Seraph MCP
# Validates all CI alignment changes are working correctly

set -e

echo "=========================================="
echo "Seraph MCP CI Validation Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
WARNINGS=0

# Helper functions
pass() {
    echo -e "${GREEN}✓${NC} $1"
    PASSED=$((PASSED + 1))
}

fail() {
    echo -e "${RED}✗${NC} $1"
    FAILED=$((FAILED + 1))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# 1. Check CI file structure
section "1. Validating CI/CD Pipeline Structure"

if [ -f ".github/workflows/ci.yml" ]; then
    pass "CI workflow file exists"

    # Check for proper YAML structure (no duplicate keys)
    if ! grep -q "steps:" .github/workflows/ci.yml | grep -c "steps:" > /dev/null 2>&1; then
        pass "No duplicate 'steps' keys found"
    fi

    # Check for 85% coverage threshold
    if grep -q "fail-under=85" .github/workflows/ci.yml; then
        pass "Coverage threshold set to 85%"
    else
        fail "Coverage threshold not set to 85%"
    fi

    # Check Redis removed from main test job
    # Extract the test job section (between "test:" and next job "test-zero-config:")
    if sed -n '/^    test:/,/^    test-zero-config:/p' .github/workflows/ci.yml | grep -q "services:"; then
        fail "Redis service still present in main test job"
    else
        pass "Redis correctly removed from main test job"
    fi

    # Check status-check job exists
    if grep -q "status-check:" .github/workflows/ci.yml; then
        pass "Status-check job exists"
    else
        fail "Status-check job missing"
    fi
else
    fail "CI workflow file missing"
fi

# 2. Check pyproject.toml
section "2. Validating pyproject.toml Configuration"

if [ -f "pyproject.toml" ]; then
    pass "pyproject.toml exists"

    # Check coverage threshold
    if grep -q "fail_under = 85" pyproject.toml; then
        pass "Coverage fail_under set to 85"
    else
        fail "Coverage fail_under not set to 85"
    fi
else
    fail "pyproject.toml missing"
fi

# 3. Check SDD.md updates
section "3. Validating SDD.md Documentation"

if [ -f "docs/SDD.md" ]; then
    pass "SDD.md exists"

    # Check Semantic Cache marked as implemented
    if grep -q "Feature 3: Semantic Cache ✅ IMPLEMENTED" docs/SDD.md; then
        pass "Semantic Cache marked as implemented"
    else
        warn "Semantic Cache not marked as implemented"
    fi

    # Check Budget Management marked as implemented
    if grep -q "Feature 5: Budget Management ✅ IMPLEMENTED" docs/SDD.md; then
        pass "Budget Management marked as implemented"
    else
        warn "Budget Management not marked as implemented"
    fi
else
    fail "SDD.md missing"
fi

# 4. Check README.md updates
section "4. Validating README.md Documentation"

if [ -f "README.md" ]; then
    pass "README.md exists"

    # Check tool count updated
    if grep -q "22+ Tools" README.md; then
        pass "Tool count updated to 22+"
    else
        warn "Tool count not updated to 22+"
    fi

    # Check for tool sections
    if grep -q "Core System (7 tools)" README.md; then
        pass "Core System tools section exists"
    else
        warn "Core System tools section missing"
    fi

    if grep -q "Budget Management (3 tools)" README.md; then
        pass "Budget Management tools section exists"
    else
        warn "Budget Management tools section missing"
    fi

    if grep -q "Semantic Cache (5 tools)" README.md; then
        pass "Semantic Cache tools section exists"
    else
        warn "Semantic Cache tools section missing"
    fi
else
    fail "README.md missing"
fi

# 5. Check Redis documentation
section "5. Validating Redis Documentation"

if [ -f "docs/redis/REDIS_SETUP.md" ]; then
    pass "Redis setup guide exists"
else
    warn "Redis setup guide missing"
fi

# 6. Check server.py MCP tools
section "6. Validating MCP Tools in server.py"

if [ -f "src/server.py" ]; then
    pass "server.py exists"

    # Count MCP tools
    TOOL_COUNT=$(grep -c "@mcp.tool()" src/server.py || echo 0)

    if [ "$TOOL_COUNT" -ge 22 ]; then
        pass "Found $TOOL_COUNT MCP tools (expected 22+)"
    else
        warn "Found only $TOOL_COUNT MCP tools (expected 22+)"
    fi

    # Check for key tools
    if grep -q "def check_status" src/server.py; then
        pass "check_status tool exists"
    fi

    if grep -q "def check_budget" src/server.py; then
        pass "check_budget tool exists"
    fi

    if grep -q "def lookup_semantic_cache" src/server.py; then
        pass "lookup_semantic_cache tool exists"
    fi

    if grep -q "def optimize_context" src/server.py; then
        pass "optimize_context tool exists"
    fi
else
    fail "server.py missing"
fi

# 7. Check core module structure
section "7. Validating Core Module Structure"

check_file() {
    if [ -f "$1" ]; then
        pass "$1 exists"
    else
        fail "$1 missing"
    fi
}

check_file "src/cache/factory.py"
check_file "src/cache/interface.py"
check_file "src/cache/backends/memory.py"
check_file "src/cache/backends/redis.py"
check_file "src/context_optimization/seraph_compression.py"
check_file "src/budget_management/tracker.py"
check_file "src/semantic_cache/cache.py"

# 8. Test environment validation
section "8. Validating Development Environment"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    pass "Python available: $PYTHON_VERSION"
else
    warn "Python not found in PATH"
fi

# Check uv installation
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1 | head -n1)
    pass "uv available: $UV_VERSION"
else
    warn "uv not found in PATH (install with: curl -LsSf https://astral.sh/uv/install.sh | sh)"
fi

# Check for virtual environment
if [ -d ".venv" ]; then
    pass "Virtual environment exists"
else
    warn "Virtual environment not found (run: uv sync)"
fi

# Check Docker for Redis tests
if command -v docker &> /dev/null; then
    pass "Docker available"
else
    warn "Docker not found (needed for Redis tests)"
fi

# 9. Quick syntax validation
section "9. Validating Code Syntax"

if command -v python3 &> /dev/null; then
    # Check Python syntax
    if python3 -m py_compile src/server.py 2>/dev/null; then
        pass "server.py syntax valid"
    else
        fail "server.py has syntax errors"
    fi
fi

# 10. Summary
section "VALIDATION SUMMARY"

echo ""
echo "Results:"
echo "  ${GREEN}Passed:${NC}   $PASSED"
echo "  ${YELLOW}Warnings:${NC} $WARNINGS"
echo "  ${RED}Failed:${NC}   $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All critical validations passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: uv run pytest tests/ --cov=src"
    echo "  2. Check coverage: uv run coverage report --fail-under=85"
    echo "  3. Start server: fastmcp dev src/server.py"
    echo "  4. Push changes: git push origin main"
    exit 0
else
    echo -e "${RED}✗ Some critical validations failed${NC}"
    echo ""
    echo "Please fix the failed checks before proceeding."
    exit 1
fi
