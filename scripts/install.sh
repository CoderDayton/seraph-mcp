#!/bin/bash
# Seraph MCP — Installation Script
# Automatically installs dependencies using the best available package manager

set -e  # Exit on error

echo "================================================"
echo "  Seraph MCP — Installation Script"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Verify Python >= 3.10
python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || {
    echo -e "${RED}❌ Python 3.10+ required${NC}"
    exit 1
}
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check for package managers
echo "Detecting package manager..."

if command -v uv &> /dev/null; then
    echo -e "${GREEN}Found uv package manager${NC}"
    PACKAGE_MANAGER="uv"
elif command -v pip3 &> /dev/null; then
    echo -e "${YELLOW}Found pip3 (uv recommended for faster installs)${NC}"
    PACKAGE_MANAGER="pip3"
elif command -v pip &> /dev/null; then
    echo -e "${YELLOW}Found pip (uv recommended for faster installs)${NC}"
    PACKAGE_MANAGER="pip"
else
    echo -e "${RED}❌ No package manager found${NC}"
    echo "Please install pip or uv"
    exit 1
fi
echo ""

# Ask for installation type
echo "Select installation type:"
echo "  1) Development (includes testing and linting tools)"
echo "  2) Production (core dependencies only)"
echo ""
read -p "Enter choice [1]: " install_type
install_type=${install_type:-1}

echo ""
echo "Installing dependencies..."
echo ""

# Install based on package manager and type
if [ "$PACKAGE_MANAGER" = "uv" ]; then
    if [ "$install_type" = "1" ]; then
        echo "Running: uv sync --all-extras"
        uv sync --all-extras
    else
        echo "Running: uv sync"
        uv sync
    fi
else
    if [ "$install_type" = "1" ]; then
        echo "Running: $PACKAGE_MANAGER install -e \".[dev]\""
        $PACKAGE_MANAGER install -e ".[dev]"
    else
        echo "Running: $PACKAGE_MANAGER install -e ."
        $PACKAGE_MANAGER install -e .
    fi
fi

echo ""
echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "
from src.cache import create_cache
from src.config import load_config
from src.observability import get_observability
print('✓ All core imports work')
" 2>&1 || {
    echo -e "${RED}❌ Installation verification failed${NC}"
    exit 1
}

echo -e "${GREEN}✓ Installation verified${NC}"
echo ""

# Offer to install pre-commit hooks
if [ "$install_type" = "1" ]; then
    read -p "Install pre-commit hooks? [Y/n]: " install_hooks
    install_hooks=${install_hooks:-Y}

    if [[ "$install_hooks" =~ ^[Yy]$ ]]; then
        echo ""
        echo "Installing pre-commit hooks..."
        pre-commit install
        echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
    fi
fi

echo ""
echo "================================================"
echo -e "${GREEN}  Installation Complete! 🎉${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
if [ "$install_type" = "1" ]; then
    echo "  1. Run tests:"
    echo "     pytest --cov=src --cov-report=term"
    echo ""
    echo "  2. Start Redis (for integration tests):"
    echo "     docker run -d -p 6379:6379 redis:7-alpine"
    echo ""
    echo "  3. Run the server:"
    echo "     fastmcp dev src/server.py"
else
    echo "  1. Set up environment:"
    echo "     cp .env.example .env"
    echo "     # Edit .env with your configuration"
    echo ""
    echo "  2. Start Redis (optional):"
    echo "     docker run -d -p 6379:6379 redis:7-alpine"
    echo ""
    echo "  3. Run the server:"
    echo "     fastmcp run fastmcp.json"
fi
echo ""
echo "For more information, see:"
echo "  - README.md"
echo "  - docs/SDD.md"
echo "  - TESTING_QUICKSTART.md"
echo ""
