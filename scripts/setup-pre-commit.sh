#!/usr/bin/env bash
#
# Setup script for pre-commit hooks
# This script installs and configures pre-commit hooks for Seraph MCP

set -e

echo "🔧 Setting up pre-commit hooks for Seraph MCP..."
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must be run from the project root directory"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dev dependencies (includes pre-commit)
echo "📦 Installing dev dependencies..."
uv sync --extra dev

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
uv run pre-commit install

# Optional: Install hooks for commit messages
echo "📝 Installing commit message hooks..."
uv run pre-commit install --hook-type commit-msg || true

# Run pre-commit on all files to ensure everything is set up correctly
echo ""
echo "🧪 Running pre-commit on all files (this may take a minute)..."
uv run pre-commit run --all-files || {
    echo ""
    echo "⚠️  Pre-commit found some issues, but they may have been auto-fixed."
    echo "Check the output above and run 'git diff' to see changes."
    echo ""
}

echo ""
echo "✅ Pre-commit hooks installed successfully!"
echo ""
echo "📖 Usage:"
echo "  - Hooks will run automatically on 'git commit'"
echo "  - Run manually: uv run pre-commit run --all-files"
echo "  - Update hooks: uv run pre-commit autoupdate"
echo "  - Skip hooks (not recommended): git commit --no-verify"
echo ""
echo "🎯 Configured hooks:"
echo "  ✓ Ruff linting and formatting"
echo "  ✓ MyPy type checking"
echo "  ✓ Security scanning (Bandit)"
echo "  ✓ Secret detection"
echo "  ✓ File format validation (JSON, YAML, TOML)"
echo "  ✓ Markdown linting"
echo ""
