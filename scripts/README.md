# Seraph MCP Scripts

Utility scripts for development, testing, and deployment of Seraph MCP.

## 📋 Available Scripts

### Setup Scripts

#### `setup-pre-commit.sh`

Installs and configures pre-commit hooks for automatic code quality checks.

```bash
./scripts/setup-pre-commit.sh
```

**What it does:**

- Installs dev dependencies (includes pre-commit)
- Configures pre-commit hooks
- Runs initial validation on all files
- Shows usage instructions

**Pre-commit hooks include:**

- Ruff linting and formatting
- MyPy type checking
- Security scanning (Bandit)
- Secret detection
- File format validation (JSON, YAML, TOML)
- Markdown linting

## 🚀 Usage Examples

### First-Time Setup

```bash
# Clone repository
git clone https://github.com/coderdayton/seraph-mcp.git
cd seraph-mcp

# Install dependencies
uv sync --extra dev

# Set up pre-commit hooks
./scripts/setup-pre-commit.sh
```

### Daily Development

Pre-commit hooks run automatically on `git commit`:

```bash
git add .
git commit -m "feat: Add new feature"
# Hooks run automatically ✨
```

### Manual Hook Execution

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files
uv run pre-commit run mypy --all-files

# Update hooks to latest versions
uv run pre-commit autoupdate
```

### Skip Hooks (Emergency Only)

```bash
# Not recommended - bypasses all quality checks
git commit --no-verify
```

## 🔧 Troubleshooting

### Pre-commit Installation Fails

```bash
# Ensure uv is installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync --extra dev

# Retry setup
./scripts/setup-pre-commit.sh
```

### Hooks Fail on Commit

1. **Check the error message** - Most issues are auto-fixable
2. **Review changes** - Run `git diff` to see what was fixed
3. **Run manually** - `uv run pre-commit run --all-files`
4. **Add fixes and retry** - `git add . && git commit`

### Hook Performance Issues

Pre-commit caches results. To clear cache:

```bash
uv run pre-commit clean
```

## 📚 Script Development

### Adding New Scripts

1. Create script in `scripts/` directory
2. Make executable: `chmod +x scripts/your-script.sh`
3. Add shebang: `#!/usr/bin/env bash`
4. Document in this README

### Script Guidelines

- **Use bash or Python** - Keep dependencies minimal
- **Add error handling** - Use `set -e` in bash scripts
- **Document usage** - Add help text and comments
- **Test thoroughly** - Run in clean environment
- **Update README** - Document new scripts here

## 🎯 Future Scripts

Planned utility scripts:

- [ ] `run-tests.sh` - Comprehensive test runner
- [ ] `benchmark.sh` - Performance benchmarking
- [ ] `deploy.sh` - Deployment automation
- [ ] `generate-docs.sh` - Documentation generation
- [ ] `validate-config.sh` - Configuration validation
- [ ] `clean.sh` - Clean build artifacts and caches

## 💡 Tips

1. **Run pre-commit before pushing** - Catch issues early
2. **Update hooks regularly** - `uv run pre-commit autoupdate`
3. **Check CI logs** - Understand automated checks
4. **Use `--all-files` sparingly** - Runs on entire codebase

## 🔗 Related Documentation

- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [.pre-commit-config.yaml](../.pre-commit-config.yaml) - Hook configuration
- [docs/SDD.md](../docs/SDD.md) - System design decisions

---

For questions or issues with scripts, open an issue on GitHub.
