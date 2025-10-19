# Publishing Seraph MCP to PyPI

This guide walks you through publishing Seraph MCP to PyPI so users can install it with `uvx seraph-mcp` or `pip install seraph-mcp`.

## Prerequisites

1. **PyPI Account**: Create accounts at:
   - https://pypi.org (production)
   - https://test.pypi.org (testing)

2. **API Token**: Generate tokens from account settings:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/

   Store tokens securely (you'll need them for `uv publish`).

3. **UV installed**: Ensure you have uv installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Pre-Publishing Checklist

Before publishing, verify these items:

- [ ] Version number updated in `pyproject.toml`
- [ ] CHANGELOG.md updated with release notes
- [ ] All tests passing: `uv run pytest`
- [ ] README.md accurate and complete
- [ ] LICENSE file present
- [ ] Repository URL correct in `pyproject.toml`
- [ ] Dependencies properly specified
- [ ] No sensitive data in source code

## Step 1: Prepare the Package

### Update Version Number

Edit `pyproject.toml`:

```toml
[project]
name = "seraph-mcp"
version = "1.0.1"  # Increment version
```

Follow semantic versioning:
- **Major** (1.0.0 → 2.0.0): Breaking changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes

### Clean Build Artifacts

```bash
# Remove old build artifacts
rm -rf dist/ build/ *.egg-info

# Ensure clean working directory
git status
```

## Step 2: Build the Package

```bash
# Build distribution packages
uv build

# This creates:
# - dist/src-1.0.0-py3-none-any.whl (wheel)
# - dist/seraph-mcp-1.0.0.tar.gz (source distribution)
```

Verify the build:
```bash
ls -lh dist/
```

## Step 3: Test on TestPyPI (Recommended)

Always test on TestPyPI before publishing to production PyPI.

### Publish to TestPyPI

```bash
# Publish to TestPyPI
uv publish --publish-url https://test.pypi.org/legacy/

# When prompted, enter your TestPyPI API token
# Username: __token__
# Password: pypi-AgE... (your token)
```

Or use environment variables:
```bash
export UV_PUBLISH_TOKEN="pypi-AgE..."
uv publish --publish-url https://test.pypi.org/legacy/
```

### Test Installation from TestPyPI

```bash
# Create a test environment
uv venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            seraph-mcp

# Test the command
seraph-mcp --help

# Test with uvx
uvx --from seraph-mcp seraph-mcp --help

# Cleanup
deactivate
rm -rf test-env
```

## Step 4: Publish to PyPI (Production)

Once TestPyPI installation works:

```bash
# Publish to production PyPI
uv publish

# When prompted:
# Username: __token__
# Password: pypi-AgE... (your production PyPI token)
```

Or with environment variable:
```bash
export UV_PUBLISH_TOKEN="pypi-AgE..."
uv publish
```

## Step 5: Verify Production Installation

```bash
# Install from PyPI
pip install seraph-mcp

# Or use uvx (no install needed)
uvx seraph-mcp

# Verify version
python -c "import importlib.metadata; print(importlib.metadata.version('seraph-mcp'))"
```

## Step 6: Tag the Release

After successful publication:

```bash
# Create git tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tag to GitHub
git push origin v1.0.0

# Create GitHub Release (optional)
# Go to: https://github.com/coderdayton/seraph-mcp/releases/new
# - Select tag: v1.0.0
# - Add release notes from CHANGELOG.md
# - Attach dist/ files (optional)
```

## Usage After Publishing

Users can now install and use Seraph MCP easily:

### Direct Installation
```bash
pip install seraph-mcp
```

### Using uvx (Recommended)
```bash
# Run without installing
uvx seraph-mcp

# Or in Claude Desktop config:
{
  "mcpServers": {
    "seraph": {
      "command": "uvx",
      "args": ["seraph-mcp"],
      "env": {
        "REDIS_URL": "redis://localhost:6379/0"
      }
    }
  }
}
```

## Troubleshooting

### "Package already exists"
If you need to republish the same version:
- Increment the version number (even for patches: 1.0.0 → 1.0.1)
- PyPI doesn't allow overwriting existing versions

### Authentication Failed
```bash
# Verify token is correct
echo $UV_PUBLISH_TOKEN

# Or pass token directly
uv publish --token pypi-AgE...
```

### Missing Dependencies
If users report import errors:
- Check `dependencies` in `pyproject.toml`
- Ensure all required packages are listed
- Test in a clean environment

### Wrong Files Included
```bash
# Check what's being included
tar -tzf dist/seraph-mcp-1.0.0.tar.gz

# Exclude files with .gitignore or MANIFEST.in if needed
```

## Automated Publishing with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Build package
        run: uv build

      - name: Publish to PyPI
        env:
          UV_PUBLISH_TOKEN: ${{ secrets.PYPI_TOKEN }}
        run: uv publish
```

Add `PYPI_TOKEN` to GitHub repository secrets:
1. Go to Settings → Secrets and variables → Actions
2. Add new secret: `PYPI_TOKEN` = your PyPI API token

## Best Practices

1. **Always test on TestPyPI first**
2. **Use semantic versioning**
3. **Keep CHANGELOG.md updated**
4. **Tag releases in git**
5. **Create GitHub Releases for major versions**
6. **Test installation in clean environment**
7. **Document breaking changes clearly**
8. **Don't include secrets or credentials**

## Security

- **Never commit API tokens** to git
- Use environment variables or `.pypirc` (add to `.gitignore`)
- Rotate tokens if compromised
- Use trusted publishing (OIDC) for GitHub Actions

## Support

- PyPI help: https://pypi.org/help/
- UV documentation: https://docs.astral.sh/uv/
- Packaging guide: https://packaging.python.org/

## Quick Reference

```bash
# Complete publishing flow
uv build                                              # Build package
uv publish --publish-url https://test.pypi.org/legacy/  # Test
uv publish                                            # Production
git tag -a v1.0.0 -m "Release 1.0.0"                 # Tag
git push origin v1.0.0                                # Push tag
```
