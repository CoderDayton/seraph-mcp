# Contributing to Seraph MCP

Thank you for your interest in contributing to Seraph MCP! This guide will help you get started with the development workflow.

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Setup Development Environment

1. **Clone the repository**

   ```bash
   git clone https://github.com/coderdayton/seraph-mcp.git
   cd seraph-mcp
   ```

2. **Install dependencies**

   ```bash
   uv sync --extra dev
   ```

3. **Set up pre-commit hooks**

   ```bash
   ./scripts/setup-pre-commit.sh
   ```

   Or manually:

   ```bash
   uv run pre-commit install
   ```

4. **Copy environment configuration**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## 🔧 Development Workflow

### Running the Server

```bash
# Development mode with hot reloading
fastmcp dev

# Production mode
fastmcp run fastmcp.json
```

### Code Quality

We use several tools to maintain code quality. All are run automatically via pre-commit hooks.

#### Linting and Formatting

```bash
# Run ruff linter
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

#### Type Checking

```bash
# Run mypy type checking
uv run mypy src/
```

#### Security Scanning

```bash
# Run bandit security scan
uv run bandit -r src/
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_cache.py

# Run tests matching pattern
uv run pytest -k "test_redis"
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` and check:

- ✅ **Ruff linting** - Code quality and style
- ✅ **Ruff formatting** - Consistent code formatting
- ✅ **MyPy** - Static type checking
- ✅ **Bandit** - Security vulnerability scanning
- ✅ **Secret detection** - Prevent committing secrets
- ✅ **File validation** - JSON, YAML, TOML syntax
- ✅ **Markdown linting** - Documentation quality
- ✅ **Basic checks** - Trailing whitespace, file endings, merge conflicts

#### Manual Pre-commit Usage

```bash
# Run all hooks on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff --all-files

# Update hooks to latest versions
uv run pre-commit autoupdate

# Skip hooks (not recommended)
git commit --no-verify
```

## 📝 Code Style Guidelines

### General Principles

1. **Zero-config first** - Features should work out-of-the-box
2. **Auto-enabling** - Detect configuration and enable features automatically
3. **Clear documentation** - Code should be self-documenting with clear names
4. **Type hints** - All functions should have type annotations
5. **Error handling** - Use proper exception chaining with `from e`

### Code Standards

- **Line length**: 120 characters
- **Indentation**: 4 spaces (never tabs)
- **Imports**: Organized by `ruff` (stdlib, third-party, local)
- **Naming**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private: `_leading_underscore`

### Example Code Structure

```python
"""Module docstring explaining purpose."""

from typing import Any

from pydantic import BaseModel

from src.cache import CacheInterface
from src.config import Config


class MyFeature:
    """
    Brief description of the class.

    Longer description if needed, explaining the purpose,
    use cases, and any important details.
    """

    def __init__(self, config: Config) -> None:
        """Initialize with configuration."""
        self.config = config

    async def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Process data and return result.

        Args:
            data: Input data dictionary

        Returns:
            Processed data dictionary

        Raises:
            ValueError: If data is invalid
        """
        try:
            # Process logic here
            return {"result": "processed"}
        except Exception as e:
            raise ValueError("Processing failed") from e
```

## 🏗️ Architecture Guidelines

### Zero-Config Philosophy

Seraph MCP operates on a zero-config principle:

1. **Works without configuration** - Memory cache and Seraph compression by default
2. **Auto-enables features** - Redis cache when `REDIS_URL` is set
3. **Progressive enhancement** - Hybrid compression when provider configured
4. **No required fields** - All configuration is optional

### Adding New Features

When adding a new feature:

1. **Make it optional** - Should not break zero-config operation
2. **Auto-detect** - Enable automatically when relevant config is present
3. **Document clearly** - Update `.env.example` and README.md
4. **Add tests** - Cover zero-config and enabled scenarios
5. **Update SDD.md** - Document architecture decisions

### File Organization

```
seraph-mcp/
├── src/
│   ├── cache/              # Caching system
│   ├── compression/        # Seraph compression
│   ├── context_optimization/ # Hybrid compression
│   ├── providers/          # AI provider integrations
│   ├── config/             # Configuration management
│   └── server.py           # Main MCP server
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
├── examples/              # Usage examples
└── scripts/               # Utility scripts
```

## 🐛 Reporting Issues

When reporting issues, please include:

1. **Clear description** - What happened vs. what you expected
2. **Reproduction steps** - Minimal code to reproduce the issue
3. **Environment** - Python version, OS, relevant configuration
4. **Logs/errors** - Full error messages and stack traces

## 🎯 Pull Request Process

1. **Create a branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure quality**

   ```bash
   # Run tests
   uv run pytest

   # Check linting
   uv run ruff check .

   # Check types
   uv run mypy src/

   # Run pre-commit
   uv run pre-commit run --all-files
   ```

4. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: Add awesome feature"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `refactor:` - Code refactoring
   - `test:` - Test updates
   - `chore:` - Maintenance tasks

5. **Push and create PR**

   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a pull request on GitHub with:
   - Clear description of changes
   - Link to related issues
   - Screenshots (if UI changes)

6. **Address review feedback**
   - Respond to comments
   - Make requested changes
   - Keep commits focused

## 📚 Additional Resources

- **Documentation**: See `docs/` directory and `CLAUDE.md`
- **Architecture**: Read `docs/SDD.md` for system design decisions
- **Examples**: Check `examples/` directory for usage patterns
- **CI/CD**: Review `.github/workflows/ci.yml` for automated checks

## 💡 Tips for Contributors

1. **Start small** - Begin with documentation fixes or minor improvements
2. **Ask questions** - Open an issue for discussion before major changes
3. **Test thoroughly** - Include unit and integration tests
4. **Document changes** - Update relevant docs and add inline comments
5. **Be patient** - Reviews may take time, but we appreciate your contribution!

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Documentation**: Check docs first for common questions

---

Thank you for contributing to Seraph MCP! Your efforts help make AI optimization more accessible to everyone. 🚀
