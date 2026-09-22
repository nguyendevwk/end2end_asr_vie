# Contributing to Vietnamese Voice Agent

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/nguyendevwk/end2end_asr_vie.git
cd end2end_asr_vie/src

# Install dependencies
uv sync

# Install dev dependencies (included by default)
uv sync

# Run tests
uv run pytest tests/ -m "not slow and not gpu"
```

## Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and [mypy](https://mypy-lang.org/) for type checking.

```bash
# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Type check
uv run mypy src/voice_agent/
```

### Guidelines

- **Type hints** are required on all functions
- **Docstrings** required for public classes and methods (Google style)
- **Async/await** for all I/O-bound operations
- **Structured logging** via `structlog` (not `print` or `logging.info`)
- **Custom exceptions** from the hierarchy in `core/exceptions.py`
- **Protocol-based** service interfaces in `core/types.py`

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

# Examples
feat: add voice cloning support
fix: resolve VAD race condition on disconnect
perf: optimize ASR streaming chunk processing
test: add integration tests for TTS streaming
docs: update API reference
```

Types: `feat`, `fix`, `perf`, `test`, `docs`, `refactor`, `chore`, `ci`

## Pull Request Process

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add or update tests
5. Ensure all checks pass:
   ```bash
   uv run ruff check src/
   uv run mypy src/voice_agent/
   uv run pytest tests/ -m "not slow and not gpu"
   ```
6. Submit a pull request with a clear description

### PR Requirements

- One logical change per PR
- Clear description of what and why
- Tests for new functionality
- No type checking or linting errors

## Adding a New Service

1. Define a Protocol in `core/types.py`
2. Implement the service in `services/`
3. Add configuration fields in `config.py`
4. Register in the Orchestrator
5. Add tests in `tests/`
6. Update documentation

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Include your environment info (Python version, GPU, OS)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
